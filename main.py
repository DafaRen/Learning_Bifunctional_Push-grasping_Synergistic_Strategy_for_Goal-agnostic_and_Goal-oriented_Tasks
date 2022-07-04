#!/usr/bin/env python

import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import sys
from scipy.ndimage.morphology import binary_dilation
import cv2
from collections import namedtuple
import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
from lwrf_infer import LwrfInfer

def main(args):


    # --------------- Setup options ---------------
    is_sim = args.is_sim # Run in simulation?
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj if is_sim else None # Number of objects to add to simulation
    tcp_host_ip = args.tcp_host_ip if not is_sim else None # IP and port to robot arm as TCP client (UR5)
    tcp_port = args.tcp_port if not is_sim else None
    rtc_host_ip = args.rtc_host_ip if not is_sim else None # IP and port to robot arm as real-time client (UR5)
    rtc_port = args.rtc_port if not is_sim else None
    max_push_episode_length = 5
    config_file = args.config_file
    
    if is_sim:
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        workspace_limits = np.asarray([[0.3,0.748], [-0.448,-0.0], [-0.04, 0.2]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- Algorithm options -------------
    method = args.method # 'reactive' (supervised learning) or 'reinforcement' (reinforcement learning ie Q-learning)
    push_rewards = args.push_rewards if method == 'reinforcement' else None  # Use immediate rewards (from change detection) for pushing?
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay # Use prioritized experience replay?
    heuristic_bootstrap = args.heuristic_bootstrap # Use handcrafted grasping algorithm when grasping fails too many times in a row?
    explore_rate_decay = args.explore_rate_decay
    grasp_only = args.grasp_only

    # -------------- Testing options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases


    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
    goal_load_snapshot = args.goal_load_snapshot
    snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
    goal_snapshot_file = os.path.abspath(args.goal_snapshot_file)  if goal_load_snapshot else None
    continue_logging = args.continue_logging # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True


    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                  is_testing, test_preset_cases, config_file)

    # Initialize trainer
    trainer = Trainer(method, push_rewards, future_reward_discount,
                      is_testing, load_snapshot, goal_load_snapshot, snapshot_file, goal_snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Define light weight refinenet model
    lwrf_model = LwrfInfer(use_cuda=trainer.use_cuda, save_path=logger.lwrf_results_directory)

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = [2, 2] if not is_testing else [0, 0]
    grasp_fail_count = [0]
    motion_fail_count = [0]
    explore_prob = 0 if not is_testing else 0.0
    explore_prob1 = 0.15 if not is_testing else 0.0
    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action' : False,
                          'primitive_action' : None,
                          'best_pix_ind' : None,
                          'push_success' : False,
                          'push_step' : 0,
                          'margin_occupy_ratio': None,
                          'margin_occupy_norm': None,
                          'grasp_success' : False,
                          'target_grasped': False}


    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def k_largest_index_argsort(a, k): 
        idx = np.argsort(a.ravel())[:-k-1:-1] 
        return np.column_stack(np.unravel_index(idx, a.shape))
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:

                # Determine whether grasping or pushing should be executed based on network predictions
                best_push_conf = np.max(push_predictions)
                best_grasp_conf = np.max(grasp_predictions)
                print('Primitive confidence scores: %f (push), %f (grasp)' % (best_push_conf, best_grasp_conf))
                nonlocal_variables['primitive_action'] = 'grasp'
                explore_actions = False
                if not grasp_only:
                    print('margin_occupy_ratio : %f' % nonlocal_variables['margin_occupy_ratio'])
                    if (best_push_conf > best_grasp_conf and nonlocal_variables['push_step'] < max_push_episode_length) or (nonlocal_variables['push_step'] < max_push_episode_length and best_grasp_conf < 1.7 and nonlocal_variables['margin_occupy_ratio']>0.1): 
                        nonlocal_variables['primitive_action'] = 'push'
                    explore_actions = np.random.uniform() < explore_prob
                    if explore_actions: # Exploitation (do best action) vs exploration (do other action)
                        print('Strategy: explore (exploration probability: %f)' % (explore_prob))
                        nonlocal_variables['primitive_action'] = 'push' if np.random.randint(0,2) == 0 else 'grasp'
                    else:
                        print('Strategy: exploit (exploration probability: %f)' % (explore_prob))
                trainer.is_exploit_log.append([0 if explore_actions else 1])
                logger.write_to_log('is-exploit', trainer.is_exploit_log)
                
                temp_iteration = trainer.iteration + 1
                # If heuristic bootstrapping is enabled: if change has not been detected more than 2 times, execute heuristic algorithm to detect grasps/pushes
                # NOTE: typically not necessary and can reduce final performance.
                if heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'push' and no_change_count[0] >= 2:
                    print('Change not detected for more than two pushes. Running heuristic pushing.')
                    nonlocal_variables['best_pix_ind'] = trainer.push_heuristic(valid_depth_heightmap)
                    no_change_count[0] = 0
                    predicted_value = push_predictions[nonlocal_variables['best_pix_ind']]
                    use_heuristic = True
                elif heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'grasp' and no_change_count[1] >= 2:
                    print('Change not detected for more than two grasps. Running heuristic grasping.')
                    nonlocal_variables['best_pix_ind'] = trainer.grasp_heuristic(valid_depth_heightmap)
                    no_change_count[1] = 0
                    predicted_value = grasp_predictions[nonlocal_variables['best_pix_ind']]
                    use_heuristic = True
                else:
                    use_heuristic = False

                    if nonlocal_variables['primitive_action'] == 'push':
                        
                        nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
                        predicted_value = np.max(push_predictions)
                        if trainer.iteration > temp_iteration:
                            random_topk=np.random.randint(0,4)
                            cont_x=np.array(prev_best_pix_ind)
                            cont_y=np.array(nonlocal_variables['best_pix_ind'])
                            if np.random.uniform() < explore_prob1:
                                k_largest_index_argsort(push_predictions, k=10)
                                rand=np.random.randint(0,10)
                                print('random-topk')
                                nonlocal_variables['best_pix_ind'] = tuple(k_largest_index_argsort(push_predictions, k=10)[rand])
                                predicted_value =push_predictions[nonlocal_variables['best_pix_ind'][0]][nonlocal_variables['best_pix_ind'][1]][nonlocal_variables['best_pix_ind'][2]]
                            if (cont_x==cont_y).all() and random_topk==0:
                                print('random-out')
                                nonlocal_variables['primitive_action'] = 'grasp'
                                nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
                                predicted_value = np.max(grasp_predictions)
                            elif (cont_x==cont_y).all() and random_topk!=0:
                                k_largest_index_argsort(push_predictions, k=10)
                                rand=np.random.randint(1,11)
                                print('error-topk')
                                nonlocal_variables['best_pix_ind'] = tuple(k_largest_index_argsort(push_predictions, k=11)[rand])
                                predicted_value =push_predictions[nonlocal_variables['best_pix_ind'][0]][nonlocal_variables['best_pix_ind'][1]][nonlocal_variables['best_pix_ind'][2]]
                    elif nonlocal_variables['primitive_action'] == 'grasp':
                        nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
                        predicted_value = np.max(grasp_predictions)
                        if trainer.iteration > temp_iteration:
                            cont_x=np.array(prev_best_pix_ind)
                            cont_y=np.array(nonlocal_variables['best_pix_ind'])
                            random_topk=np.random.randint(0,4)
                            if np.random.uniform() < explore_prob1:
                                print('random-topk')
                                k_largest_index_argsort(grasp_predictions, k=10)
                                rand=np.random.randint(0,10)
                                nonlocal_variables['best_pix_ind'] = tuple(k_largest_index_argsort(grasp_predictions, k=10)[rand])
                                predicted_value =grasp_predictions[nonlocal_variables['best_pix_ind'][0]][nonlocal_variables['best_pix_ind'][1]][nonlocal_variables['best_pix_ind'][2]]
                            if (cont_x==cont_y).all() and random_topk==0:
                                print('random-out')
                                nonlocal_variables['primitive_action'] = 'push'
                                nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
                                predicted_value = np.max(push_predictions)
                            elif (cont_x==cont_y).all() and random_topk!=0:
                                k_largest_index_argsort(grasp_predictions, k=10)
                                print('error-topk')
                                rand=np.random.randint(1,11)
                                nonlocal_variables['best_pix_ind'] = tuple(k_largest_index_argsort(grasp_predictions, k=11)[rand])
                                predicted_value =grasp_predictions[nonlocal_variables['best_pix_ind'][0]][nonlocal_variables['best_pix_ind'][1]][nonlocal_variables['best_pix_ind'][2]]        
                if trainer.iteration> max(200,temp_iteration) and grasp_fail_count[0] ==2:
                    
                    if prev_primitive_action=='grasp' and prev_grasp_success==False:
                        nonlocal_variables['primitive_action'] = 'push'
                        nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
                        predicted_value = np.max(push_predictions) 
                trainer.use_heuristic_log.append([1 if use_heuristic else 0])
                logger.write_to_log('use-heuristic', trainer.use_heuristic_log)

                # Save predicted confidence value
                trainer.goal_predicted_value_log.append([predicted_value])
                logger.write_to_log('goal-predicted-value', trainer.goal_predicted_value_log)

                if nonlocal_variables['primitive_action'] == 'grasp':
                    predicted_value_feat = grasp_predictions_feat[nonlocal_variables['best_pix_ind'][0]][nonlocal_variables['best_pix_ind'][1]][nonlocal_variables['best_pix_ind'][2]]
                elif nonlocal_variables['primitive_action'] == 'push':
                    predicted_value_feat = push_predictions_feat[nonlocal_variables['best_pix_ind'][0]][nonlocal_variables['best_pix_ind'][1]][nonlocal_variables['best_pix_ind'][2]]
                trainer.predicted_value_log.append([predicted_value_feat])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)
                
                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0]*(360.0/trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]
                primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]

                # If pushing, adjust start position, and make sure z value is safe and not too low
                
                if nonlocal_variables['primitive_action'] == 'push': # or nonlocal_variables['primitive_action'] == 'place':
                    # simulation parameter
                    finger_width = 0.02
                    safe_kernel_width = int(np.round((finger_width/2)/heightmap_resolution))
                    local_region = valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]),
                                                         max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
                    if local_region.size == 0:
                        safe_z_position = workspace_limits[2][0] - 0.01
                    else:
                        safe_z_position = np.max_z_position = 0.5 * np.max(local_region) + workspace_limits[2][0] - 0.01
                    primitive_position[2] = safe_z_position
                    print('3D z position:', primitive_position[2])

                # Save executed primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    trainer.executed_action_log.append([0, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 0 - push
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 1 - grasp
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    push_pred_vis = trainer.get_prediction_vis(push_predictions_feat, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                    cv2.imwrite('push.png', push_pred_vis)
                    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions_feat, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('grasp.png', grasp_pred_vis)
                    
                    goal_push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, goal_push_pred_vis, 'goal_push')
                    cv2.imwrite('goal_push.png', goal_push_pred_vis)
                    goal_grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, goal_grasp_pred_vis, 'goal_grasp')
                    cv2.imwrite('goal_grasp.png', goal_grasp_pred_vis)
                # Initialize variables that influence reward
                nonlocal_variables['push_success'] = False
                nonlocal_variables['grasp_success'] = False
                nonlocal_variables['target_grasped'] = False
                change_detected = False
                
                motion_fail_count[0] += 1
                
                
                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    robot.push(primitive_position, best_rotation_angle, workspace_limits)
                    nonlocal_variables['push_step'] += 1
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    
                    grasp_fail_count[0] += 1
                    grasped_object_name = robot.grasp(primitive_position, best_rotation_angle, workspace_limits)
                    if grasped_object_name in segment_results['labels']:
                        nonlocal_variables['grasp_success'] = True
                        print('Grasping succeed, grasped', grasped_object_name)
                        nonlocal_variables['target_grasped'] = grasped_object_name == target_name
                        print('Target grasped?:', nonlocal_variables['target_grasped'])
                        if nonlocal_variables['target_grasped']:
                            motion_fail_count[0] = 0
                            grasp_fail_count[0] = 0
                            nonlocal_variables['push_step'] = 0
                        else:
                            # posthoc labeling for data augmentation
                            augment_id = segment_results['labels'].index(grasped_object_name)
                            augment_mask_heightmap = seg_mask_heightmaps[:, :, augment_id]
# =============================================================================
#                             logger.save_augment_masks(trainer.iteration, augment_mask_heightmap)
#                             trainer.augment_ids.append(trainer.iteration)
#                             logger.write_to_log('augment-ids', trainer.augment_ids)
# =============================================================================
                    else:
                        print('Grasping failed')
                trainer.target_grasped_log.append(int(nonlocal_variables['target_grasped']))
                logger.write_to_log('target-grasped', trainer.target_grasped_log)

                nonlocal_variables['executing_action'] = False

            time.sleep(0.01)
    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    target_name = None

    # Start main training/testing loop
    while True:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim: robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration
        
        # Use lwrf to segment/detect target object
        segment_results = lwrf_model.segment(color_img)

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap, seg_mask_heightmaps = utils.get_heightmap(
            color_img, depth_img, segment_results['masks'], robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        mask_heightmaps = utils.process_mask_heightmaps(segment_results, seg_mask_heightmaps)
        
        if len(mask_heightmaps['names']) == 0:
            
            target_name = None
            robot.restart_sim()
            robot.add_objects()
            grasp_fail_count[0] = 0
            motion_fail_count[0] = 0
            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            #if is_testing: # If at end of test run, re-load original weights (before test run)
            #    trainer.model.load_state_dict(torch.load(snapshot_file))
            continue
        # Choose target
        if len(mask_heightmaps['names']) == 0 and is_testing:            
            target_mask_heightmap = np.ones_like(valid_depth_heightmap)
        else:
            # lwrf_model.display_instances(title=str(trainer.iteration))
            if target_name in mask_heightmaps['names']:
                target_mask_heightmap = mask_heightmaps['heightmaps'][mask_heightmaps['names'].index(target_name)]
            else:
                target_id = random.randint(0, len(mask_heightmaps['names'])-1)
                target_name = mask_heightmaps['names'][target_id]
                target_mask_heightmap = mask_heightmaps['heightmaps'][target_id]
            print('lwrf segments:', mask_heightmaps['names'])
            print('Target name:', target_name)

            nonlocal_variables['margin_occupy_ratio'], nonlocal_variables['margin_occupy_norm'] = utils.check_grasp_margin(target_mask_heightmap, depth_heightmap)


        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, target_mask_heightmap)

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        stuff_count_one = np.zeros(valid_depth_heightmap.shape)
        struct2 = sc.ndimage.generate_binary_structure(2, 2)
        stuff_count_one[valid_depth_heightmap > 0.02] = 1
        stuff_count_one = binary_dilation(stuff_count_one,structure=struct2, iterations=15).astype(np.float32)
        stuff_count_push = np.zeros(valid_depth_heightmap.shape)
        stuff_count_push[valid_depth_heightmap > 0.02] = 1
        stuff_count_push = binary_dilation(stuff_count_push,structure=struct2, iterations=5).astype(np.float32)
        empty_threshold = 300
        if is_sim and is_testing:
            empty_threshold = 10
        if nonlocal_variables['push_step'] == max_push_episode_length + 1 or np.sum(stuff_count) < empty_threshold or (is_sim and no_change_count[0] + no_change_count[1] > 10):
            no_change_count = [0, 0]
            nonlocal_variables['push_step'] = 0 
            if is_sim:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                robot.restart_sim()
                robot.add_objects()
                grasp_fail_count[0] = 0
                motion_fail_count[0] = 0
                if is_testing: # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))
                continue
            else:
                # print('Not enough stuff on the table (value: %d)! Pausing for 30 seconds.' % (np.sum(stuff_count)))
                # time.sleep(30)
                print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (np.sum(stuff_count)))
                robot.restart_real()

            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True # Exit after training thread (backprop and saving labels)
            continue

        if not exit_called:

            # Run forward pass with network to get affordances
            push_predictions_feat, grasp_predictions_feat, state_feat = trainer.forward(color_heightmap, valid_depth_heightmap, is_volatile=True)
            
            #if trainer.iteration<500 and np.random.randint(0,10)<8:
            grasp_predictions_feat = grasp_predictions_feat * stuff_count
            push_predictions_feat = push_predictions_feat * stuff_count_one
            
            #goal_push_predictions, goal_grasp_predictions, goal_state_feat = trainer.goal_forward(grasp_predictions_feat, push_predictions_feat, target_mask_heightmap, is_volatile=True)
            # Execute best primitive action on robot in another thread

            
            grasp_predictions = grasp_predictions_feat * target_mask_heightmap
            push_predictions = push_predictions_feat * binary_dilation(target_mask_heightmap, iterations=15).astype(np.float32)
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():
            
            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            change_threshold = 300
            change_value = np.sum(depth_diff)
            change_detected = change_value > change_threshold or prev_grasp_success
            
            struct2 = sc.ndimage.generate_binary_structure(2, 2)
            
            
            prev_depth_heightmap[np.isnan(prev_depth_heightmap)] = 0
            prev_stuff_count_push = np.zeros(prev_depth_heightmap.shape)
            prev_stuff_count_push[prev_depth_heightmap > 0.3] = 0
            prev_stuff_count_push[prev_depth_heightmap > 0.02] = 1
            prev_stuff_count_push = binary_dilation(prev_stuff_count_push,structure=struct2, iterations=5).astype(np.float32)
            depth_vanish=np.sum(prev_stuff_count_push)-np.sum(stuff_count_push)
            
            if prev_primitive_action == 'push' and depth_vanish < -100:
                change_detected = True

            print('Change detected: %r (change_value: %d)' % (change_detected, change_value))
            print('Change detected1: %r (depth_vanish: %d)' % (change_detected, depth_vanish))
            
            margin_increased = False
            
            # Detect push changes
            if not prev_target_grasped:
                margin_increase_threshold = 0.1
                margin_increase_val = prev_margin_occupy_ratio-nonlocal_variables['margin_occupy_ratio']
                margin_change_threshold =50
                margin_increase = prev_margin_occupy_norm-nonlocal_variables['margin_occupy_norm']
                print('Grasp margin increased: (value: %f = %f - %f)' % (margin_increase_val, prev_margin_occupy_ratio,nonlocal_variables['margin_occupy_ratio']))
                if (margin_increase_val > margin_increase_threshold):
                    margin_increased = True
                    

            push_effective = margin_increased
            
            if change_detected:
                if prev_primitive_action == 'push':
                    no_change_count[0] = 0
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] = 0
            else:
                if prev_primitive_action == 'push':
                    no_change_count[0] += 1
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] += 1

            # Compute training labels
            label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_push_success, prev_grasp_success, change_detected,
                                                                     push_effective, prev_push_predictions, prev_grasp_predictions, color_heightmap, valid_depth_heightmap)
# =============================================================================
#             goal_label_value, goal_prev_reward_value = trainer.get_goal_label_value(prev_primitive_action, push_effective, prev_target_grasped ,
#                                                                      change_detected, prev_push_predictions, prev_grasp_predictions, grasp_predictions_feat, push_predictions_feat, target_mask_heightmap)
# =============================================================================
            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)
            
# =============================================================================
#             trainer.goal_label_value_log.append([goal_label_value])
#             logger.write_to_log('goal-label-value', trainer.goal_label_value_log)
#             trainer.goal_reward_value_log.append([goal_prev_reward_value])
#             logger.write_to_log('goal-reward-value', trainer.goal_reward_value_log)
# =============================================================================
            if trainer.iteration>0:
            # Backpropagate
                trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value)
                
            # Adjust exploration probability
            if not is_testing:
                explore_prob = max(0.2 * np.power(0.9998, trainer.iteration),0.1) if explore_rate_decay else 0.5
                explore_prob1 = max(0.15 * np.power(0.9995, trainer.iteration),0.05) if explore_rate_decay else 0.5
            # Do sampling for experience replay
            if trainer.iteration>0 and experience_replay and not is_testing:
                sample_primitive_action = prev_primitive_action
                if sample_primitive_action == 'push':
                    sample_primitive_action_id = 0
                    sample_reward_value = 0 if prev_reward_value == 0.5 else 0.5
                elif sample_primitive_action == 'grasp':
                    sample_primitive_action_id = 1
                    sample_reward_value = 0 if prev_reward_value == 1 else 1

                # Get samples of the same primitive but with different results
                sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[1:trainer.iteration,0] == sample_reward_value, np.asarray(trainer.executed_action_log)[1:trainer.iteration,0] == sample_primitive_action_id))

                if sample_ind.size > 1:
                    
                    for x in range(2):
                        
                        # Find sample with highest surprise value
                        
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - np.asarray(trainer.label_value_log)[sample_ind[:,0]])
                        sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])
                        sorted_sample_ind = sample_ind[sorted_surprise_ind,0]
                        pow_law_exp = 2
                        rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
                        sample_iteration = sorted_sample_ind[rand_sample_ind]
                        print('Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))
    
                        # Load sample RGB-D heightmap
                        sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.color.png' % (sample_iteration)))
                        sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                        sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.depth.png' % (sample_iteration)), -1)
                        sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000
    
                        # Compute forward pass with sample
                        with torch.no_grad():
                            sample_push_predictions, sample_grasp_predictions, sample_state_feat = trainer.forward(sample_color_heightmap, sample_depth_heightmap, is_volatile=True)
    
                        # Load next sample RGB-D heightmap
                        next_sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.color.png' % (sample_iteration+1)))
                        next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
                        next_sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.depth.png' % (sample_iteration+1)), -1)
                        next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32)/100000
    
                        sample_push_success = sample_reward_value == 0.5
                        sample_grasp_success = sample_reward_value == 1
                        sample_change_detected = sample_push_success
                        # new_sample_label_value, _ = trainer.get_label_value(sample_primitive_action, sample_push_success, sample_grasp_success, sample_change_detected, sample_push_predictions, sample_grasp_predictions, next_sample_color_heightmap, next_sample_depth_heightmap)
    
                        # Get labels for sample and backpropagate
                        sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration,1:4]).astype(int)
                        trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration])
    
                        # Recompute prediction value and label for replay buffer
                        if sample_primitive_action == 'push':
                            trainer.predicted_value_log[sample_iteration] = [np.max(sample_push_predictions)]
                            # trainer.label_value_log[sample_iteration] = [new_sample_label_value]
                        elif sample_primitive_action == 'grasp':
                            trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]
                            # trainer.label_value_log[sample_iteration] = [new_sample_label_value]

                else:
                    print('Not enough prior training samples. Skipping experience replay.')
# =============================================================================
#             # Do sampling for experience replay--goal_grasping
#             if trainer.iteration>0:
#                 sample_primitive_action = prev_primitive_action
#                 if sample_primitive_action == 'push':
#                     sample_primitive_action_id = 0
#                     sample_reward_value = 0 if goal_prev_reward_value == 0.5 else 0.5
#                 elif sample_primitive_action == 'grasp':
#                     sample_primitive_action_id = 1
#                     sample_reward_value = 0 if goal_prev_reward_value == 1 else 1
# 
#                 # Get samples of the same primitive but with different results
#                 sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.goal_reward_value_log)[1:trainer.iteration,0] == sample_reward_value, np.asarray(trainer.executed_action_log)[1:trainer.iteration,0] == sample_primitive_action_id))
# 
#                 if sample_ind.size > 1:
#                     
#                     for x in range(2):
#                         
#                         # Find sample with highest surprise value
#                         
#                         sample_surprise_values = np.abs(np.asarray(trainer.goal_predicted_value_log)[sample_ind[:,0]] - np.asarray(trainer.goal_label_value_log)[sample_ind[:,0]])
#                         sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])
#                         sorted_sample_ind = sample_ind[sorted_surprise_ind,0]
#                         pow_law_exp = 2
#                         rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
#                         sample_iteration = sorted_sample_ind[rand_sample_ind]
#                         print('Goal-Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))
#     
#                         # Load sample RGB-D heightmap
#                         sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.color.png' % (sample_iteration)))
#                         sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
#                         sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.depth.png' % (sample_iteration)), -1)
#                         sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000
#                         replay_mask_heightmap = cv2.imread(os.path.join(logger.target_mask_heightmaps_directory, '%06d.mask.png' % (sample_iteration)), -1)
#                         replay_mask_heightmap = replay_mask_heightmap.astype(np.float32) / 255
#                         # Compute forward pass with sample
#                         with torch.no_grad():
#                             sample_push_predictions, sample_grasp_predictions, sample_state_feat = trainer.forward(sample_color_heightmap, sample_depth_heightmap, is_volatile=True)
#                             goal_push_predictions, goal_grasp_predictions, goal_state_feat = trainer.goal_forward(sample_grasp_predictions, sample_push_predictions, replay_mask_heightmap, is_volatile=True)
#                         # Load next sample RGB-D heightmap
#                         next_sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.color.png' % (sample_iteration+1)))
#                         next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
#                         next_sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.depth.png' % (sample_iteration+1)), -1)
#                         next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32)/100000
#     
#                         sample_push_success = sample_reward_value == 0.5
#                         sample_grasp_success = sample_reward_value == 1
#                         sample_change_detected = sample_push_success
#                         # new_sample_label_value, _ = trainer.get_label_value(sample_primitive_action, sample_push_success, sample_grasp_success, sample_change_detected, sample_push_predictions, sample_grasp_predictions, next_sample_color_heightmap, next_sample_depth_heightmap)
#     
#                         # Get labels for sample and backpropagate
#                         sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration,1:4]).astype(int)
#                         trainer.goal_backprop(sample_grasp_predictions, sample_push_predictions, replay_mask_heightmap, sample_primitive_action, sample_best_pix_ind, trainer.goal_label_value_log[sample_iteration])
#     
#                         # Recompute prediction value and label for replay buffer
#                         if sample_primitive_action == 'push':
#                             trainer.goal_predicted_value_log[sample_iteration] = [np.max(goal_push_predictions)]
#                             # trainer.label_value_log[sample_iteration] = [new_sample_label_value]
#                         elif sample_primitive_action == 'grasp':
#                             trainer.goal_predicted_value_log[sample_iteration] = [np.max(goal_grasp_predictions)]
#                             # trainer.label_value_log[sample_iteration] = [new_sample_label_value]
# 
#                 else:
#                     print('Not enough prior training samples. Skipping experience replay.')
#                 
# =============================================================================
            # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, method)
                
                if trainer.iteration % 100 == 0:
                    logger.save_model(trainer.iteration, trainer.model, method)
                    
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()
                        
                if trainer.iteration % 50 == 0:
                    trainer.model_tg.load_state_dict(trainer.model.state_dict())
        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)
            
        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        
        prev_mask_heightmaps = mask_heightmaps.copy()
        prev_target_mask_heightmap = target_mask_heightmap.copy()
        
        prev_grasp_predictions_feat = grasp_predictions_feat
        prev_push_predictions_feat = push_predictions_feat
         
        prev_push_success = nonlocal_variables['push_success']
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_push_predictions = push_predictions.copy()
        prev_grasp_predictions = grasp_predictions.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']
        prev_target_grasped = nonlocal_variables['target_grasped']
        prev_margin_occupy_ratio = nonlocal_variables['margin_occupy_ratio']
        prev_margin_occupy_norm = nonlocal_variables['margin_occupy_norm']
        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=30,                                help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='58.199.175.44',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30003,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='58.199.175.44',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')
    
    parser.add_argument('--config_file', dest='config_file', action='store', default='simulation/random/random-23blocks.txt')
    
    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement',                               help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)


    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--goal_load_snapshot', dest='goal_load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--goal_snapshot_file', dest='goal_snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
