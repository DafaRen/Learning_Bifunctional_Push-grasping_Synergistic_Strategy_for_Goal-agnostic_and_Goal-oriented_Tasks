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


def main(args):


    # --------------- Setup options ---------------
    is_sim = args.is_sim # Run in simulation?
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj if is_sim else None # Number of objects to add to simulation
    tcp_host_ip = args.tcp_host_ip if not is_sim else None # IP and port to robot arm as TCP client (UR5)
    tcp_port = args.tcp_port if not is_sim else None
    rtc_host_ip = args.rtc_host_ip if not is_sim else None # IP and port to robot arm as real-time client (UR5)
    rtc_port = args.rtc_port if not is_sim else None
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
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
    continue_logging = args.continue_logging # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True


    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                  is_testing, test_preset_cases, test_preset_file)

    # Initialize trainer
    trainer = Trainer(method, push_rewards, future_reward_discount,
                      is_testing, load_snapshot, snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = [2, 2] if not is_testing else [0, 0]
    explore_prob = 0.4 if not is_testing else 0.0
    explore_prob1 = 0.15 if not is_testing else 0.0
    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action' : False,
                          'primitive_action' : None,
                          'best_pix_ind' : None,
                          'push_success' : False,
                          'no_mask_backprop' : False,
                          'grasp_success' : False}


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
                    if is_testing and method == 'reactive':
                        if best_push_conf > 2*best_grasp_conf:
                            nonlocal_variables['primitive_action'] = 'push'
                    else:
                        if best_push_conf > best_grasp_conf:
                            nonlocal_variables['primitive_action'] = 'push'
                    
                    explore_actions = np.random.uniform() < explore_prob
                    if explore_actions: # Exploitation (do best action) vs exploration (do other action)
                        print('Strategy: explore (exploration probability: %f)' % (explore_prob))
                        nonlocal_variables['primitive_action'] = 'push' if np.random.randint(0,2) == 0 else 'grasp'
                    else:
                        print('Strategy: exploit (exploration probability: %f)' % (explore_prob))
                trainer.is_exploit_log.append([0 if explore_actions else 1])
                logger.write_to_log('is-exploit', trainer.is_exploit_log)
                if continue_logging:
                    temp_iteration = trainer.iteration + 1
                else:
                    temp_iteration = 2
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
                        if trainer.iteration >temp_iteration:
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
                if trainer.iteration> max(1000,temp_iteration)and np.random.randint(0,2)==0:
                    
                    if prev_primitive_action=='grasp' and prev_grasp_success==False:
                        nonlocal_variables['primitive_action'] = 'push'
                        nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
                        predicted_value = np.max(push_predictions) 
                trainer.use_heuristic_log.append([1 if use_heuristic else 0])
                logger.write_to_log('use-heuristic', trainer.use_heuristic_log)

                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0]*(360.0/trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]
                primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]

                # If pushing, adjust start position, and make sure z value is safe and not too low
                if nonlocal_variables['primitive_action'] == 'push': # or nonlocal_variables['primitive_action'] == 'place':
                    finger_width = 0.024
                    safe_kernel_width = int(np.round((finger_width/2)/heightmap_resolution))
                    local_region = valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]), max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
                    if local_region.size == 0:
                        safe_z_position = workspace_limits[2][0]
                    else:
                        safe_z_position = 0.5*np.max(local_region) + workspace_limits[2][0]
                    primitive_position[2] = safe_z_position
                
                # Save executed primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    trainer.executed_action_log.append([0, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 0 - push
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 1 - grasp
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                    cv2.imwrite('visualization.push.png', push_pred_vis)
                    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('visualization.grasp.png', grasp_pred_vis)

                # Initialize variables that influence reward
                nonlocal_variables['push_success'] = False
                nonlocal_variables['grasp_success'] = False
                change_detected = False

                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    nonlocal_variables['push_success'] = robot.push(primitive_position, best_rotation_angle, workspace_limits)
                    print('Push successful: %r' % (nonlocal_variables['push_success']))
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    nonlocal_variables['grasp_success'] = robot.grasp(primitive_position, best_rotation_angle, workspace_limits)
                    print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))

                nonlocal_variables['executing_action'] = False

            time.sleep(0.01)
    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------


    # Start main training/testing loop
    while True:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim: robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')

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
        if np.sum(stuff_count) < empty_threshold or (is_sim and no_change_count[0] + no_change_count[1] > 10):
            no_change_count = [0, 0]
            if is_sim:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                robot.restart_sim()
                robot.add_objects()
                if is_testing: # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))
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
            push_predictions, grasp_predictions, state_feat = trainer.forward(color_heightmap, valid_depth_heightmap, is_volatile=True)
            no_mask_best_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
            #if trainer.iteration<500 and np.random.randint(0,10)<8:
            grasp_predictions = grasp_predictions * stuff_count
            push_predictions = push_predictions * stuff_count_one
            
            mask_best_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
            if not (np.array(no_mask_best_ind) == np.array(mask_best_ind)).all():
                nonlocal_variables['no_mask_backprop'] = True
            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():

            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            change_threshold = 400
            change_value = np.sum(depth_diff)
            change_detected = change_value > change_threshold or prev_grasp_success
            
            struct2 = sc.ndimage.generate_binary_structure(2, 2)
            
            
            prev_depth_heightmap[np.isnan(prev_depth_heightmap)] = 0
            prev_stuff_count_push = np.zeros(prev_depth_heightmap.shape)
            prev_stuff_count_push[prev_depth_heightmap > 0.3] = 0
            prev_stuff_count_push[prev_depth_heightmap > 0.02] = 1
            prev_stuff_count_push = binary_dilation(prev_stuff_count_push,structure=struct2, iterations=5).astype(np.float32)
            depth_vanish=np.sum(prev_stuff_count_push)-np.sum(stuff_count_push)
            if prev_primitive_action == 'push' and depth_vanish >=-30:
                change_detected = False
            if prev_primitive_action == 'push' and depth_vanish < -120:
                change_detected = True
            
            print('Change detected: %r (value: %d)' % (change_detected, change_value))
            print('Change detected1: %r (value: %d)' % (change_detected, depth_vanish))
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
            label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_push_success, prev_grasp_success, change_detected, prev_push_predictions, prev_grasp_predictions, color_heightmap, valid_depth_heightmap)
            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)
            if trainer.iteration>0:
            # Backpropagate
                trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value)
            if nonlocal_variables['no_mask_backprop'] == True and prev_primitive_action == 'grasp':
                print('no_mask_backprop')
                no_msak_label_value = 0
                trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_no_mask_best_ind, no_msak_label_value)
                nonlocal_variables['no_mask_backprop'] = False
            # Adjust exploration probability
            if not is_testing:
                explore_prob = max(0.2 * np.power(0.9998, trainer.iteration),0.1) if explore_rate_decay else 0.5
                explore_prob1 = max(0.15 * np.power(0.9995, trainer.iteration),0.05) if explore_rate_decay else 0.5
            # Do sampling for experience replay
            if trainer.iteration>0 and experience_replay and not is_testing:
                sample_primitive_action = prev_primitive_action
                if sample_primitive_action == 'push':
                    sample_primitive_action_id = 0
                    if method == 'reactive':
                        sample_reward_value = 0 if prev_reward_value == 1 else 1 # random.randint(1, 2) # 2
                    elif method == 'reinforcement':
                        sample_reward_value = 0 if prev_reward_value == 0.5 else 0.5
                elif sample_primitive_action == 'grasp':
                    sample_primitive_action_id = 1
                    if method == 'reactive':
                        sample_reward_value = 0 if prev_reward_value == 1 else 1
                    elif method == 'reinforcement':
                        sample_reward_value = 0 if prev_reward_value == 1 else 1

                # Get samples of the same primitive but with different results
                sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[1:trainer.iteration,0] == sample_reward_value, np.asarray(trainer.executed_action_log)[1:trainer.iteration,0] == sample_primitive_action_id))

                if sample_ind.size > 0:

                    # Find sample with highest surprise value
                    if method == 'reactive':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - (1 - sample_reward_value))
                    elif method == 'reinforcement':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - np.asarray(trainer.label_value_log)[sample_ind[:,0]])
                    sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])
                    sorted_sample_ind = sample_ind[sorted_surprise_ind,0]
                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    print('Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                    # Load sample RGB-D heightmap
                    sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                    sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                    sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000

                    # Compute forward pass with sample
                    with torch.no_grad():
                        sample_push_predictions, sample_grasp_predictions, sample_state_feat = trainer.forward(sample_color_heightmap, sample_depth_heightmap, is_volatile=True)

                    # Load next sample RGB-D heightmap
                    next_sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration+1)))
                    next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    next_sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration+1)), -1)
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
                if sample_ind.size > 0 :

                    # Find sample with highest surprise value
                    if method == 'reactive':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - (1 - sample_reward_value))
                    elif method == 'reinforcement':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - np.asarray(trainer.label_value_log)[sample_ind[:,0]])
                    sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])
                    sorted_sample_ind = sample_ind[sorted_surprise_ind,0]
                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    print('Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                    # Load sample RGB-D heightmap
                    sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                    sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                    sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000

                    # Compute forward pass with sample
                    with torch.no_grad():
                        sample_push_predictions, sample_grasp_predictions, sample_state_feat = trainer.forward(sample_color_heightmap, sample_depth_heightmap, is_volatile=True)

                    # Load next sample RGB-D heightmap
                    next_sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration+1)))
                    next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    next_sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration+1)), -1)
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

            # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, method)
                if trainer.iteration % 50 == 0:
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
        prev_push_success = nonlocal_variables['push_success']
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_push_predictions = push_predictions.copy()
        prev_grasp_predictions = grasp_predictions.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']
        prev_no_mask_best_ind = no_mask_best_ind
        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='58.199.175.44',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30003,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='58.199.175.44',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

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
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
