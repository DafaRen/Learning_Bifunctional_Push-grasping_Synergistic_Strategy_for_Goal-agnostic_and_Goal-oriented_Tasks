# Learning Bifunctional Push-grasping Synergistic Strategy for Goal-agnostic and Goal-oriented Tasks

Video:  [YouTube](https://youtu.be/cOud5iLoJPk) 

Dafa Ren, Shuang Wu, Xiaofan Wang and Xiaoqiang Ren

Both goal-agnostic and goal-oriented tasks have practical value for robotic grasping: goal-agnostic tasks target all objects in the workspace, while goal-oriented tasks aim at grasping pre-assigned goal objects. However, most current grasping methods are only better at coping with one task. In this work, we propose a bifunctional push-grasping synergistic strategy for goal-agnostic and goal-oriented grasping tasks. Our method integrates pushing along with grasping to pick up all objects or pre-assigned goal objects with high action efficiency depending on the task requirement. We introduce a bifunctional network, which takes in visual observations and outputs dense pixel-wise maps of Q values for pushing and grasping primitive actions, to increase the available samples in the action space. Then we propose a hierarchical reinforcement learning framework to coordinate the two tasks by considering the goal-agnostic task as a combination of multiple goal-oriented tasks. To reduce the training difficulty of the hierarchical framework, we design a two-stage training method to train the two types of tasks separately. We perform pretraining of the model in simulation, and then transfer the learned model to the real world without any additional realworld fine-tuning. Experimental results show that the proposed approach outperforms existing methods in task completion rate and grasp success rate with less motion number.



## Installation

- Ubuntu 20.04
- Python 3
  - torch>=0.4.0, torchvision
  - numpy, scipy, opencv-python, matplotlib, skimage
- CoppeliaSim
- Cuda 10.1
- GTX 2080 Ti, 12GB memory is tested

## Train

First run CoppeliaSim (navigate to your CoppeliaSim directory and run`./coppeliaSim.sh`) and open the file `simulation/simulation.ttt` from this repository. Then download the pre-trained models by running

```python
sh downloads.sh
```

#### 1.  Goal-agnostic task training

In this stage, the system focuses on precise grasping in the goal-agnostic task. Open the file `goal-agnostic task training` and  run the following

```shell
python main.py --is_sim --push_rewards --experience_replay --explore_rate_decay --save_visualizations
```

#### 2.  Goal-oriented Task Training

The goal of this stage is to enable goal-oriented tasks through the synergy of grasping and pre-grasping actions.

```python
python main.py --is_sim --push_rewards --experience_replay --explore_rate_decay --save_visualizations \
    --load_snapshot --snapshot_file 'goal-agnostic task training/logs/YOUR-SESSION-DIRECTORY-NAME-HERE/models/snapshot-backup.reinforcement.pth' \ 
```

To plot the performance of a session over training time, run the following:

```shell
python plot.py 'logs/YOUR-SESSION-DIRECTORY-NAME-HERE'
```

## Test

To test the pre-trained model, simply change the location of `--snapshot_file`:

```python
python main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 10     --push_rewards --experience_replay --explore_rate_decay     --is_testing --test_preset_cases --config_file 'simulation/preset/test-10-obj-06.txt'     --load_snapshot --snapshot_file 'YOUR-SNAPSHOT-FILE-HERE'     --save_visualizations 
```

## Evaluation

We use three metrics to evaluate the performance: completion, grasp success rate, motion number. 

```python
python evaluate.py --session_directory 'YOUR SESSION DIRECTORY' --num_obj_complete 1
```

## Acknowledgment

We use the following code in our project

- [Visual Pushing and Grasping Toolbox][1]

- [Light-Weight RefineNet for Real-Time Semantic Segmentation][2]

[1]: https://github.com/andyzeng/visual-pushing-grasping
[2]: https://github.com/DrSleep/light-weight-refinenet





