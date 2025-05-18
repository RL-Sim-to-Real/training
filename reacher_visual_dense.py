import warnings

 
warnings.filterwarnings("ignore")

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere, WrappedEnv
from jsac.helpers.logger import Logger
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent

import time

import argparse
import shutil
import numpy as np


import gymnasium as gym
import franka_genesis
from franka_genesis.FrankaVisionReacher.VisionReacherVelocity import FrankaVisionReacherEnv
import multiprocessing as mp
# import cv2
from utils import visualize_policy

config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    
    'latent_dim': 50,

    'mlp': [1024, 1024],
}

def parse_args():
    parser = argparse.ArgumentParser()
    
    # environment
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default='img_prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")

    parser.add_argument('--task_name', default='reacher_visual_dense', type=str)
    parser.add_argument('--image_height', default=64, type=int)          # Mode: img, img_prop
    parser.add_argument('--image_width', default=64, type=int)          # Mode: img, img_prop     
    parser.add_argument('--image_history', default=0, type=int)          # Mode: img, img_prop
    parser.add_argument('--dt', default=0.15, type=float)
    parser.add_argument('--episode_steps', default=100, type=int) 

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1_000_000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=5_000, type=int)
    parser.add_argument('--env_steps', default=1_000_000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=False, action='store_true')
    parser.add_argument('--global_norm', default=1.0, type=float)
    
    # critic
    parser.add_argument('--critic_lr', default=1e-4, type=float) 
    parser.add_argument('--num_critic_networks', default=5, type=int)
    parser.add_argument('--num_critic_updates', default=1, type=int)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    
    # actor
    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    parser.add_argument('--actor_sync_freq', default=8, type=int)   # Sync mode: False
    
    # encoder
    parser.add_argument('--spatial_softmax', default=False, action='store_true')    # Mode: img, img_prop

    # sac
    parser.add_argument('--temp_lr', default=1e-4, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    
    # misc
    parser.add_argument('--num_cameras', default=1, type=int)
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--log_every', default=1, type=int)
    parser.add_argument('--eval_steps', default=-1, type=int)
    parser.add_argument('--num_eval_episodes', default=0, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, 
                        action='store_true')
    parser.add_argument('--xtick', default=10_000, type=int)
    parser.add_argument('--save_wandb', default=False, action='store_true')

    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=50_000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--start_step', default=0, type=int)
    parser.add_argument('--start_episode', default=0, type=int)

    parser.add_argument('--img_aug_path', default='', type=str)
    parser.add_argument('--buffer_save_path', default='', type=str) # ./buffers/
    parser.add_argument('--buffer_load_path', default='', type=str) # ./buffers/

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    mp.set_start_method('spawn')

    args = parse_args()
    env = FrankaVisionReacherEnv(max_episode_length=500, render_mode="")
    env = WrappedEnv(env, episode_max_steps=args.episode_steps)
    args.name = f'{args.task_name}'
    args.work_dir += f'/results/{args.name}/seed_{args.seed}/'
    
    if os.path.exists(args.work_dir):
        inp = input('The work directory already exists. ' +
                    'Please select one of the following: \n' +  
                    '  1) Press Enter to resume the run.\n' + 
                    '  2) Press X to remove the previous work' + 
                    ' directory and start a new run.\n' + 
                    '  3) Press any other key to exit.\n')
        if inp == 'X' or inp == 'x':
            shutil.rmtree(args.work_dir)
            print('Previous work dir removed.')
        elif inp == '':
            pass
        else:
            exit(0)

    make_dir(args.work_dir)

    if args.buffer_save_path:
        if args.buffer_save_path == ".":
            args.buffer_save_path = os.path.join(args.work_dir, 'buffers')
        make_dir(args.buffer_save_path)

    L = Logger(args.work_dir, args.xtick, vars(args), 
                   args.save_tensorboard, args.save_wandb)
    
    if args.buffer_load_path == ".":
        args.buffer_load_path = os.path.join(args.work_dir, 'buffers')

    args.model_dir = os.path.join(args.work_dir, 'checkpoints') 
    
    if args.save_model:
        make_dir(args.model_dir)
        make_dir(args.work_dir + '/videos')
    

    proprioception_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    env_action_space = env.action_space

    print(f"{proprioception_shape}, {action_shape}, {env_action_space}")
    set_seed_everywhere(seed=args.seed)
    
    args.single_image_shape = (args.image_width, args.image_height, 3)
    args.proprioception_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    args.image_shape = (64, 64, 3)
    args.net_params = config
    agent = SACRADAgent(vars(args))
    state, _ = env.reset()
    print("Env reset")
    first_step = True
    task_start_time = time.time()
    print("Starting training...")
    while env.total_steps < args.env_steps:
        t1 = time.time()
        action = agent.sample_actions(state)
        t2 = time.time()
        next_state, reward, done, info = env.step(action)
        # truncate = env.total_steps % max_episode_length == 0
        # done = done or truncate
        t3 = time.time()
        # if not done:
        #     mask = 1.0
        # else:
        #     mask = 0.0
        mask = 1.0 # always 1 due to continuing task
        agent.add(state, action, reward, next_state, mask, first_step)
        obs = next_state
        first_step = False
        
        if done:
            
            obs, _ = env.reset()
            info['tag'] = 'train'
            info['elapsed_time'] = time.time() - task_start_time
            info['dump'] = True
            L.push(info)
            first_step = True
        if env.total_steps > args.init_steps and env.total_steps % args.update_every == 0:

            update_infos = agent.update()
            
            if update_infos is not None:


                for update_info in update_infos:
                    update_info['action_sample_time'] = (t2 - t1) * 1000
                    update_info['env_time'] = (t3 - t2) * 1000
                    update_info['step'] = env.total_steps
                    update_info['tag'] = 'train'
                    update_info['dump'] = False

                    L.push(update_info)
                    # exit(0)

        if env.total_steps % args.xtick == 0:
            L.plot()

        if args.save_model and env.total_steps % args.save_model_freq == 0 and \
            env.total_steps < args.env_steps:
            agent.checkpoint(env.total_steps)

            ## visualize checkpoint
            visualize_policy(agent, env, args.work_dir)
    if args.save_model:
        agent.checkpoint(env.total_steps)

        ## visualize checkpoint
        visualize_policy(agent, env, args.work_dir)
    L.plot()
    L.close()

    agent.close()

    end_time = time.time()
    print(f'\nFinished in {end_time - task_start_time}s')

