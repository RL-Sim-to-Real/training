from franka_real.FrankaReacherEnv import FrankaReacherEnv
import time
import numpy as np 
from multiprocessing import shared_memory, Process, Queue, Lock

from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere, WrappedEnv
from jsac.helpers.logger import Logger
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent

import cv2


np.set_printoptions(precision=3, linewidth=10000, suppress=True)
import os
import json
import time
import tkinter as tk
from tkinter import ttk
import argparse

import signal

config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    
    'latent': 50,

    'mlp': [1024, 1024],
}


def parse_args():
    parser = argparse.ArgumentParser()
    
    # environment
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default='prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")

    parser.add_argument('--task_name', default='reacher_non_visual_dense', type=str)
    parser.add_argument('--image_height', default=90, type=int)          # Mode: img, img_prop
    parser.add_argument('--image_width', default=160, type=int)          # Mode: img, img_prop     
    parser.add_argument('--image_history', default=3, type=int)          # Mode: img, img_prop
    parser.add_argument('--dt', default=0.15, type=float)
    parser.add_argument('--episode_steps', default=100, type=int) 

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=120_000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=5_000, type=int)
    parser.add_argument('--env_steps', default=120_000, type=int)
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

def handle_cntrl_c(signal, frame, env):
    env.reset()
    env.close()
    exit(0)

def run_policy(args):
    env = FrankaReacherEnv(dt=0.04)
    env = WrappedEnv(env, start_step= args.start_step, 
                     start_episode=args.start_episode, episode_max_steps=200)
    
    args.work_dir += f'/results/{args.task_name}/seed_{args.seed}/'

    if os.path.exists(os.path.abspath(args.work_dir)):
        print("loading model from work directory")
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

    proprioception_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    env_action_space = env.action_space

    print(f"{proprioception_shape}, {action_shape}, {env_action_space}")
    set_seed_everywhere(seed=args.seed)
    
    # args.single_image_shape = (args.image_width, args.image_height, 3)
    args.proprioception_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    args.image_shape = None
    args.net_params = config
    agent = SACRADAgent(vars(args))
    obs, _ = env.reset()
    # returns = []
    print("Env reset")
    print(obs)


    done = False
    rewards = []
    signal.signal(signal.SIGINT, lambda signal, frame: handle_cntrl_c(signal, frame, env))
    while not done:
        action = agent.sample_actions(obs, deterministic=True)
        action = action * 0.5  # action scaling done for safety
        next_obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")

        obs = next_obs
        ee_pos = obs[14:17]
        target = obs[17:]
        distance = np.linalg.norm(ee_pos - target)

        print(f"Distance: {distance}")
        print(f"EE Pos: {ee_pos}, Target: {target}")

        
        if done:
            obs, _ = env.reset()
            done = False
            time.sleep(2)
            print("Episode Done")

    env.close()
    exit(0)
    print("Policy run complete")
    
            



if __name__ == "__main__":
    args = parse_args()

    print("Running Policy")
    run_policy(args)
    # env = FrankaReacherEnv()
    # env.reset()