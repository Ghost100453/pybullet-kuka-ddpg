# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import getopt
import sys
import tensorflow as tf
import gym
import math as m
import time
import numpy as np
from pybullet_robot_envs import robot_data
# import robot_data
import pybullet_data
from datetime import datetime
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from termcolor import colored
from stable_baselines import DDPG, HER, A2C, TD3, DQN
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import logger
from pybullet_robot_envs.envs.kuka_envs.kuka_push_gym_env_her import kukaPushGymEnvHer
import os
import inspect
from os import path
# print(currentdir)
parentdir = path.abspath(path.join(__file__, "../../../../../.."))
os.sys.path.insert(0, parentdir)
print(parentdir)

from mpi4py import MPI


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256, 256],
                                           layer_norm=True,
                                           act_fun=tf.nn.relu,
                                           feature_extraction="lnmlp")


best_mean_reward, n_steps = -np.inf, 0
log_dir = "../pybullet_logs/kuka_push_ddpg_her/"
log_dir_policy = '../policies/pushing_DDPG_HER_PHASE_1'

def callback(_locals, _globals):
    global n_steps, best_mean_reward, log_dir
    # Print stats every 1000 calls
    if (n_steps) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(datetime.now(), x[-1], 'timesteps')
            print(datetime.now(
            ), "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print(datetime.now(), "Saving new best model")
                _locals['self'].save(log_dir_policy + 'best_model.pkl')
    n_steps += 1
    return True

def main(argv):
    numControlledJoints = 6
    fixed = False
    normalize_observations = False
    gamma = 0.99
    batch_size = 64
    memory_limit = 1000000
    normalize_returns = True
    timesteps = 1000000
    policy_name = "pushing_policy"
    discreteAction = 0
    rend = False

    kukaenv = kukaPushGymEnvHer(urdfRoot=robot_data.getDataPath(), renders=rend, useIK=0, isDiscrete=discreteAction,
                                numControlledJoints=numControlledJoints, fixedPositionObj=fixed, includeVelObs=True)
    kukaenv = Monitor(kukaenv, log_dir, allow_early_resets=True)

    n_actions = kukaenv.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model_class = DDPG
    goal_selection_strategy = 'future'
    model = HER(CustomPolicy, kukaenv, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                verbose=1, tensorboard_log="../pybullet_logs/kuka_push_ddpg/pushing_DDPG_HER_PHASE_1", buffer_size=1000000, 
                batch_size=64, random_exploration=0.3, action_noise=action_noise)

    print(colored("-----Timesteps:", "red"))
    print(colored(timesteps, "red"))
    print(colored("-----Number Joints Controlled:", "red"))
    print(colored(numControlledJoints, "red"))
    print(colored("-----Object Position Fixed:", "red"))
    print(colored(fixed, "red"))
    print(colored("-----Policy Name:", "red"))
    print(colored(policy_name, "red"))
    print(colored("------", "red"))
    print(colored("Launch the script with -h for further info", "red"))

    model.learn(total_timesteps=timesteps, log_interval=100, callback=callback)

    print("Saving model to kuka.pkl")
    model.save("../pybullet_logs/kukapush_ddpg_her/" + policy_name)

    del model  # remove to demonstrate saving and loading

if __name__ == '__main__':
    main(sys.argv[1:])
