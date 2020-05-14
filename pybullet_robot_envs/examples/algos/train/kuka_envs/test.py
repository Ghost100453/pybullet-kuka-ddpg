# TODO
# # Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
from os import path
#print(currentdir)
parentdir =path.abspath(path.join(__file__ ,"../../../../../.."))
os.sys.path.insert(0, parentdir)
print(parentdir)

from pybullet_robot_envs.envs.kuka_envs.kuka_reach_gym_env_obstacle import kukaReachGymEnv
from stable_baselines import logger
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from termcolor import colored
from baselines import logger
from baselines.her import her

import datetime
import pybullet_data
import robot_data
import numpy as np
import time
import math as m
import gym
import sys, getopt

def main(argv):

    numControlledJoints = 6
    fixed = False
    normalize_observations = False
    gamma = 0.9
    batch_size = 16
    memory_limit = 1000000
    normalize_returns = True
    timesteps = 1000000
    policy_name = "reaching_policy"
    # COMMAND LINE PARAMS MANAGEMENT:


    discreteAction = 0
    # rend = True
    rend = False

    kukaenv = kukaReachGymEnv(urdfRoot=robot_data.getDataPath(), renders=rend, useIK=0, isDiscrete=discreteAction, numControlledJoints = numControlledJoints, fixedPositionObj = fixed, includeVelObs = True)
    n_actions = kukaenv.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    kukaenv = DummyVecEnv([lambda: kukaenv])
    full_log = True
    model = DDPG(LnMlpPolicy, kukaenv,normalize_observations = normalize_observations, gamma=gamma,batch_size=batch_size,
                    memory_limit=memory_limit, normalize_returns = normalize_returns, verbose=1, param_noise=param_noise,
                    action_noise=action_noise, tensorboard_log="../pybullet_logs/kuka_reach_ddpg/reaching_obstacle_DDPG_PHASE_1",full_tensorboard_log=full_log,reward_scale = 1)

    print(colored("-----Timesteps:","red"))
    print(colored(timesteps,"red"))
    print(colored("-----Number Joints Controlled:","red"))
    print(colored(numControlledJoints,"red"))
    print(colored("-----Object Position Fixed:","red"))
    print(colored(fixed,"red"))
    print(colored("-----Policy Name:","red"))
    print(colored(policy_name,"red"))
    print(colored("------","red"))
    print(colored("Launch the script with -h for further info","red"))

    model.learn(total_timesteps=timesteps)

    print("Saving model to kuka.pkl")
    model.save("../pybullet_logs/kukareach_obstacle_ddpg/"+ policy_name)

    del model # remove to demonstrate saving and loading

if __name__ == '__main__':
    main(sys.argv[1:])
