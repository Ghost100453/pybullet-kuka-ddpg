# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
from os import path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir =path.abspath(path.join(__file__ ,"../../../../../.."))
os.sys.path.insert(0, parentdir)
print(parentdir)
from pybullet_robot_envs.envs.kuka_envs.kuka_reach_gym_env import kukaReachGymEnv
from pybullet_robot_envs.envs.kuka_envs.kuka_reach_gym_env_her import kukaReachGymEnvHer


from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG, HER
from termcolor import colored

import datetime
import pybullet_data
import robot_data
import numpy as np
import time
import math as m
import gym
import sys, getopt



def main(argv):
    # -p
    fixed = False
    # -j
    numControlledJoints = 6
    # -n
    policy_name = "reaching_policy"

    # COMMAND LINE PARAMS MANAGEMENT:
    try:
        opts, args = getopt.getopt(argv,"hj:p:n:",["j=","p=","n="])
    except getopt.GetoptError:
        print ('test.py -j <numJoints> -p <fixedPoseObject> -p <policy_name> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('------------------ Default values:')
            print('test.py  -j <numJoints: 6> -p <fixedPoseObject: False> -n <policy_name:"pushing_policy"> ')
            print('------------------')
            return 0
            sys.exit()
        elif opt in ("-j", "--j"):
            if(numControlledJoints >7):
                print("Check dimension state")
                return 0
            else:
                numControlledJoints = int(arg)
        elif opt in ("-p", "--p"):
            fixed = bool(arg)
        elif opt in ("-n","--n"):
            policy_name = str(arg)


    print(colored("-----Number Joints Controlled:","red"))
    print(colored(numControlledJoints,"red"))
    print(colored("-----Object Position Fixed:","red"))
    print(colored(fixed,"red"))
    print(colored("-----Policy Name:","red"))
    print(colored(policy_name,"red"))
    print(colored("------","red"))
    print(colored("Launch the script with -h for further info","red"))

    # model = DDPG.load("../pybullet_logs/kukareach_ddpg/"+ policy_name)
    kukaenv = kukaReachGymEnvHer(urdfRoot=robot_data.getDataPath(), renders=True, useIK=0, numControlledJoints = numControlledJoints, fixedPositionObj = fixed, includeVelObs = True)
    model = HER.load("../pybullet_logs/kukareach_ddpg_her/"+ policy_name,env=kukaenv)

    for i in range(10):
        obs = kukaenv.reset()
        dones = False
        while not dones:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = kukaenv.step(action)
        
        kukaenv.render()
        print('info:', info)
        time.sleep(3)


if __name__ == '__main__':
    main(sys.argv[1:])
