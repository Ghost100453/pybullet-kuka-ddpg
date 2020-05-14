# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

from kuka_gym_env import kukaGymEnv
from pkg_resources import parse_version
import robot_data
import pybullet_data
import random
from kukakr6 import kukakr6
import pybullet as p
import time
import numpy as np
from gym.utils import seeding
from gym import spaces
import gym
import math as m


largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class kukaReachGymEnv(kukaGymEnv):

    def __init__(self, urdfRoot=robot_data.getDataPath(),
                 useIK=0,
                 isDiscrete=0,
                 actionRepeat=1,
                 renders=False,
                 maxSteps=1000,
                 dist_delta=0.03, 
                 numControlledJoints=6, 
                 fixedPositionObj=True, 
                 includeVelObs=True,
                 reward_type = 1):
        super().__init__(urdfRoot, useIK, isDiscrete, actionRepeat, renders,
                       maxSteps, dist_delta, numControlledJoints, fixedPositionObj)
        self.reward_type = reward_type

        # self.reset()
        observationDim = len(self._observation)
        observation_high = np.array([largeValObservation] * observationDim)
        self.observation_space = spaces.Box(-observation_high,
                                            observation_high, dtype='float32')

        if (self._isDiscrete):
            self.action_space = spaces.Discrete(
                self._kuka.getActionDimension())
        else:
            # self.action_dim = 2 #self._kuka.getActionDimension()
            self._action_bound = 1
            action_high = np.array([self._action_bound] * self.action_dim)
            self.action_space = spaces.Box(-action_high,
                                           action_high, dtype='float32')
        self.viewer = None
  
    def reset(self):
        self._reset()
        # Randomize start position of object and target.
        if (self.fixedPositionObj):
            self._objID = p.loadURDF(os.path.join(
                self._urdfRoot, "kuka_kr6_support/cube_small.urdf"), basePosition=[0.7, 0.0, 0.64], useFixedBase=True)
        else:
            self.target_pose = self._sample_pose()[0]
            self._objID = p.loadURDF(os.path.join(
                self._urdfRoot, "kuka_kr6_support/cube_small.urdf"), basePosition=self.target_pose, useFixedBase=True)

        self._debugGUI()
        p.setGravity(0, 0, -9.8)
        # Let the world run for a bit
        for _ in range(10):
            p.stepSimulation()

        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def getExtendedObservation(self):
        # reach env
        # get robot state and target object state
        self._observation = self._kuka.getObservation()
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)

        self._observation.extend(list(objPos))
        self._observation.extend(list(objOrn))
        return self._observation
    
    def _termination(self):
        endEffPose = self._kuka.getObservation()[0:3]
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        d = goal_distance(np.array(objPos), np.array(endEffPose))
        if d <= self._target_dist_min:
            print('successed to reach goal, obj position is {} and endEffPosition is {}'.format(
                objPos, endEffPose))
            self.terminated = True

        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return True

        return False

    def _compute_reward(self):
        # 可以设置选项， dense reward or sparse reward
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        endEffPose = self._kuka.getObservation()[0:3]
        d = goal_distance(np.array(objPos), np.array(endEffPose))
        if self.reward_type == 1:
            reward = -1
            if d <= self._target_dist_min:
                reward = 0
        else:
            reward = -d
            if d <= self._target_dist_min:
                reward = np.float32(1000.0) + (100 - d*80)
        return reward
