# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.
import math as m
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from kukakr6 import kukakr6
import random
import pybullet_data
import robot_data
from pkg_resources import parse_version
from datetime import datetime
from kuka_gym_env import kukaGymEnv
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)


largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class kukaPushGymEnv(kukaGymEnv):
    def __init__(self, urdfRoot=robot_data.getDataPath(),
                 useIK=0,
                 isDiscrete=0,
                 actionRepeat=1,
                 renders=False,
                 maxSteps=1000,
                 dist_delta=0.03,
                 numControlledJoints=6,
                 fixedPositionObj=False,
                 includeVelObs=True,
                 reward_type = 1):
        super().__init__(urdfRoot, useIK, isDiscrete, actionRepeat, renders, maxSteps,
                         dist_delta, numControlledJoints, fixedPositionObj, includeVelObs)
        self.reward_type = reward_type
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
        self.obj_pose, self.target_pose = self._sample_pose()
        if (self.fixedPositionObj):
            self._objID = p.loadURDF(os.path.join(
                self._urdfRoot, "kuka_kr6_support/cube.urdf"), basePosition=[0.7, 0.0, 0.64], useFixedBase=False)
        else:
            self._objID = p.loadURDF(os.path.join(
                self._urdfRoot, "kuka_kr6_support/cube.urdf"), basePosition=self.obj_pose, useFixedBase=False)

        self._targetID = p.loadURDF(os.path.join(
            self._urdfRoot, "kuka_kr6_support/cube_small.urdf"), basePosition=self.target_pose, useFixedBase=True)

        # Let the world run for a bit
        for _ in range(10):
            p.stepSimulation()

        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def getExtendedObservation(self):

        # get robot observations
        self._observation = self._kuka.getObservation()
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)

        self._observation.extend(list(objPos))
        self._observation.extend(list(objOrn))
        # add target object position
        self._observation.extend(self.target_pose)
        return self._observation
    
    def _termination(self):
        endEffPose = self._kuka.getObservation()[0:3]
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        d = goal_distance(np.array(objPos), np.array(self.target_pose))
        if d <= self._target_dist_min:
            print('successed to reach goal, obj position is {} and endEffPosition is {}'.format(
                objPos, endEffPose))
            self.terminated = True
        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return True
        return False

    def _compute_reward(self):
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        endEffAct = self._kuka.getObservation()[0:3]
        d2 = goal_distance(np.array(objPos), np.array(self.target_pose))
        # d = d1 + d2
        # reward = -d
        reward = -1
        if d2 <= self._target_dist_min:
            if self.reward_type == 1:
                reward = 0
            else:
                reward = np.float32(1000.0) + (100 - d2*80)
        else:
            if self.reward_type == 1:
                reward = -1
            else:
                reward = -d2
        return reward