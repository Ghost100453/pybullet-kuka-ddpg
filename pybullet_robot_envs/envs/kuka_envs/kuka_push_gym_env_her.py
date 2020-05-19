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
from collections import OrderedDict
from datetime import datetime
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


class kukaPushGymEnvHer(kukaGymEnv):
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
        observation_dim = len(self._observation['observation'])
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-largeValObservation, largeValObservation, shape=(observation_dim,), dtype=np.float32),
            'achieved_goal': spaces.Box(-largeValObservation, largeValObservation, shape=(3,), dtype=np.float32),
            'desired_goal': spaces.Box(-largeValObservation, largeValObservation, shape=(3,), dtype=np.float32)
        })
        if (self._isDiscrete):
            self.action_space = spaces.Discrete(
                self._kuka.getActionDimension())
        else:
            self._action_bound = 1
            action_high = np.array([self._action_bound] * self.action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype='float32')
        self.viewer = None
    
    def reset(self):
        self._reset()
        # Randomize start position of object and target.
        self.obj_pose, self.target_pose = self._sample_pose()
        self.init_obj_pos = self.obj_pose
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
        return self._observation

    def getExtendedObservation(self):
        # get robot observations
        observation = self._kuka.getObservation()
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        observation.extend(list(objPos))
        observation.extend(list(objOrn))
    
        return OrderedDict([
            ('observation', np.asarray(list(observation).copy())),
            ('achieved_goal', np.asarray(list(objPos).copy())),
            ('desired_goal', np.asarray(list(self.target_pose).copy()))
        ])

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
        endEffPos = self._kuka.getObservation()[0:3]
        d = goal_distance(np.array(objPos), np.array(self.target_pose))
        # d = d1 + d2
        # reward = -d
        if self.reward_type == 1:
            if goal_distance(np.array(self.init_obj_pos), np.array(objPos)) < 0.1:
                return -2
            return -(d > self._target_dist_min).astype(np.float32)
        else:
            if self.init_obj_pos == objPos:
                return -d-goal_distance(np.array(objPos), np.array(endEffPos))
            return -d

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = goal_distance(np.array(achieved_goal), np.array(desired_goal))
        # print(self.init_obj_pos)
        # print(achieved_goal.shape)
        init_obj_pos_array = np.array([self.init_obj_pos]*achieved_goal.shape[0])
        # print(init_obj_pos_array)
        if self.reward_type == 1:
            index = goal_distance(np.array(init_obj_pos_array), np.array(achieved_goal)) < 0.1
            result = -(d > self._target_dist_min).astype(np.float32)
            result[index] = -2
            # print(result)
            return result
        else:
            return -d