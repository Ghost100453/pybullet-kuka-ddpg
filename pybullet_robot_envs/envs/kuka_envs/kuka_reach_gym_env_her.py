# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

from collections import OrderedDict
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

class kukaReachGymEnvHer(kukaGymEnv):
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
                 reward_type=1):
        super().__init__(urdfRoot, useIK, isDiscrete, actionRepeat, renders,
                         maxSteps, dist_delta, numControlledJoints, fixedPositionObj)
        self.reward_type = reward_type
        self.reset()
        # self._observation = self.reset()
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
            self.action_space = spaces.Box(-action_high,
                                           action_high, dtype=np.float32)
        self.viewer = None

    def reset(self):
        self._reset()
        if (self.fixedPositionObj):
            self._objID = p.loadURDF(os.path.join(
                self._urdfRoot, "kuka_kr6_support/cube_small.urdf"), basePosition=[0.7, 0.0, 0.64], useFixedBase=True)
        else:
            self.target_pose = self._sample_pose()[0]
            self._objID = p.loadURDF(os.path.join(
                self._urdfRoot, "kuka_kr6_support/cube_small.urdf"), basePosition=self.target_pose, useFixedBase=True)

        self._debugGUI()
        for _ in range(10):
            p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return self._observation

    def getExtendedObservation(self):
        # get robot observations
        observation = self._kuka.getObservation()
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)

        endEffPose = list(observation[0:3])

        return OrderedDict([
            ('observation', np.asarray(list(observation).copy())),
            ('achieved_goal', np.asarray(endEffPose.copy())),
            ('desired_goal', np.asarray(list(objPos).copy()))
        ])

    def _termination(self):
        endEffPose = self._kuka.getObservation()[0:3]
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        d = goal_distance(np.array(objPos), np.array(endEffPose))

        if d <= self._target_dist_min:
            # print('successed to reach goal, obj position is {} and endEffPosition is {}'.format(
            #     objPos, endEffPose))
            self.terminated = True
        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return True

        return False
    
    def _compute_reward(self):
        d = goal_distance(np.array(self._observation['achieved_goal']), np.array(self._observation['desired_goal']))
        if self.reward_type == 1:
            return -(d > self._target_dist_min).astype(np.float32)
        elif self.reward_type == 2:
            return -d
        else:
            reward = -d
            if d < self._target_dist_min:
                reward = np.float32(1000.0) + (100 - d*80)   
            return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = goal_distance(np.array(achieved_goal), np.array(desired_goal))
        if self.reward_type == 1:
            return -(d > self._target_dist_min).astype(np.float32)
        elif self.reward_type == 2:
            return -d
        else:
            # reward = -d
            # if d < self._target_dist_min:
            #     reward = np.float32(1000.0) + (100 - d*80)   
            # result = -d
            index = d < self._target_dist_min
            result = -(d).astype(np.float32)
            result[index] += 100
            # print(result)
            return result
    
    def _sample_pose(self):
        self.ws_lim = [[0.1, 0.65], [-0.5, 0.5], [0, 0.2]]
        px1 = np.random.uniform(
            low=self.ws_lim[0][0]+0.005*np.random.rand(), high=self.ws_lim[0][1]-0.005*np.random.rand())
        py1 = np.random.uniform(
            low=self.ws_lim[1][0]+0.005*np.random.rand(), high=self.ws_lim[1][1]-0.005*np.random.rand())
        
        pz1 = np.random.uniform(low=self.ws_lim[2][0],high=self.ws_lim[2][1])
        pz2 = np.random.uniform(low=self.ws_lim[2][0],high=self.ws_lim[2][1])


        if px1 < 0.45:
            px2 = px1 + np.random.uniform(0.1, 0.2)
        else:
            px2 = px1 - np.random.uniform(0.1, 0.2)
        if py1 < 0:
            py2 = py1 + np.random.uniform(0.2, 0.3)
        else:
            py2 = py1 - np.random.uniform(0.2, 0.3)

        pz = self._h_table
        pz1 += self._h_table
        pz2 += self._h_table
        pose1 = [px1, py1, pz1]
        pose2 = [px2, py2, pz2]
        # pose1 = [px1, py1, pz]
        # pose2 = [px2, py2, pz]
        return pose1, pose2