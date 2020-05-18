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


class kukaPickGymEnvHer(kukaGymEnv):
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
        self.obj_pose, _ = self._sample_pose()
        self.target_pose = np.array([0.5, 0.0, 0.84])
        self.fixedPositionObj = True
        if (self.fixedPositionObj):
            self._objID = p.loadURDF(os.path.join(
                self._urdfRoot, "kuka_kr6_support/cube.urdf"), basePosition=[0.6, 0.0, 0.64], useFixedBase=False)
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
    
    def step(self, action):
        if self._useIK:
            # TO DO
            return 0
        else:
            action[-1] = action[-2]
            endEffPos = self._kuka.getObservation()[0:3]
            objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
            if goal_distance(np.array(objPos), np.array(endEffPos)) > 0.1:
                action[-2] = action[-1] = 0.02
            else:
                action[-2] = action[-1] = -0.02
            # if endEffPos[2] - self._h_table > 0.1:
            #     action[-2] = action[-1] = 0.02
            # else:
            #     action[-2] = action[-1] = 0.02
            action = [float(i*0.05) for i in action]
            return self.step2(action)
    
    def step2(self, action):
        # print('actionRepeat:', self._actionRepeat)
        for i in range(self._actionRepeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
            # print(datetime.now(), 'envStepCounter:', self._envStepCounter, 'terminated:', self._termination())
            if self._renders:
                time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()
        reward = self.compute_reward(self._observation['achieved_goal'], self._observation['desired_goal'], None)[0]
        done = self._termination()
        info = {'is_success': False}
        if self.terminated:
            info['is_success'] = True
        return self._observation, reward, done, info
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # d = goal_distance(np.array(achieved_goal), np.array(desired_goal))
        # if self.reward_type == 1:
        #     return -(d > self._target_dist_min).astype(np.float32)
        # else:
        #     return -d
        achieved_goal = np.array(achieved_goal).reshape(-1,3)
        if self.reward_type == 1:
            return -(achieved_goal< self._target_dist_min+self._h_table)[:,2].astype(np.float32)
        else:
            return (self._h_table + self._target_dist_min - achieved_goal)[:,2]