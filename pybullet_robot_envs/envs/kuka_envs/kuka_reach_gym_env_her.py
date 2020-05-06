# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

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

from collections import OrderedDict


largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis = -1)

class kukaReachGymEnvHer(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }

    def __init__(self, urdfRoot=robot_data.getDataPath(),
                 useIK = 0,
                 isDiscrete = 0,
                 actionRepeat = 1,
                 renders = False,
                 maxSteps = 1000,
                 dist_delta = 0.03, numControlledJoints = 6, fixedPositionObj = True, includeVelObs = True):

        self.action_dim = numControlledJoints
        self._isDiscrete = isDiscrete
        self._timeStep = 1./240.
        self._useIK = useIK
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = False
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._h_table = []
        self._target_dist_max = 0.3
        self._target_dist_min = 0.1
        self._p = p
        self.fixedPositionObj = fixedPositionObj
        self.includeVelObs = includeVelObs

        if self._renders:
          cid = p.connect(p.SHARED_MEMORY)
          if (cid<0):
             cid = p.connect(p.GUI)
          p.resetDebugVisualizerCamera(2.5,90,-60,[0.52,-0.2,-0.33])
        else:
            p.connect(p.DIRECT)

        #self.seed()
        # initialize simulation environment
        self._observation = self.reset()
        observation_dim = len(self._observation['observation'])
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-largeValObservation, largeValObservation, shape=(observation_dim,), dtype=np.float32),
            'achieved_goal': spaces.Box(-largeValObservation, largeValObservation, shape=(3,), dtype=np.float32),
            'desired_goal': spaces.Box(-largeValObservation, largeValObservation, shape=(3,), dtype=np.float32)
        })

        if (self._isDiscrete):
            self.action_space = spaces.Discrete(self._kuka.getActionDimension())

        else:
            #self.action_dim = 2 #self._kuka.getActionDimension()
            self._action_bound = 1
            action_high = np.array([self._action_bound] * self.action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        self.viewer = None

    def reset(self):
        self.terminated = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        self._envStepCounter = 0
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), useFixedBase= True)
        # Load robot
        self._kuka = kukakr6(self._urdfRoot, timeStep=self._timeStep, basePosition =[0,0,0.625], useInverseKinematics= self._useIK, action_space = self.action_dim, includeVelObs = self.includeVelObs)
        # Load table and object for simulation
        tableId = p.loadURDF(os.path.join(self._urdfRoot, "kuka_kr6_support/table.urdf"), useFixedBase=True)
        # tableId = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table.urdf"), useFixedBase=True)


        table_info = p.getVisualShapeData(tableId,-1)[0]
        self._h_table =table_info[5][2] + table_info[3][2]

        #limit panda workspace to table plane
        self._kuka.workspace_lim[2][0] = self._h_table
        # Randomize start position of object and target.

        if (self.fixedPositionObj):
            self._objID = p.loadURDF( os.path.join(self._urdfRoot,"kuka_kr6_support/cube_small.urdf"), basePosition = [0.7,0.0,0.64], useFixedBase=True)
        else:
            self.target_pose = self._sample_pose()[0]
            self._objID = p.loadURDF( os.path.join(self._urdfRoot,"kuka_kr6_support/cube_small.urdf"), basePosition= self.target_pose, useFixedBase=True)

        self._debugGUI()
        p.setGravity(0,0,-9.8)
        # Let the world run for a bit
        for _ in range(10):
            p.stepSimulation()

        self._observation = self.getExtendedObservation()
        return self._observation


    def getExtendedObservation(self):

        #get robot observations
        observation = self._kuka.getObservation()
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)

        observation.extend(list(objPos))
        observation.extend(list(objOrn))

        endEffPose = list(observation[0:3])

        return OrderedDict([
            ('observation', np.asarray(list(observation).copy())),
            ('achieved_goal', np.asarray(endEffPose.copy())),
            ('desired_goal', np.asarray(list(objPos).copy()))
            ])

    def step(self, action):
        if self._useIK:
            #TO DO
            return 0
        else:
            action = [float(i*0.05) for i in action]
            return self.step2(action)

    def step2(self,action):

        for i in range(self._actionRepeat):
            self._kuka.applyAction(action)
            p.stepSimulation()

            if self._termination()[0]:
                print('hello', self._envStepCounter)
                break

            self._envStepCounter += 1

        if self._renders:
            time.sleep(self._timeStep)

        self._observation = self.getExtendedObservation()

        reward = self.compute_reward(self._observation['achieved_goal'], self._observation['desired_goal'], None)

        done = self._termination()
        info = {'is_success':False}
        if self.terminated:
            info['is_success'] = True

        return self._observation, reward, done[0], info



    def render(self, mode="rgb_array", close=False):
        ## TODO Check the behavior of this function
        if mode != "rgb_array":
          return np.array([])

        base_pos,orn = self._p.getBasePositionAndOrientation(self._kuka.kukaId)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
            #renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def _termination(self):

        endEffPose = self._kuka.getObservation()[0:3]
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        d = goal_distance(np.array(objPos), np.array(endEffPose))

        if d <= self._target_dist_min:
            print('successed to reach goal, obj position is {} and endEffPosition is {}'.format(objPos, endEffPose))
            self.terminated = True


        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return [True]

        return [False]


    def compute_reward(self, achieved_goal, desired_goal, info):

        
        d = goal_distance(np.array(achieved_goal), np.array(desired_goal))
        reward = -d
        if d <= self._target_dist_min:
            reward = np.float32(1000.0) + (100 - d*80)
        return reward


    def _sample_pose(self):
        ws_lim = self._kuka.workspace_lim
        px,tx = np.random.uniform(low=ws_lim[0][0], high=ws_lim[0][1], size=(2))
        py,ty = np.random.uniform(low=ws_lim[1][0], high=ws_lim[1][1], size=(2))
        pz,tz = self._h_table, self._h_table
        obj_pose = [px,py,pz]
        tg_pose = [tx,ty,tz]

        return obj_pose, tg_pose



    def _debugGUI(self):
        #TO DO
        return 0
