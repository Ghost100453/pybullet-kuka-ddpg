import os
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

from datetime import datetime
from pkg_resources import parse_version
# import robot_data
from pybullet_robot_envs import robot_data
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


class kukaGymEnv(gym.Env):

    def __init__(self, urdfRoot=robot_data.getDataPath(),
                 useIK=0,
                 isDiscrete=0,
                 actionRepeat=1,
                 renders=False,
                 maxSteps=1000,
                 dist_delta=0.03, 
                 numControlledJoints=6, 
                 fixedPositionObj=False, 
                 includeVelObs=True):

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
        self.ws_lim = [[0.3, 0.5], [-0.2, 0.2], [0, 1]]

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.5, 90, -60, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        # self.seed()
        # initialize simulation environment
        # self.reset()

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
        # self._debugGUI()
        p.setGravity(0, 0, -9.8)
        # Let the world run for a bit
        for _ in range(10):
            p.stepSimulation()

        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def _reset(self):
        self.terminated = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        self._envStepCounter = 0
        p.loadURDF(os.path.join(pybullet_data.getDataPath(),
                                "plane.urdf"), useFixedBase=True)
        # Load robot
        self._kuka = kukakr6(self._urdfRoot, timeStep=self._timeStep, basePosition=[0, 0, 0.75],
                             useInverseKinematics=self._useIK, action_space=self.action_dim, includeVelObs=self.includeVelObs)
        # Load table and object for simulation
        tableId = p.loadURDF(os.path.join(
            self._urdfRoot, "kuka_kr6_support/table.urdf"), useFixedBase=True)
        # tableId = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table.urdf"), useFixedBase=True)
        table_info = p.getVisualShapeData(tableId, -1)[0]
        self._h_table = table_info[5][2] + table_info[3][2]
        # limit panda workspace to table plane
        self._kuka.workspace_lim[2][0] = self._h_table
        p.setGravity(0, 0, -9.8)
       

    def getExtendedObservation(self):

        # get robot observations
        self._observation = self._kuka.getObservation()
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)

        self._observation.extend(list(objPos))
        self._observation.extend(list(objOrn))
        # add target object position
        self._observation.extend(self.target_pose)
        return self._observation

    def step(self, action):
        if self._useIK:
            # TO DO
            return 0
        else:
            action = [float(i*0.05) for i in action]
            return self.step2(action)

    def step2(self, action):
        # print('actionRepeat:', self._actionRepeat)
        for i in range(self._actionRepeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            # print('termination:', self._termination())
            if self._termination():
                break
            self._envStepCounter += 1
            # print(datetime.now(), 'envStepCounter:', self._envStepCounter, 'terminated:', self._termination())
            if self._renders:
                time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()
        reward = self._compute_reward()
        done = self._termination()
        # 这里info要提供'is_success'信息，方便stable baselines 记录统计success rate数据
        # stable baselines 记录 如果success rate为0，是不显示的，但是success rate需要info数据
        # 这里返回为空，所以输出没有success rate
        info = {'is_success': False}
        if self.terminated:
            info['is_success'] = True
        return self._observation, reward, done, info

    def render(self, mode="rgb_array", close=False):
        # TODO Check the behavior of this function
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(
            self._kuka.kukaId)
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
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        # Termination对每个环境是不同的需要重写
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
        # 可以设置选项， dense reward or sparse reward
        objPos, objOrn = p.getBasePositionAndOrientation(self._objID)
        endEffAct = self._kuka.getObservation()[0:3]
        # d1 = goal_distance(np.array(endEffAct), np.array(objPos))
        d = goal_distance(np.array(objPos), np.array(self.target_pose))
        # d = d1 + d2
        # reward = -d
        reward = -1
        if d2 <= self._target_dist_min:
            # reward = np.float32(1000.0) + (100 - d*80)
            reward = 0
        # 将dense reward(密集型)改为sparse reward(稀疏型)
        # 采用sparse reward的原因有
        # 1. 比较符合现实情况，只有做到了和没做到两种
        # 2. 相比dense reward减少了reward的波动，使得训练更为稳定
        return reward

    def _sample_pose(self):
        px1 = np.random.uniform(
            low=self.ws_lim[0][0]+0.005*np.random.rand(), high=self.ws_lim[0][1]-0.005*np.random.rand())
        py1 = np.random.uniform(
            low=self.ws_lim[1][0]+0.005*np.random.rand(), high=self.ws_lim[1][1]-0.005*np.random.rand())

        if px1 < 0.45:
            px2 = px1 + np.random.uniform(0.1, 0.2)
        else:
            px2 = px1 - np.random.uniform(0.1, 0.2)
        if py1 < 0:
            py2 = py1 + np.random.uniform(0.2, 0.3)
        else:
            py2 = py1 - np.random.uniform(0.2, 0.3)

        pz = self._h_table
        pose1 = [px1, py1, pz]
        pose2 = [px2, py2, pz]
        return pose1, pose2

    def _debugGUI(self):
        # TO DO
        return 0
