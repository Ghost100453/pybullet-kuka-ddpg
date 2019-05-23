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
import icub
import random
import pybullet_data
import robot_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class iCubPushGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }

    def __init__(self,
                 urdfRoot=robot_data.getDataPath(),
                 useIK=0,
                 control_arm='r',
                 isDiscrete=0,
                 actionRepeat=1,
                 renders=False,
                 maxSteps = 1000,
                 dist_delta=0.03):

        self._control_arm=control_arm
        self._isDiscrete = isDiscrete
        self._timeStep = 1./240.
        self._useIK = 1 if self._isDiscrete else useIK
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._h_table = []

        # Initialize PyBullet simulator
        self._p = p
        if self._renders:
          cid = p.connect(p.SHARED_MEMORY)
          if (cid<0):
             cid = p.connect(p.GUI)
          p.resetDebugVisualizerCamera(2.5,90,-60,[0.52,-0.2,-0.33])
        else:
            p.connect(p.DIRECT)

        self.seed()
        # initialize simulation environment
        self.reset()

        observationDim = len(self._observation)
        observation_high = np.array([largeValObservation] * observationDim)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype='float32')

        if (self._isDiscrete):
            self.action_space = spaces.Discrete(13)
        else:
            action_dim = self._icub.getActionDimension()
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype='float32')

        self.viewer = None

    def reset(self):

        self.terminated = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        self._envStepCounter = 0

        # Load robot
        self._icub = icub.iCub(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, useInverseKinematics=self._useIK, arm=self._control_arm)

        ## Load table and object for simulation
        p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"),[0,0,0])
        tableId = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"table/table.urdf"), [0.85, 0.0, 0.0])
        table_info = p.getVisualShapeData(tableId,-1)[0]
        self._h_table =table_info[5][2] + table_info[3][2]

        #limit iCub workspace to table plane
        self._icub.workspace_lim[2][0] = self._h_table

        # Randomize start position of object and target.
        obj_pose, self.target_pose = self._sample_pose()
        self._objID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "lego/lego.urdf"), [0.3,0.0,0.8])
        self._targetID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "domino/domino.urdf"), self.target_pose)

        self._debugGUI()
        p.setGravity(0,0,-9.8)
        # Let the world run for a bit
        for _ in range(10):
            p.stepSimulation()

        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def getExtendedObservation(self):
        # get robot observation
        self._observation = self._icub.getObservation()
        # read hand position/velocity
        handState = p.getLinkState(self._icub.icubId, self._icub.motorIndices[-1], computeLinkVelocity=1)
        handPos = handState[0]
        handOrn = handState[1]
        handLinkPos = handState[4]
        handLinkOrn = handState[5]
        handLinkVelL = handState[6]
        handLinkVelA = handState[7]

        # get object position and transform it wrt hand c.o.m. frame
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self._objID)
        invHandPos, invHandOrn = p.invertTransform(handPos, handOrn)
        handEul = p.getEulerFromQuaternion(handOrn)

        cubePosInHand, cubeOrnInHand = p.multiplyTransforms(invHandPos, invHandOrn,
                                                                cubePos, cubeOrn)

        self._observation.extend(list(cubePosInHand))
        self._observation.extend(list(cubeOrnInHand))
        #self._observation.extend(list(handLinkVelL))
        #self._observation.extend(list(handLinkVelA))
        return self._observation

    def step(self, action):

        if (self._isDiscrete):
            dv = 0.005
            dv1 = 0.05
            dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0][action]
            dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
            droll = [0, 0, 0, 0, 0, 0, 0, -dv1, dv1, 0, 0, 0, 0][action]
            dpitch = [0, 0, 0, 0, 0, 0, 0, 0, 0, -dv1, dv1, 0, 0][action]
            dyaw = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dv1, dv1][action]

            realAction = [dx, dy, dz, droll, dpitch, dyaw]
            return self.step2(realAction)

        elif self._useIK:

            dv = 0.005
            realPos = [a*0.005 for a in action[:3]]
            realOrn = []

            if self.action_space.shape[-1] is 6:
                realOrn = [a*0.05 for a in action[3:]]

            return self.step2(realPos+realOrn)

        else:
            return self.step2([a*0.05 for a in action])

    def step2(self, action):
        for i in range(self._actionRepeat):
            self._icub.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()
        done = self._termination()

        reward = 0
        return np.array(self._observation), reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="rgb_array", close=False):
        ## TODO Check the behavior of this function
        if mode != "rgb_array":
          return np.array([])

        base_pos,orn = self._p.getBasePositionAndOrientation(self._icub.icubId)
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

    # TODO
    def _termination(self):
        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return True

        #if target as reached:
            #self.terminated = 1
            #...
            #self._observation = self.getExtendedObservation()
            #return True
        return False

    # TODO
    def _compute_reward(self):
        reward = -1000
        return reward

    def _sample_pose(self):
        ws_lim = self._icub.workspace_lim
        px,tx = np.random.uniform(low=ws_lim[0][0], high=ws_lim[0][1], size=(2))
        py,ty = np.random.uniform(low=ws_lim[1][0], high=ws_lim[1][1], size=(2))
        pz,tz = self._h_table, self._h_table
        obj_pose = [px,py,pz]
        tg_pose = [tx,ty,tz]

        return obj_pose, tg_pose

    def _debugGUI(self):
        ws = self._icub.workspace_lim
        p1 = [ws[0][0],ws[1][0],ws[2][0]] # xmin,ymin
        p2 = [ws[0][1],ws[1][0],ws[2][0]] # xmax,ymin
        p3 = [ws[0][1],ws[1][1],ws[2][0]] # xmax,ymax
        p4 = [ws[0][0],ws[1][1],ws[2][0]] # xmin,ymax

        p.addUserDebugLine(p1,p2,lineColorRGB=[0,0,1],lineWidth=2.0,lifeTime=0)
        p.addUserDebugLine(p2,p3,lineColorRGB=[0,0,1],lineWidth=2.0,lifeTime=0)
        p.addUserDebugLine(p3,p4,lineColorRGB=[0,0,1],lineWidth=2.0,lifeTime=0)
        p.addUserDebugLine(p4,p1,lineColorRGB=[0,0,1],lineWidth=2.0,lifeTime=0)

        p.addUserDebugLine([0,0,0],[0.1,0,0],[1,0,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_right_arm[-1])
        p.addUserDebugLine([0,0,0],[0,0.1,0],[0,1,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_right_arm[-1])
        p.addUserDebugLine([0,0,0],[0,0,0.1],[0,0,1],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_right_arm[-1])

        p.addUserDebugLine([0,0,0],[0.1,0,0],[1,0,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_left_arm[-1])
        p.addUserDebugLine([0,0,0],[0,0.1,0],[0,1,0],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_left_arm[-1])
        p.addUserDebugLine([0,0,0],[0,0,0.1],[0,0,1],parentObjectUniqueId=self._icub.icubId, parentLinkIndex=self._icub.indices_left_arm[-1])

        p.addUserDebugLine([0,0,0],[0.1,0,0],[1,0,0],parentObjectUniqueId=self._objID)
        p.addUserDebugLine([0,0,0],[0,0.1,0],[0,1,0],parentObjectUniqueId=self._objID)
        p.addUserDebugLine([0,0,0],[0,0,0.1],[0,0,1],parentObjectUniqueId=self._objID)
