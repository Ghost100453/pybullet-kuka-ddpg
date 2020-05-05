# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import math as m
from pybullet_robot_envs import robot_data
# import robot_data
import pybullet_data
import math
import copy
import numpy as np
import pybullet as p
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class kukakr6:

    def __init__(self, urdfRootPath=robot_data.getDataPath(), timeStep=0.01, useInverseKinematics=0, basePosition=[-0.6, -0.4, 0.625], useFixedBase=True, action_space=6, includeVelObs=True):
        self.urdfRootPath = os.path.join(urdfRootPath, "kuka_kr6_support/urdf/kr6r700sixx.urdf")
        # self.urdfRootPath = os.path.join(urdfRootPath, "kuka_kr6_support/urdf/kuka_model.urdf")

        self.timeStep = timeStep
        self.useInverseKinematics = useInverseKinematics
        self.useNullSpace = 0
        self.useOrientation = 1
        self.useSimulation = 1
        self.basePosition = basePosition
        self.useFixedBase = useFixedBase
        self.workspace_lim = [[0.3, 0.60], [-0.3, 0.3], [0, 1]]
        self.workspace_lim_endEff = [[0.1, 0.70], [-0.4, 0.4], [0.65, 1]]
        self.endEffLink = 6
        self.action_space = action_space
        self.includeVelObs = includeVelObs
        self.numJoints = 6
        self.reset()

    def reset(self):
        # 加入碰撞检测
        urdfFlags = p.URDF_USE_SELF_COLLISION
        self.kukaId = p.loadURDF(
            self.urdfRootPath, basePosition=self.basePosition, useFixedBase=self.useFixedBase, flags = urdfFlags)

        for i in range(self.numJoints):
            p.resetJointState(self.kukaId, i, 0)
            p.setJointMotorControl2(self.kukaId, i, p.POSITION_CONTROL, targetPosition=0, targetVelocity=0.0,
                                    positionGain=0.25, velocityGain=0.75, force=50)
        if self.useInverseKinematics:
            self.endEffPos = [0.4, 0, 0.85]  # x,y,z
            self.endEffOrn = [0.3, 0.4, 0.35]  # roll,pitch,yaw

    def getJointsRanges(self):
        # to-be-defined
        return 0

    def getActionDimension(self):
        return self.action_space

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        state = p.getLinkState(
            self.kukaId, self.endEffLink, computeLinkVelocity=1)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)
        observation.extend(list(pos))
        observation.extend(list(euler))  # roll, pitch, yaw
        if (self.includeVelObs):
            velL = state[6]
            velA = state[7]
            observation.extend(list(velL))
            observation.extend(list(velA))

        jointStates = p.getJointStates(self.kukaId, range(8))
        jointPoses = [x[0] for x in jointStates]
        observation.extend(list(jointPoses))

        return observation

    def applyAction(self, action):

        if(self.useInverseKinematics):
            assert len(action) >= 3, ('IK dim differs from ', len(action))
            assert len(action) <= 6, ('IK dim differs from ', len(action))

            dx, dy, dz = action[:3]

            self.endEffPos[0] = min(self.workspace_lim_endEff[0][1], max(
                self.workspace_lim_endEff[0][0], self.endEffPos[0] + dx))
            self.endEffPos[1] = min(self.workspace_lim_endEff[1][1], max(
                self.workspace_lim_endEff[1][0], self.endEffPos[1] + dx))
            self.endEffPos[2] = min(self.workspace_lim_endEff[2][1], max(
                self.workspace_lim_endEff[2][0], self.endEffPos[2] + dx))

            if not self.useOrientation:
                quat_orn = p.getQuaternionFromEuler(self.handOrn)

            elif len(action) is 6:
                droll, dpitch, dyaw = action[3:]
                self.endEffOrn[0] = min(
                    m.pi, max(-m.pi, self.endEffOrn[0] + droll))
                self.endEffOrn[1] = min(
                    m.pi, max(-m.pi, self.endEffOrn[1] + dpitch))
                self.endEffOrn[2] = min(
                    m.pi, max(-m.pi, self.endEffOrn[2] + dyaw))
                quat_orn = p.getQuaternionFromEuler(self.endEffOrn)

            else:
                quat_orn = p.getLinkState(self.kukaId, self.endEffLink)[5]

            jointPoses = p.calculateInverseKinematics(
                self.kukaId, self.endEffLink, self.endEffPos, quat_orn)

            if (self.useSimulation):
                for i in range(self.numJoints):
                    jointInfo = p.getJointInfo(self.kukaId, i)
                    if jointInfo[3] > -1:
                        p.setJointMotorControl2(bodyUniqueId=self.kukaId,
                                                jointIndex=i,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=jointPoses[i],
                                                targetVelocity=0,
                                                positionGain=0.25,
                                                velocityGain=0.75,
                                                force=50)
            else:
                for i in range(self.numJoints):
                    p.resetJointState(self.kukaId, i, jointPoses[i])

        else:
            assert len(action) == self.action_space, (
                'number of motor commands differs from number of motor to control', len(action))

            for a in range(len(action)):

                curr_motor_pos = p.getJointState(self.kukaId, a)[0]
                new_motor_pos = curr_motor_pos + \
                    action[a]  # supposed to be a delta

                p.setJointMotorControl2(self.kukaId,
                                        a,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        targetVelocity=0,
                                        positionGain=0.25,
                                        velocityGain=0.75,
                                        force=100)
