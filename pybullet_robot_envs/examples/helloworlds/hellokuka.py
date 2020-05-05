# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


import pybullet as p
import pybullet_data
import time
import os
import math as m
import robot_data

# Open GUI and set pybullet_data in the path
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load plane contained in pybullet_data
planeId = p.loadURDF("plane.urdf")

# Set gravity for simulation
p.setGravity(0,0,-9.8)

dir_path = os.path.dirname(os.path.realpath(__file__))
for root, dirs, files in os.walk(os.path.dirname(dir_path)):
    for file in files:
        if file.endswith('.urdf'):
            print (root)
            p.setAdditionalSearchPath(root)

kukaId = p.loadURDF(os.path.join(robot_data.getDataPath(),"kuka_kr6_support/urdf/kr6r700sixx.urdf"))
# kukaId = p.loadURDF(os.path.join(robot_data.getDataPath(),"kuka_kr6_support/urdf/kuka_model.urdf"))


# set constraint between base_link and world
cid = p.createConstraint(kukaId,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],
						 [p.getBasePositionAndOrientation(kukaId)[0][0],
						  p.getBasePositionAndOrientation(kukaId)[0][1],
						  p.getBasePositionAndOrientation(kukaId)[0][2]*1.2],
						 p.getBasePositionAndOrientation(kukaId)[1])
numJoints = p.getNumJoints(kukaId)
print(numJoints)



for i in range(numJoints):
	info = p.getJointInfo(kukaId, i)
	print(info[:4])

endEffLink = 6
for i in range(numJoints):
	state = p.getLinkState(kukaId, endEffLink, computeLinkVelocity=1)
	print(state)

while True:
	p.stepSimulation()
	time.sleep(0.01)
