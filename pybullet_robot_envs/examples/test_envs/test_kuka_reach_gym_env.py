import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_robot_envs.envs.kuka_envs.kuka_reach_gym_env import kukaReachGymEnv
from pybullet_robot_envs import robot_data
import pybullet_data

import math
import time

def main():
    use_IK = 0
    discreteAction = 0
    use_IK = 1 if discreteAction else use_IK

    env = kukaReachGymEnv(urdfRoot=robot_data.getDataPath(), renders=True, useIK=use_IK, isDiscrete=discreteAction)
    motorsIds = []

    dv = 1
    numJoint = env._p.getNumJoints(env._kuka.kukaId)
    for i in range(numJoint):
        info = env._p.getJointInfo(env._kuka.kukaId,i)
        jointName = info[1]
        print(info)
        motorsIds.append(env._p.addUserDebugParameter(jointName.decode("utf-8"), -dv, dv, 0.0))

    done = False
    env._p.addUserDebugText('current hand position',[0,-0.5,1.4],[1.1,0,0])
    idx = env._p.addUserDebugText(' ',[0,-0.5,1.2],[1,0,0])

    for t in range(int(1e7)):
        #env.render()
        action = []
        for motorId in range(6):
            action.append(env._p.readUserDebugParameter(motorId))

        action = int(action[0]) if discreteAction else action
        # print(action)
        #print(env.step(action))
        # action = env.action_space.sample()
        # print(action)
        # action[-3] = action[-2]

        state, reward, done, _ = env.step(action)
        if t%100==0:
            print("reward ", reward)
            print("done ", done)
            env._p.addUserDebugText(' '.join(str(round(e,2)) for e in state[:6]),[0,-0.5,1.2],[1,0,0],replaceItemUniqueId=idx)

if __name__ == "__main__":
    main()