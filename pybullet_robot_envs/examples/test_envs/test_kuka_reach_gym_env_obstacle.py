import time
import math
import pybullet_data
from pybullet_robot_envs import robot_data
from pybullet_robot_envs.envs.kuka_envs.kuka_reach_gym_env_obstacle import kukaReachGymEnvOb
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import argparse
def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.5, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--reward_type', type=int, default=1, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--obs-num', type=int, default=2, help='the number of obstacles')


    args = parser.parse_args()

    return args


def main(args):
    use_IK = 0
    discreteAction = 0
    use_IK = 1 if discreteAction else use_IK

    env = kukaReachGymEnvOb(urdfRoot=robot_data.getDataPath(), 
                            maxSteps=100000, actionRepeat=20, 
                            renders=True, useIK=use_IK, 
                            isDiscrete=discreteAction, obstacles_num=args.obs_num)
    motorsIds = []

    dv = 1
    numJoint = env._p.getNumJoints(env._kuka.kukaId)
    for i in range(numJoint):
        info = env._p.getJointInfo(env._kuka.kukaId, i)
        jointName = info[1]
        print(info)
        motorsIds.append(env._p.addUserDebugParameter(
            jointName.decode("utf-8"), -dv, dv, 0.0))

    done = False
    env._p.addUserDebugText('current hand position', [
                            0, -0.5, 1.4], [1.1, 0, 0])
    idx = env._p.addUserDebugText(' ', [0, -0.5, 1.2], [1, 0, 0])

    for t in range(int(1e7)):
        # env.render()
        action = []
        for motorId in range(6):
            action.append(env._p.readUserDebugParameter(motorId))

        # action = int(action[0]) if discreteAction else action
        # print(action)
        # print(env.step(action))
        action = env.action_space.sample()
        # print(action)
        # action[-3] = action[-2]

        state, reward, done, _ = env.step(action)
        time.sleep(0.1)
        if t % 100 == 0:
            print("reward ", reward)
            print("done ", done)
            env._p.addUserDebugText(' '.join(str(round(e, 2)) for e in state['observation'][:6]), [
                                    0, -0.5, 1.2], [1, 0, 0], replaceItemUniqueId=idx)


if __name__ == "__main__":
    args = get_args()
    main(args)
