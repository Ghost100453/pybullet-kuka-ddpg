# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print(currentdir)
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))))
os.sys.path.insert(0, parentdir)
print(parentdir)


from stable_baselines import HER, DQN, SAC, DDPG
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from pybullet_robot_envs.envs.panda_envs.panda_push_gym_env_HER import pandaPushGymEnvHER
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor


import robot_data
import tensorflow as tf
from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import numpy as np

from datetime import datetime



class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[256,256,256],
                                           layer_norm=False,
                                           act_fun=tf.nn.relu,
                                           feature_extraction="lnmlp")

best_mean_reward, n_steps = -np.inf, 0
log_dir="../pybullet_logs/panda_push_ddpg/stable_baselines/"
log_dir_policy = '../policies/pushing_DDPG_HER_PHASE_1'


def callback(_locals, _globals):
    global n_steps, best_mean_reward, log_dir
    # Print stats every 1000 calls
    if (n_steps) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(datetime.now(), x[-1], 'timesteps')
            print(datetime.now(), "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print(datetime.now(), "Saving new best model")
                _locals['self'].save(log_dir_policy + 'best_model.pkl')
    n_steps += 1
    return True

def main():
    global log_dir
    model_class = DDPG  # works also with SAC and DDPG
    action_space = 7
    # 固定物块
    fixed = True
    #0 completely fixed, 1 slightly random radius, 2 big random radius,
    object_position = 1
    normalize_observations = False
    gamma = 0.9
    memory_limit = 1000000
    normalize_returns = True
    timesteps = 3000000
    policy_name = "pushing_policy"
    discreteAction = 0
    rend = False
    env = pandaPushGymEnvHER(urdfRoot=robot_data.getDataPath(), renders=rend, useIK=0,
            isDiscrete=discreteAction, action_space = action_space,
            fixedPositionObj = fixed, includeVelObs = True, object_position=object_position)

    env = Monitor(env, log_dir, allow_early_resets=True)

    goal_selection_strategy = 'future'
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    # Wrap the model

    model = HER(CustomPolicy, env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                verbose=1,tensorboard_log="../pybullet_logs/panda_push_ddpg/stable_baselines/pushing_DDPG_HER_PHASE_1", buffer_size=1000000,batch_size=256,
                random_exploration=0.3, action_noise=action_noise)

    load_policy = False
    if (load_policy):
        model = HER.load("../policies/pushing_DDPG_HER_PHASE_1.pkl", env=env, n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        tensorboard_log="../pybullet_logs/panda_push_ddpg/stable_baselines/pushing_DDPG_HER_PHASE_1",
        buffer_size=1000000,batch_size=256,random_exploration=0.3, action_noise=action_noise)

    print("Training Phase")
    model.learn(timesteps,log_interval=100, callback = callback)
    print("Saving Policy PHASE_1")
    model.save("../policies/pushing_DDPG_HER_PHASE_1")

if __name__ == "__main__":
    main()
