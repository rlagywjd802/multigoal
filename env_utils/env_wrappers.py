#!/usr/bin/env python
import numpy as np
import sys
import os
import numpy as np
import scipy.misc
import yaml
import gym
from gym.spaces import box as gym_box
import argparse
import matplotlib.pyplot as plt
import time

from gym import Wrapper
from multigoal.utils import print_format as pf
from gym.spaces import box as gym_box
from gym.spaces import Tuple as gym_tuple

from gym import Wrapper

from e2eap_training.rllab_utils.core.keras_tools import KerasClassifier, blocksDummyClassifier



class obsTupleWrap(Wrapper):
    """
    Wraps existing observation space into a tuple
    The wrapper can also add an action into observation tuple: (obs,act)
    One could also swap some dimensions if need be
    (useful for images, if you need to move channel dimension for compatibility with some frameworks)
    """
    def __init__(self, env, obs_dimorder=None, add_action_to_obs=False):
        super(obsTupleWrap, self).__init__(env)
        self.obs_dimorder = obs_dimorder
        self.add_action_to_obs = add_action_to_obs
        if obs_dimorder is not None:
            new_obsspace = gym_box.Box(np.transpose(self.env.observation_space.low,  self.obs_dimorder),
                        np.transpose(self.env.observation_space.high, self.obs_dimorder))
            if self.add_action_to_obs:
                self.observation_space = gym_tuple((new_obsspace, self.env.action_space))
            else:
                self.observation_space = gym_tuple((new_obsspace,))
        else:
            if self.add_action_to_obs:
                self.observation_space = gym_tuple((self.env.observation_space, self.env.action_space))
            else:
                self.observation_space = gym_tuple((self.env.observation_space,))

    @property
    def goal(self):
        return self.env.goal

    @goal.setter
    def goal(self, goal):
        self.env.goal = goal

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        if self.obs_dimorder is not None:
            ob = np.transpose(ob, self.obs_dimorder)

        if self.add_action_to_obs:
            ob = (ob, action)
        else:
            ob = (ob,)
        return ob, reward, done, info

    def _reset(self):
        ob = self.env.reset()
        if self.obs_dimorder is not None:
            ob = np.transpose(ob, self.obs_dimorder)

        if self.add_action_to_obs:
            ob = (ob, np.zeros_like(self.action_space.low))
        else:
            ob = (ob,)
        return ob

    def use_mnist_stop_criteria(self):
        return self.env.use_mnist_stop_criteria

    def use_mnist_stop_criteria(self, use):
        # print('stopWrap: use_mnist_stop_criteria = ', self.env.use_mnist_stop_criteria)
        self.env.use_mnist_stop_criteria = use

    def use_mnist_reward(self):
        return self.env.use_mnist_reward

    def use_mnist_reward(self, use):
        # print('stopWrap: use_mnist_reward = ', self.env.use_mnist_reward)
        self.env.use_mnist_reward = use

    @property
    def use_distance2center_stop_criteria(self):
        return self.env.use_distance2center_stop_criteria

    @use_distance2center_stop_criteria.setter
    def use_distance2center_stop_criteria(self, use):
        self.env.use_distance2center_stop_criteria = use

    def reload_model(self, pose=None, goal=None):
        ob = self.env.reload_model(pose, goal)
        if self.obs_dimorder is not None:
            ob = np.transpose(ob, self.obs_dimorder)

        if self.add_action_to_obs:
            ob = (ob, np.zeros_like(self.action_space.low))
        else:
            ob = (ob,)
        return ob

    def pose2goal(self, pose):
        return self.env.pose2goal(pose)

    def is_multigoal(self):
        return self.env.is_multigoal()


# ===========================
#   Main function
# ===========================
def main(argv):
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     "-e", "--env",
    #     help="Environment name. Ex: Pendulum-v0"
    # )
    args = parser.parse_args()

    #############################
    # Init plottting
    fig = plt.figure(1)
    # ax = plt.subplot(111)
    plt.show(block=False)

    ############################
    # x0 = [1., 1., 0, 0] # [x0,y0,vx0,vy0]
    # target = [20., 40.]
    render = True
    time_limit = 50

    env_name = 'Reacher-v1'
    env = gym.make(env_name)
    env = obsTupleWrap(env, max_episode_steps=time_limit)
    env.max_episode_steps = time_limit
    s = env.reset()

    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high)
        print('Action space:', env.action_space.low, env.action_space.high)
    except:
        print('Observation space:', env.observation_space.spaces[0].low, env.observation_space.spaces[0].high)
        print('Action space:', env.action_space.low, env.action_space.high)

    # action = (env.action_space.low + env.action_space.high) / 2
    action = [0.0, 0.01]
    observations = []

    t = 0
    plot_step = 10

    while True:
        if render:
            env.render()
        s, r, done, info = env.step(action)
        observations.append(s)
        # print('Step: ', t, ' Obs:', s)

        if t % plot_step == 0:
            plt.clf()

            observations_arr = np.array(observations)
            # print('obs arr shape:', observations_arr.shape)
            dimenstions = observations_arr.shape[1]
            # for dim in range(dimenstions):
            for dim in [4,5,6,7]:
                plt.plot(observations_arr[:,dim])

            plt.legend([str(x) for x in range(observations_arr.shape[1])])
            plt.pause(0.01)
            plt.draw()


        if done:
            break
        t += 1

    time.sleep(100)


if __name__ == '__main__':
    main(sys.argv)