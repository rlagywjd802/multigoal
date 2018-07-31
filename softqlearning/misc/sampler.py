import numpy as np
import time

from rllab.misc import logger


def rollout(env, policy, path_length, render=False, speedup=None):
    Da = np.prod(env.action_space.shape)
    Do = np.prod(env.observation_space.flat_dim)    ## edited

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length + 1, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length, ))
    rewards = np.zeros((path_length, ))
    agent_infos = []
    env_infos = []

    t = 0
    for t in range(path_length):

        action, agent_info = policy.get_action(observation)
        next_obs, reward, terminal, env_info = env.step(action)

        agent_infos.append(agent_info)
        env_infos.append(env_info)

        actions[t] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if terminal:
            break

    observations[t + 1] = observation

    path = {
        'observations': observations[:t + 1],
        'actions': actions[:t + 1],
        'rewards': rewards[:t + 1],
        'terminals': terminals[:t + 1],
        'next_observations': observations[1:t + 2],
        'agent_infos': agent_infos,
        'env_infos': env_infos
    }

    return path


def rollouts(env, policy, path_length, n_paths):
    paths = list()
    for i in range(n_paths):
        paths.append(rollout(env, policy, path_length))

    return paths


class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policies = None
        self.pool = None

    def initialize(self, env, policies, pool):
        self.env = env
        self.policies = policies
        self.pool = pool

    def sample(self):
        raise NotImplementedError

    def batch_ready(self):
        # print('pool size:',self.pool.size)
        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()

    def log_diagnostics(self):
        logger.record_tabular('pool-size', self.pool.size)


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

        ##
        self.start_pose = None
        self.goal = None
        self.start_pose_id = 0
        self.goal_id = 0


    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action, _ = self.policies['seek'].get_action(self._current_observation)
        next_observation, reward, terminal, info = self.env.step(action)

        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation)

        if terminal or self._path_length >= self._max_path_length:
            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation

    def sample_sql(self, animated=False):
        if self._current_observation is None:
            self.env.reset()
            self.start_pose, self.start_pose_id = self.policies['hide'].sample_one_start()
            self.start_pose = np.array(self.start_pose)
            self.goal, self.goal_id = self.policies['hide'].sample_one_goal()
            self._current_observation = self.env.env.env.reload_model(pose=self.start_pose, goal=self.goal)
            print("++++++++++++++++++++++++++++++++++++")
            print('start_pose:', self.start_pose[0][0:2], '     goal:', self.goal[0][0:2])
            print('start_pose:', np.array(self._current_observation[0][0:2])*2.4, '     goal:', np.array(self._current_observation[0][-2:])*2.4)
            if animated:
                self.env.render()

        action, _ = self.policies['seek'].get_action(self._current_observation)
        # print('action:', action)
        if animated:
            self.env.render()
        next_observation, reward, terminal, info = self.env.step(action)

        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observation=self._current_observation[0],
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation)

        if terminal or self._path_length >= self._max_path_length:
            print('path_length:', self._path_length)
            # starts에 대한 rewards값 저장
            goal_reached = float(self._path_length<self._max_path_length)
            if self.policies['hide'].reverse_mode:
                self.policies['hide'].rewards[self.start_pose_id].append(goal_reached)  # self.reverse_mode = True
            else:
                self.policies['hide'].rewards[self.goal_id].append(goal_reached)
            # print("***************hide_rewards****************")
            # print(self.policies['hide'].rewards)
            # print("***************seek_rewards****************")
            # print(self.policies['seek'].rewards)
            # print("*******************************************")
            self.policies['seek'].reset()  ##
            path_length = self._path_length
            self._current_observation = None
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return
            self._path_return = 0
            self._n_episodes += 1
            return self.start_pose, True, path_length
        else:
            self._current_observation = next_observation
            return self.start_pose, 0, None

    def log_diagnostics(self):
        super(SimpleSampler, self).log_diagnostics()
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)


class DummySampler(Sampler):
    def __init__(self, batch_size, max_path_length):
        super(DummySampler, self).__init__(
            max_path_length=max_path_length,
            min_pool_size=0,
            batch_size=batch_size)

    def sample(self):
        pass
