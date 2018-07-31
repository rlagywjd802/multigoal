import argparse
import os
import sys
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

## env - BlocksSimpleXYQ
import multigoal.envs.blocks_simple as bsmp
import rllab.misc.logger as logger

## for saving path info
from rllab.misc import tensor_utils
from rllab_utils.misc import tensor_utils as e2e_tensor_utils
from rllab_utils.misc.glob_config import glob_config



# ===========================
#   Agent
# ===========================
class simpleAgent():
    """
    Simple agent for testing
    """
    def __init__(self, hide=False, force=[0.0, 1.0], hide_steps_num=5):
        self.hide = hide
        self.hide_step_num = hide_steps_num
        self.step = 0
        self.force = np.array(force)

    def get_action(self, obs):
        self.step += 1
        if self.hide:
            if self.step > self.hide_step_num:
                action = self.force, {}
            else:
                action = -self.force, {}
            #print('Hide action: ', action)
        else:
            action = self.force, {}
            #print('Seek action: ', action)
        return action

    def reset(self):
        self.step = 0
        pass

class brownianAgent(object):
    def __init__(self, env, mode,
                 r_min, r_max, N_min, N_max, N_window,
                 action_variance, goal_states,
                 oversample_times=5, step_limit=1,
                 starts_new_num=300, starts_new_select_prob=0.7,
                 starts_old_max_size=10000, render_rollouts=False,
                 sample_alg=0, obs_indx=0, use_classifier_sampler=False, plot=True, out_dir=None,
                 multigoal=True, prob_weight_power=1.0, classif_bufsize=None, classif_label_alg=0, classif_weight_func=0,
                 ):
        self.env = env

        self._multigoal = multigoal
        self.reverse_mode = True

        self.r_min = r_min
        self.r_max = r_max

        self.N_max = N_max

        self.plot = plot
        self.out_dir = out_dir
        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        self.N_window = int(N_window)
        self.N_min = int(N_window)

        self.oversample_times = oversample_times
        self.mode = mode
        self.step_limit = step_limit
        self.starts_new_select_num = starts_new_num
        self.starts_new_select_prob = starts_new_select_prob
        self.goal_states = copy.deepcopy(goal_states)
        self.init_goals = copy.deepcopy(goal_states)

        self.starts_old_max_size = int(starts_old_max_size)
        if classif_bufsize is None:
            self.classif_bufsize = starts_old_max_size
        else:
            self.classif_bufsize = classif_bufsize

        self.action_variance = action_variance
        self.render_rollouts = render_rollouts

        self.sample_alg = sample_alg
        self.starts = copy.deepcopy(self.goal_states)
        self.starts_old = copy.deepcopy(self.goal_states)
        self.starts_old_obs = [self.env.unwrapped.state2obs(state) for state in self.goal_states]
        print("Brownian Agent Init : initial starts : ", self.starts)
        print("Brownian Agent Init : initial starts_old : ", self.starts_old)

        self.starts_rejected_obs = [] # Rejected starts (too hard or too easy)
        self.starts_rejected = []
        self.starts_rejected_labels = [] #0 - too hard, 1 - too easy
        self.rejected_starts_vec = []
        # print('starts cur/old:', self.starts, self.starts_old)

        self.good_starts_for_classifier = []

        # Need val to be a list since for every rollout we have a reward and we calculate average at the end
        self.reset_rewards(val=[1.0] * int(self.N_min + 1))
        print("Brownian Agent Init : initial rewards : ", self.rewards)

        self.brownian_samples_num = 0

        self.itr = 0

    def reset(self):
        self.action_prev = np.zeros_like(self.env.action_space.low)

    def reset_rewards(self, val=None):
        if val is None:
            val = []
        if not isinstance(val, list):
            val = [val]
        self.rewards = [copy.deepcopy(val) for x in range(len(self.starts))]

    def sample_one_start(self):
        """
        This function will sample single start from the current start pool
        and single goal from the pool of old starts
        :return:
        """
        id = np.random.randint(low=0, high=len(self.starts))
        state = self.starts[id]
        return state, id

    def sample_one_goal(self):
        if self._multigoal:
            pools = [self.starts_old, self.starts]
            pool_id = np.random.binomial(1, p=self.starts_new_select_prob)
            pool2sample = pools[pool_id]
            id = np.random.randint(low=0, high=len(pool2sample))
            state = pool2sample[id]
        else:
            id = np.random.randint(low=0, high=len(self.init_goals))
            state = self.init_goals[id]

        # print('id/prob',pool_id, self.starts_new_select_prob)
        return state, id

    def sample_from_either_pool(self, pools, p):
        pool_id = np.random.binomial(1, p=p)    # return 0 or 1 as the probability of p
        pool2sample = pools[pool_id]            # select between starts_old(1-p) and starts_finished(p)
        # print('pool_id %d , len %d' % (pool_id, len(pool2sample)))
        id = np.random.randint(low=0, high=len(pool2sample))
        state = pool2sample[id]                 # select one data from pool2sample
        return state, id

    def select_starts(self, r_min=None, r_max=None, N_min=None, N_max=None, N_window=None):
        """
        This function will select starts based on accumulated statistics of rewards
        It would also clear this statistics for the next iteration
        :return:
        """
        self.itr += 1
        if r_min is None:
            r_min = self.r_min
        if r_max is None:
            r_max = self.r_max
        if N_min is None:
            N_min = self.N_min
        if N_max is None:
            N_max = self.N_max
        if N_window is None:
            N_window = self.N_window

        ## Keeping the good starts only
        starts = []
        rewards = []
        starts_finished = []
        starts_finished_obs = []
        starts_rejected_num = 0
        # print(self.__class__.__name__, ': starts: ', self.starts)

        self.good_starts_for_classifier = []
        self.good_obs_for_classifier = []
        self.bad_starts_for_classifier = []
        self.bad_obs_for_classifier = []
        # hard_easy_border = (r_max + r_min) / 2.0
        hard_easy_border = 0.5

        #print("strats: ", self.starts)
        for id, start in enumerate(self.starts):
            print('%d: ' % id, start[0][0:2])
            print('%d: ' % id, self.rewards[id])
            N_cur = len(self.rewards[id])
            ## Getting labels for the classifier
            if N_cur > min(2, N_min):
                reward_avg_classif = np.mean(self.rewards[id][-N_window:])
                if reward_avg_classif < hard_easy_border:
                    self.good_starts_for_classifier.append(start)
                    self.good_obs_for_classifier.append(self.env.unwrapped.state2obs(start))
                else:
                    self.bad_starts_for_classifier.append(start)
                    self.bad_obs_for_classifier.append(self.env.unwrapped.state2obs(start))

            if N_cur < N_min:
                starts.append(start)
                rewards.append(self.rewards[id])
            else:
                reward_avg = np.mean(self.rewards[id][-N_window:])
                print('avg reward: ', reward_avg)
                if r_min <= reward_avg:
                    if reward_avg > r_max:
                        starts_rejected_num += 1
                        self.starts_rejected_obs.append(self.env.unwrapped.state2obs(start))
                        self.starts_rejected.append(start)
                        self.starts_rejected_labels.append(1)
                    elif N_cur >= N_window:
                        starts_finished.append(start)
                        starts_finished_obs.append(self.env.unwrapped.state2obs(start))
                    else:
                        starts.append(start)
                        rewards.append(self.rewards[id])
                elif N_cur < N_max:
                    starts.append(start)
                    rewards.append(self.rewards[id])
                else:
                    starts_rejected_num += 1
                    self.starts_rejected_obs.append(self.env.unwrapped.state2obs(start))
                    self.starts_rejected.append(start)
                    self.starts_rejected_labels.append(0)

        self.starts = starts
        self.rewards = rewards

        self.starts_old.extend(starts_finished)
        self.starts_old = self.starts_old[-self.starts_old_max_size:]

        self.starts_old_obs.extend(starts_finished_obs)
        self.starts_old_obs = self.starts_old_obs[-self.starts_old_max_size:]

        self.starts_rejected_obs = self.starts_rejected_obs[-self.starts_old_max_size:]
        self.rejected_starts_vec.append(len(self.starts_rejected_obs))

        samples_required = self.starts_new_select_num  - len(self.starts)
        logger.log('Starts rejected %d' % starts_rejected_num)

        return starts_finished, samples_required

    def get_action(self, obs=None):
        action = self.action_prev \
                 + np.random.normal(loc=np.zeros_like(self.env.action_space.low),
                                    scale=self.action_variance)
        # print('action:', action)
        return action, {'step_limit': self.step_limit}

    def sample_nearby(self, starts_finished, starts_new_select_num, itr=None, clear_figures=True):
        """
        Samples a bunch of new initial conditions around the existing good set of initial conditions
        :return:
        """
        # Sometimes after filtering no starts are left to populate from
        # thus it would make sense to re-populate from the old starts instead

        if starts_new_select_num == 0:
            return

        # In case re-sampling happens because of pure rejection
        # and no finished starts appeared - resample from old starts
        if len(starts_finished) == 0:
            starts_finished = self.starts_old

        oversampled_starts = []
        if self.sample_alg == 1:
            oversampled_starts.extend(starts_finished)

        samples_before = len(oversampled_starts)

        while len(oversampled_starts) < starts_new_select_num * self.oversample_times:
            if self.sample_alg == 0:
                pools = [self.starts_old, starts_finished]
            else:
                pools = [self.starts_old, oversampled_starts]

            start_state, id_temp = self.sample_from_either_pool(pools=pools,
                                                           p=self.starts_new_select_prob)
            # print('Start: ', start_state, ' id:', id, 'starts num:', len(self.starts))
            # start_state = np.array(start_state)
            # print('brown_agent: hide rollout ...')
            path = rollout_hide(env=self.env, agents={'hide': self}, mode=self.mode,
                               init_state=start_state, init_goal=start_state)
            oversampled_starts.extend(path['states'][1:]) #Excluding the first state since it is already in the pile
            # time.sleep(1)

        self.brownian_samples_num += (len(oversampled_starts) - samples_before)

        ## Sampling stage from oversampled components
        logger.log('Rejected/Accepted/Min starts for classif: %d / %d' %
                   (len(self.starts_rejected_obs), len(self.starts_old_obs)))

        # If using classif, sample according to the probabilities of samples being successful
        # Otherwise sample uniformly
        starts_new_selected_ids = np.random.choice(len(oversampled_starts), starts_new_select_num, replace=False)
        starts_new_selected = [oversampled_starts[id] for id in starts_new_selected_ids]

        if self.plot:
            plt.figure(29)
            plt.clf()
            plt.plot(self.rejected_starts_vec)
            plt.title('Rejected samples num')
        # logger.log('Rejected samples num: %d', self.rejected_starts_vec[-1])
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        self.starts.extend(starts_new_selected)
        self.rewards.extend([[] for i in range(starts_new_select_num)])
        # print('starts: ', self.starts, 'shape: ', self.starts.shape)

        return self.brownian_samples_num

    def plot_starts(self, starts, rewards, xlim=[-1, 1], ylim=[-1, 1],
                    img_name=None, clear=True, env=None, fig_id=20, out_dir=None):
        plt.figure(fig_id)
        if clear:
            plt.clf()
        if len(starts) == 0:
            return
        # I have to do that because goals have different length
        x,y, x_dead, y_dead = [],[],[],[]
        colors, colors_dead = [],[]
        min_color, max_color = np.array([0,0,1]),np.array([1,0,0])

        if env.spec.id[:7] == 'Reacher':
            goals_temp = [v[2] for i, v in enumerate(starts)]
            starts = goals_temp
            xlim = [-0.22, 0.22]
            ylim = [-0.22, 0.22]
            scale = 1.0

        if env.spec.id[:12] == 'BlocksSimple':
            goals_temp = [v[0] for i, v in enumerate(starts)]
            # print('starts:', starts)
            # print('goal_temp',goals_temp)
            starts = goals_temp
            scale = 2.4

        for i,v in enumerate(starts):
            v = np.array(v).flatten() / scale
            rew_cur = rewards[i]
            x.append(v[0])
            y.append(v[1])
            rew_cur = np.clip(np.mean(rew_cur),0.,1.)
            color = rew_cur * max_color + (1.-rew_cur)*min_color
            colors.append(color)

        x_dead.extend(x)
        y_dead.extend(y)
        colors_dead.extend(colors)
        x = x_dead
        y = y_dead
        colors = colors_dead
        # x = np.array(x)
        # y = np.array(y)
        plt.scatter(x, y, s=20, c=colors, alpha=0.5)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(img_name)
        plt.pause(.01)
        plt.show()
        plt.pause(.01)
        if img_name is not None and out_dir is not None:
            plt.savefig(out_dir + img_name + '.jpg')


# ===========================
#   rollouts
# ===========================
def rollout(env):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    obs = env.reset()
    path_length = 0
    max_path_length = env.max_episode_steps

    while path_length < max_path_length:
        env.render()
        a, agent_info = env.action_space.sample()
        obs_next, r, d, env_info = env.step(a)

        observations.append(obs)
        actions.append(a)
        rewards.append(r)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        obs = obs_next
        if d:
            break

    paths = dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

    return paths

def rollout_hide_seek(env, agents, mode, max_path_length=np.inf):
    hide_observations = []
    hide_actions = []
    hide_rewards = []
    hide_agent_infos = []
    hide_env_infos = []

    print("rollout : HIDE")
    obs = env.reset()
    print('before --> goal hide: ', env.goal, 'obs:', obs)
    obs = env.reload_model()
    print('after --> goal hide: ', env.goal, 'obs:', obs)

    agents['hide'].reset()
    hide_path_length = 0
    ## Test different modes

    while hide_path_length < max_path_length:
        a, agent_info = agents['hide'].get_action(obs)

def rollout_hide(env, agents, mode, max_path_length=np.inf,
                 init_state = None, init_goal=None):
    # Reset the model configuration
    env.reset()
    obs = env.reload_model(pose=init_state, goal=init_goal)

    hide_observations = []
    hide_states = []
    hide_actions = []
    hide_rewards = []
    hide_agent_infos = []
    hide_env_infos = []

    print("rollout : HIDE")
    agents['hide'].reset()
    hide_path_length = 0

    while hide_path_length < max_path_length:
        env.render()
        a, agent_info = agents['hide'].get_action(obs)
        hide_states.append(env.unwrapped.get_all_pose())
        #print('-------> state: ', obs[0:2])
        obs_next,r, d, env_info = env.step(a)

        hide_observations.append(obs)
        hide_rewards.append(r)
        hide_actions.append(a)
        hide_agent_infos.append(agent_info)
        hide_env_infos.append(env_info)
        hide_path_length += 1
        obs = obs_next
        if d:
            break

    hide_paths = dict(
        observations=e2e_tensor_utils.stack_tensor_list(hide_observations),
        actions=tensor_utils.stack_tensor_list(hide_actions),
        rewards=tensor_utils.stack_tensor_list(hide_rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(hide_agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(hide_env_infos),
        states=hide_states,
    )
    print('Episode done:', hide_path_length)
    return hide_paths

def rollout_seek(env, agents, mode, max_path_length=np.inf):
    seek_observations = []
    seek_actions = []
    seek_rewards = []
    seek_agent_infos = []
    seek_env_infos = []

    print("rollout : SEEK")
    obs = env.reset()
    agents['seek'].reset()
    seek_path_length = 0

    while seek_path_length < max_path_length:
        env.render()
        a, agent_info = agents['seek'].get_action(obs)
        obs_next, r, d, env_info = env.step(a)
        seek_observations.append(obs)
        seek_rewards.append(r)
        seek_actions.append(a)
        seek_agent_infos.append(agent_info)
        seek_env_infos.append(env_info)
        seek_path_length += 1
        obs = obs_next
        if d:
            break

    seek_paths = dict(
        observations=e2e_tensor_utils.stack_tensor_list(seek_observations),
        actions=tensor_utils.stack_tensor_list(seek_actions),
        rewards=tensor_utils.stack_tensor_list(seek_rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(seek_agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(seek_env_infos),
    )

    seek_paths['actions'] = seek_paths['actions'].astype(glob_config.dtype)
    seek_paths['rewards'] = seek_paths['rewards'].astype(glob_config.dtype)

    return {'seek':seek_paths}

def rollout_brownian(env, agents, mode, max_path_length=np.inf):

    ##############################################
    ## HIDE AGENT
    start_pose, start_pose_id = agents['hide'].sample_one_start()   # sample one start from starts
    start_pose = np.array(start_pose)

    ##############################################
    ## SEEK AGENT
    env.reset()
    goal, goal_id = agents['hide'].sample_one_goal()   # sample one goal from pools(starts & starts_old)
    obs = env.reload_model(pose=start_pose, goal=goal)

    print("rollout: SEEK")

    seek_observations = []
    seek_actions = []
    seek_rewards = []
    seek_agent_infos = []
    seek_env_infos = []

    agents['seek'].reset()
    seek_path_length = 0

    while seek_path_length < max_path_length:
        env.render()
        a, agent_info = agents['seek'].get_action(obs)
        obs_next, r, d, env_info = env.step(a)
        seek_observations.append(obs)
        seek_rewards.append(r)
        seek_actions.append(a)
        seek_agent_infos.append(agent_info)
        seek_env_infos.append(env_info)
        seek_path_length += 1
        if d:
            break
        obs = obs_next


# ===========================
#   test rollouts
# ===========================
def test_rollout(env, rollouts_num):
    print("========================")
    print("Test Rollout Start")
    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high)
        print('Action space:', env.action_space.low, env.action_space.high)
    except:
        print('Observation space:', env.observation_space.spaces[0].low, env.observation_space[0].spaces[0].high)
        print('Action space:', env.action_space[0].spaces[0].low, env.action_space[0].spaces[0].high)

    s = env.reset()

    for path_i in range(rollouts_num):
        paths = rollout(env=env)
        observations_arr = np.array(paths['observations'])
        time = len(paths['rewards'])
        print('len path: ', time)

def test_seek_rollout(env, rollouts_num):
    print("========================")
    print("Test Seek Rollout Start")
    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high)
        print('Action space:', env.action_space.low, env.action_space.high)
    except:
        print('Observation space:', env.observation_space.spaces[0].low, env.observation_space[0].spaces[0].high)
        print('Action space:', env.action_space[0].spaces[0].low, env.action_space[0].spaces[0].high)

    s = env.reset()
    agents = {'hide': simpleAgent(hide=True), 'seek': simpleAgent()}

    for path_i in range(rollouts_num):
        paths = rollout_seek(env=env, agents=agents, max_path_length=env.max_episode_steps, mode='reach_center_and_stop')
        # observations = np.concatenate([paths['hide']['observations'][0], paths['seek']['observations'][0]])
        # observations_arr = np.array(observations)
        # time_seek = len(paths['seek']['rewards'])
        # time_hide = len(paths['hide']['rewards'])
        # print('len seek/hide:', time_seek, time_hide)

def test_hide_rollout(env, rollouts_num):
    print("========================")
    print("Test Hide Rollout Start")
    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high)
        print('Action space:', env.action_space.low, env.action_space.high)
    except:
        print('Observation space:', env.observation_space.spaces[0].low, env.observation_space[0].spaces[0].high)
        print('Action space:', env.action_space[0].spaces[0].low, env.action_space[0].spaces[0].high)

    s = env.reset()
    agents = {'hide': simpleAgent(hide=True), 'seek': simpleAgent()}

    for path_i in range(rollouts_num):
        paths = rollout_hide(env=env, agents=agents, max_path_length=env.max_episode_steps, mode='reach_center_and_stop')
        # observations = np.concatenate([paths['hide']['observations'][0], paths['seek']['observations'][0]])
        # observations_arr = np.array(observations)
        # time_seek = len(paths['seek']['rewards'])
        # time_hide = len(paths['hide']['rewards'])
        # print('len seek/hide:', time_seek, time_hide)

def test_brown_multigoal_rollout(env, rollouts_num):
    print("========================")
    print("Test Brownian Multigoal Rollout Start")
    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high)
        print('Action space:', env.action_space.low, env.action_space.high)
    except:
        print('Observation space:', env.observation_space.spaces[0].low, env.observation_space[0].spaces[0].high)
        print('Action space:', env.action_space[0].spaces[0].low, env.action_space[0].spaces[0].high)

    # Init plottting
    fig = plt.figure(1)
    plt.show(block=False)

    s = env.reset()
    iter_num = 5
    hide_tmax = 5
    #init_state = ([0.5, 0.5, 0.08, 1.0, 0., 0., 0., 0.5, 0.5], [0., 0., 0., 0., 0., 0., 0., 0.])
    init_state = ([0, 0, 0.08, 1.0, 0., 0., 0., 0, 0], [0., 0., 0., 0., 0., 0., 0., 0.])
    agents = {'hide': brownianAgent(env=env,
                                  mode='reach_center_and_stop',
                                  r_min=0.1, r_max=0.8,
                                  N_min=1, N_max=20, N_window=10,
                                  action_variance=0.5, oversample_times=5,
                                  step_limit=hide_tmax,
                                  starts_new_num=10, starts_new_select_prob=0.8,
                                  goal_states=[init_state],
                                  starts_old_max_size=100,
                                  render_rollouts=True),
              'seek': simpleAgent()}

    for iter in range(iter_num):
        print("-----------------------------------")
        print("Sampling ...")
        starts_finished, samples_required_num = agents['hide'].select_starts()
        print('Samples required: ', samples_required_num)
        agents['hide'].sample_nearby(starts_finished, samples_required_num)
        print('Samples acquired: ', len(agents['hide'].starts))
        # for start in agents['hide'].starts:
        #     print(start, type(start))
        # time.sleep(3)
        for path_i in range(rollouts_num):
            paths = rollout_brownian(env=env, agents=agents, mode='reach_center_and_stop', max_path_length=env.max_episode_steps)


# ===========================
#   Main function
# ===========================

def main(argv):
    # If blocks_multigoal is true, then target doesn't work
    # Else, goal position is determined by target value for every rollout
    blocks_multigoal = True
    timelen_max = 100
    blocks_simple_xml = "blocks_simple_maze1.xml"
    target = [-1.0, 0.0]

    env = bsmp.BlocksSimpleXYQ(multi_goal=blocks_multigoal,
                               time_limit=timelen_max,
                               env_config=blocks_simple_xml,
                               goal=target)

    test_mode = 3
    rollout_num = 10
    # test_mode
    # 0 - normal rollout
    # 1 - hide/seek rollout
    # 2 - seek rollout
    if test_mode == 0:
        test_rollout(env, rollout_num)
    elif test_mode == 1:
        test_hide_rollout(env, rollout_num)
    elif test_mode == 2:
        test_seek_rollout(env, rollout_num)
    elif test_mode == 3:
        test_brown_multigoal_rollout(env, rollout_num)

if __name__ == '__main__':
    main(sys.argv)

