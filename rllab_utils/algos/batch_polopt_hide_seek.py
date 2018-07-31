import copy
import time
import os
from collections import deque
import inspect

import numpy as np
import matplotlib.pyplot as plt

import rllab.misc.logger as logger
import rllab.plotter as plotter
import theano
from e2eap_training.utils import print_format as pf
from rllab.algos import util
from rllab.algos.base import RLAlgorithm
from rllab.misc import special
from rllab.misc import tensor_utils
from multigoal.rllab_utils.misc.utils import sample_multidim
import multigoal.rllab_utils.envs.globals as glob
from multigoal.rllab_utils.misc import tensor_utils as e2e_tensor_utils
from multigoal.rllab_utils.misc.utils import rollout_hide_seek, rollout_seek, rollout
from multigoal.rllab_utils.sampler import parallel_sampler_comp as parallel_sampler

import pickle
import joblib
from scipy import io as sio

from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# import seaborn as sns

# -------------------

## TODO
# - dynamics model does not work yet. Things to fix for dynamics:
#   - parallel sampler
#   - train_dynamics() function
#   - ReplayPool since it does not use tuple observations

class myPlotter:
    def __init__(self, out_dir=None, fig_start=1, fig_size=(6, 5), graph_names=None):
        # Turn on interactive plotting
        plt.ion()

        # Create the main, super plot
        if graph_names is None:
            graph_names = ['xy_time', 'xy_time_test', 'xy_timerew', 'xy_taskclassrew', 'xy_tasklabels', 'xy_tasklabels_train']


        self.colors_all = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.figures = {}
        self.axis = {}
        self.colors = {}
        fig_free = fig_start - 1
        self.graph_names = []
        gr_indx = -1
        for name in graph_names:
            gr_indx += 1
            self.graph_names.append(name)
            fig_free += 1
            self.figures[name] = plt.figure(fig_free, figsize=fig_size)
            self.axis[name] = self.figures[name].add_subplot(111)
            self.axis[name].set_title(name, fontsize=18)
            self.colors[name] = self.colors_all[gr_indx % len(self.colors_all)]

        plt.figure(11, figsize=fig_size)
        plt.figure(12, figsize=fig_size)

        plt.show()
        if out_dir is None:
            out_dir = '.'

        if out_dir[-1] != '/':
            out_dir += '/'
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def plot_goals(self, goals, color, xlim=[-1,1], ylim=[-1,1], img_name=None, clear=False, name='', scale=2.4, env=None, fig_id=11):
        plt.figure(fig_id)
        if clear:
            plt.clf()
        if len(goals) == 0:
            return

        if env.spec.id[:7] == 'Reacher':
            goals_temp = [v[2] for i, v in enumerate(goals)]
            goals = goals_temp
            xlim = [-0.22, 0.22]
            ylim = [-0.22, 0.22]

        if env.spec.id[:12] == 'BlocksSimple':
            goals_temp = [v[0] for i, v in enumerate(goals)]
            goals = goals_temp

        # I have to do that because goals have different length
        x,y = [],[]
        goals_count = 0
        for i,v in enumerate(goals):
            v = np.array(v).flatten() / scale
            x.append(v[0])
            y.append(v[1])
            goals_count += 1

        x = np.array(x)
        y = np.array(y)
        print('plotting %d %s goals' % (goals_count, name))
        plt.scatter(x, y, s=20, c=color, alpha=0.5)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title('goals new/old')
        plt.pause(.01)
        plt.show()
        plt.pause(.01)
        img_name = 'goals'
        if img_name is not None:
            plt.savefig(self.out_dir + img_name + '.jpg')

    def plot_goal_rewards(self, goals, rewards, xlim=[-1,1], ylim=[-1,1], img_name=None, clear=False, name='', show_dead_points=False, scale=2.4, env=None, fig_id=12):
        plt.figure(fig_id)
        if clear:
            plt.clf()
        if len(goals) == 0:
            return
        # I have to do that because goals have different length
        x,y, x_dead, y_dead = [],[],[],[]
        colors, colors_dead = [],[]
        min_color, max_color = np.array([0,0,1]),np.array([1,0,0])

        if env.spec.id[:7] == 'Reacher':
            goals_temp = [v[2] for i, v in enumerate(goals)]
            goals = goals_temp
            xlim = [-0.22, 0.22]
            ylim = [-0.22, 0.22]

        if env.spec.id[:12] == 'BlocksSimple':
            goals_temp = [v[0] for i, v in enumerate(goals)]
            goals = goals_temp

        for i,v in enumerate(goals):
            v = np.array(v).flatten() / scale
            rew_cur = rewards[i]
            if len(rew_cur) == 0 and show_dead_points:
                x_dead.append(v[0])
                y_dead.append(v[1])
                colors_dead.append([0,0,0])
            elif len(rew_cur) != 0 :
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
        plt.title('goal rewards')
        plt.pause(.01)
        plt.show()
        plt.pause(.01)
        if img_name is not None:
            plt.savefig(self.out_dir + img_name + '.jpg')

    def plot_xy_time(self, x, y, t, t_max, img_name=None, name='xy_time', xlim=[-1,1], ylim=[-1,1]):
        # print("-------------------->plot ", name)
        self.axis[name].cla()
        t = np.array(t)
        x = np.array(x)
        y = np.array(y)
        samples = x.size
        min_color = np.tile([0., 0., 1.], [samples, 1]) # blue
        max_color = np.tile([1., 0., 0.], [samples, 1]) # red
        color_scale = np.tile(np.expand_dims(t / t_max, axis=1), [1, 3])
        colors = color_scale * max_color + (1. - color_scale) * min_color
        self.axis[name].scatter(x, y, s=30, c=colors, alpha=0.5)
        self.axis[name].set_xlim(xlim)
        self.axis[name].set_ylim(ylim)
        self.axis[name].set_title(name)
        plt.pause(.01)
        plt.show()
        plt.pause(.01)
        if img_name is not None:
            self.figures[name].savefig(self.out_dir + img_name + '.jpg')

    def plot_xy_timereward(self, x, y, r, img_name=None, name='xy_timerew',
                           r_min=0., r_max=1., marker_size=30, xlim=[-1,1], ylim=[-1,1]):
        # print("-------------------->plot ", name)
        self.axis[name].cla()
        r = np.array(r)
        x = np.array(x)
        y = np.array(y)
        samples = x.size
        min_color = np.tile([0., 0., 1.], [samples, 1])
        max_color = np.tile([1., 0., 0.], [samples, 1])

        if r_min is None or r_max is None:
            r_min = np.min(r)
            r_max = np.max(r)

        r_adj = (np.copy(r) - r_min) / (r_max - r_min)

        color_scale = np.tile(np.expand_dims(r_adj, axis=1), [1, 3])
        colors = color_scale * max_color + (1. - color_scale) * min_color
        self.axis[name].scatter(x, y, s=marker_size, c=colors, alpha=0.5)
        self.axis[name].set_xlim(xlim)
        self.axis[name].set_ylim(ylim)
        self.axis[name].set_title(name)
        plt.pause(.01)
        plt.show()
        plt.pause(.01)
        if img_name is not None:
            self.figures[name].savefig(self.out_dir + img_name + '.jpg')

    def plot_xy_reward(self, x, y, r,
                       img_name=None, name='xy_taskclassrew',
                       r_min=None, r_max=None, marker_size=30, xlim=[-1,1], ylim=[-1,1]):
        # print("-------------------->plot ", name)
        self.axis[name].cla()
        r = np.array(r)
        x = np.array(x)
        y = np.array(y)
        samples = x.size
        min_color = np.tile([0., 0., 1.], [samples, 1]) # blue
        max_color = np.tile([1., 0., 0.], [samples, 1]) # red
        if r_min is None or r_max is None:
            r_min = np.min(r)
            r_max = np.max(r)

        r_adj = (np.copy(r) - r_min) / (r_max - r_min)

        color_scale = np.tile(np.expand_dims(r_adj, axis=1), [1, 3])
        colors = color_scale * max_color + (1. - color_scale) * min_color
        # print('colors:', colors)
        self.axis[name].scatter(x, y, s=marker_size, c=colors, alpha=0.5)
        self.axis[name].set_xlim(xlim)
        self.axis[name].set_ylim(ylim)
        self.axis[name].set_title(name)
        plt.pause(.01)
        plt.show()
        plt.pause(.01)
        if img_name is not None:
            self.figures[name].savefig(self.out_dir + img_name + '.jpg')

    def plot_goal_vec(self, goals, init_xy, labels, xlim=[-1, 1], ylim=[-1, 1], fig_id=9, img_name='xy_goal_vec', name='xy_goal_vec'):
        goals = np.array(goals)
        init_xy = np.array(init_xy)
        vec = goals - init_xy

        X = init_xy[:, 0]
        Y = init_xy[:, 1]
        U = vec[:, 0]
        V = vec[:, 1]

        colors = copy.deepcopy(labels)
        colors = ['r' if x == 1 else 'b' for x in colors]

        plt.figure(fig_id)
        plt.clf()
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1.0, color=colors)
        plt.pause(0.1)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()

        if img_name is not None:
            plt.savefig(self.out_dir + img_name + '.jpg')



class SimpleReplayPool(object):
    """Replay pool"""

    def __init__(
            self, max_pool_size, observation_shape, action_dim,
            observation_dtype=theano.config.floatX,  # @UndefinedVariable
            action_dtype=theano.config.floatX):  # @UndefinedVariable
        self._observation_shape = observation_shape
        self._action_dim = action_dim
        self._observation_dtype = observation_dtype
        self._action_dtype = action_dtype
        self._max_pool_size = max_pool_size

        self._observations = np.zeros(
            (max_pool_size,) + observation_shape,
            dtype=observation_dtype
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
            dtype=action_dtype
        )
        self._rewards = np.zeros(max_pool_size, dtype='float32')
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size = self._size + 1

    def random_batch(self, batch_size):
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(
                self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            transition_index = (index + 1) % self._max_pool_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    def mean_obs_act(self):
        if self._size >= self._max_pool_size:
            obs = self._observations
            act = self._actions
        else:
            obs = self._observations[:self._top + 1]
            act = self._actions[:self._top + 1]
        obs_mean = np.mean(obs, axis=0)
        obs_std = np.std(obs, axis=0)
        act_mean = np.mean(act, axis=0)
        act_std = np.std(act, axis=0)
        return obs_mean, obs_std, act_mean, act_std

    @property
    def size(self):
        return self._size



class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """
    def __init__(
            self,
            env,
            policies,
            baselines,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            batch_size_uniform=None, #for brownian agent mixing uniform and brownian sampling
            brown_uniform_anneal=False, #brownian training: annealing batch size for brownian agent
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            whole_paths=True,
            center_adv=True,
            positive_adv=False,
            record_states=False,
            store_paths=True,
            algorithm_parallelized=False,
            env_test=None,
            test_episodes_num = 25,

            ## EXPLORATION PARAMS
            rew_bnn_use=True,
            bnn_params=None,
            eta=1.,
            snn_n_samples=10,
            prior_sd=0.5,
            use_kl_ratio=False,
            kl_q_len=10,
            use_reverse_kl_reg=False,
            reverse_kl_reg_factor=1e-3,
            use_replay_pool=True,
            replay_pool_size=100000,
            min_pool_size=500,
            n_updates_per_sample=500,
            pool_batch_size=10,
            eta_discount=1.0,
            n_itr_update=5,
            reward_alpha=0.001,
            kl_alpha=0.001,
            normalize_reward=False,
            kl_batch_size=1,
            use_kl_ratio_q=False,
            unn_learning_rate=0.001,
            second_order_update=False,
            compression=False,
            information_gain=True,
            subsamp_bnn_obs_step=2,
            show_rollout_chance=0.0,

            ## HIDE/SEEK PARAMS
            mode = 'seek_force_only',
            use_hide=None,
            use_hide_alg=0,
            rew_hide__search_time_coeff = 0.01,  # 1.
            rew_hide__action_coeff = -0.01,  # -1.
            rew_seek__action_coeff = -0.01,  # -1.
            rew_hide__digit_entropy_coeff = 1,  # 1.
            rew_hide__digit_correct_coeff = 1,  # 1. #make <0 if we want to penalize correct predicitons by seek
            rew_hide__time_step = -0.01,  # -0.01 # Just penalty for taking time steps
            rew_hide__act_dist_coeff = -0.05,
            rew_hide__search_force_coeff = 0.1,
            rew_hide__center_reached_coeff = 0.,
            rew_seek__taskclassif_coeff = None,
            rew_seek__final_digit_entropy_coeff = 1,  # 1.
            rew_seek__digit_entropy_coeff=0.01,  # 1.
            rew_seek__final_digit_correct_coeff=1,  # 1.
            rew_seek__digit_correct_coeff = 0.01,  # 1.
            rew_seek__time_step = -0.01,  # -0.01  # Just penalty for taking time steps
            rew_seek__act_dist_coeff = -0.05,
            rew_seek__center_reached_coeff = 0.,
            rew_seek__dist2target_coeff = 0.,
            rew_seek__mnistANDtargetloc_coeff = 0.,
            rew_seek__final_mnistANDtargetloc_coeff = 0.,
            train_seek_every=1.,
            timelen_max=10,
            timelen_avg=4,
            timelen_reward_fun = 'get_timelen_reward_with_penalty',
            adaptive_timelen_avg=False,
            adaptive_percentile=False,
            adaptive_percentile_regulation_zone = [0.0, 1.0],
            timelen_avg_hist_size=100,
            task_classifier='gp',
            rew_hide__search_time_power=3,
            rew_hide__taskclassif_coeff=None,
            rew_hide__taskclassif_power=3,
            rew_hide__taskclassif_middle=0.25,
            rew_hide__actcontrol_middle=None, #action control coeff offset. If None == turned off
            taskclassif_adaptive_middle=False,
            taskclassif_adaptive_middle_regulation_zone=[0.0, 1.0],
            taskclassif_pool_size=100,
            taskclassif_use_allpoints=True,
            taskclassif_balance_positive_labels=True,
            taskclassif_add_goal_as_pos_sampl_num=1,
            taskclassif_rew_alg = 'get_prob_reward',
            taskclassif_balance_all_labels = False,
            hide_stop_improve_after=None,
            hide_tmax=None,
            starts_update_every_itr=5,
            starts_adaptive_update_itr=False,
            center_reached_ratio_max = 0.8,
            center_reached_ratio_min = 0.5,
            brown_adaptive_variance=None,
            brown_variance_min=0.1,
            brown_var_control_coeff=2.0,
            brown_tmax_adaptive=False,
            brown_t_adaptive=None,
            brown_prob_middle_adaptive=False,
            brown_success_rate_pref = 0.6,
            brown_seed_agent_period=1,
            brown_itr_min=1,
            brown_itr_max=10,
            obs_indx=1, #made it 1 since originally Blocks was using 1
            **kwargs
    ):
        """
        :param env: Environment
        :param policies: Policy
        :param baselines: Baseline
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param whole_paths: Make sure that the samples contain whole trajectories, even if the actual batch size is
        slightly larger than the specified batch_size.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :param subsamp_bnn_obs_step: (int > 0) Step to subsample image observations for dynamics model (spatial stride)
        :param taskclassif_use_allpoints: (bool) if True - uses all points from iteration instead of just the end points
        :return:
        """
        ## Processing arguments to save them in a file
        arguments = locals()
        del arguments['baselines']
        del arguments['policies']
        del arguments['env_test']
        del arguments['self']
        arguments['env'] = env.env.spec.id

        # pf.print_sec0('OPT ARGUMENTS:')
        # print(arguments)
        # pf.print_sec0_end()

        self.log_dir = logger.get_snapshot_dir()
        if self.log_dir[-1] != '/':
            self.log_dir += '/'

        self.diagnostics_dir = self.log_dir + 'diagnostics_log/'
        if not os.path.exists(self.diagnostics_dir):
            os.makedirs(self.diagnostics_dir)

        #self.params_filename = self.log_dir + 'params_opt.yaml'
        #yaml_utils.save_dict2yaml(arguments, self.params_filename)

        self.diagnostics = {}
        self.diagnostics['iter'] = []
        self.env = env
        self.env_test = env_test
        self.test_episodes_num = test_episodes_num
        self.policies = policies
        self.baselines = baselines
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.batch_size_uniform = batch_size_uniform
        self.brown_uniform_anneal = brown_uniform_anneal
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.whole_paths = whole_paths
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.subsamp_bnn_obs_step = subsamp_bnn_obs_step
        self.show_rollout_chance = show_rollout_chance

        ######################################
        ## EXPLORATION PARAMETERS
        self.rew_bnn_use = rew_bnn_use
        self.eta = eta
        self.snn_n_samples = snn_n_samples
        self.prior_sd = prior_sd
        self.use_kl_ratio = use_kl_ratio
        self.kl_q_len = kl_q_len
        self.use_reverse_kl_reg = use_reverse_kl_reg
        self.reverse_kl_reg_factor = reverse_kl_reg_factor
        self.use_replay_pool = use_replay_pool
        self.replay_pool_size = replay_pool_size
        self.min_pool_size = min_pool_size
        self.n_updates_per_sample = n_updates_per_sample
        self.pool_batch_size = pool_batch_size
        self.eta_discount = eta_discount
        self.n_itr_update = n_itr_update
        self.reward_alpha = reward_alpha
        self.kl_alpha = kl_alpha
        self.normalize_reward = normalize_reward
        self.kl_batch_size = kl_batch_size
        self.use_kl_ratio_q = use_kl_ratio_q
        self.bnn_params = bnn_params
        self.unn_learning_rate = unn_learning_rate
        self.second_order_update = second_order_update
        self.compression = compression
        self.information_gain = information_gain
        self.test_episodes_num = test_episodes_num
        self.train_seek_every = train_seek_every
        self.timelen_max = timelen_max
        self.timelen_avg  = timelen_avg
        self.timelen_avg_hist_size = timelen_avg_hist_size
        self.adaptive_timelen_avg = adaptive_timelen_avg
        self.adaptive_percentile = adaptive_percentile
        self.adaptive_percentile_regulation_zone = adaptive_percentile_regulation_zone
        self.hide_stop_improve_after = hide_stop_improve_after

        self.timelen_eplst = []
        self.hide_tmax = hide_tmax
        if self.hide_tmax is None:
            pf.print_warn('hide_tmax is None, thus hide will use the same tmax as seek')
        self.rew_hide__search_time_power = rew_hide__search_time_power
        self.rew_hide__taskclassif_power = rew_hide__taskclassif_power
        self.rew_hide__taskclassif_middle = rew_hide__taskclassif_middle
        self.timelen_reward_fun = timelen_reward_fun
        self.taskclassif_balance_positive_labels = taskclassif_balance_positive_labels #balance positive labels
        self.taskclassif_balance_all_labels = taskclassif_balance_all_labels
        self.taskclassif_rew_alg = taskclassif_rew_alg
        # ----------------------------
        # Rewards
        self.mode = mode

        self.use_hide = use_hide
        self.use_hide_alg = use_hide_alg
        self.rew_hide__search_time_coeff = rew_hide__search_time_coeff # 1.

        self.rew_hide__actcontrol_middle = rew_hide__actcontrol_middle
        self.rew_hide__action_coeff = rew_hide__action_coeff # -1.
        self.rew_hide__digit_entropy_coeff = rew_hide__digit_entropy_coeff # 1.
        self.rew_hide__digit_correct_coeff = rew_hide__digit_correct_coeff # 1. #make <0 if we want to penalize correct predicitons by seek
        self.rew_hide__time_step = rew_hide__time_step # -0.01 # Just penalty for taking time steps
        self.rew_hide__act_dist_coeff = rew_hide__act_dist_coeff  # Coeff for punishing large actions
        self.rew_hide__search_force_coeff = rew_hide__search_force_coeff # Reward hide for seek taking actions coeff
        self.rew_hide__center_reached_coeff = rew_hide__center_reached_coeff

        self.rew_seek__taskclassif_coeff = rew_seek__taskclassif_coeff
        self.rew_seek__action_coeff = rew_seek__action_coeff # -1.
        self.rew_seek__final_digit_entropy_coeff = rew_seek__final_digit_entropy_coeff # 1.
        self.rew_seek__digit_entropy_coeff = rew_seek__digit_entropy_coeff  # 1.
        self.rew_seek__digit_correct_coeff = rew_seek__digit_correct_coeff# 1.
        self.rew_seek__final_digit_correct_coeff = rew_seek__final_digit_correct_coeff  # 1.
        self.rew_seek__time_step = rew_seek__time_step # -0.01  # Just penalty for taking time steps
        self.rew_seek__act_dist_coeff = rew_seek__act_dist_coeff # Coeff for punishing large actions
        self.rew_seek__center_reached_coeff = rew_seek__center_reached_coeff
        self.rew_seek__dist2target_coeff = rew_seek__dist2target_coeff

        self.rew_seek__mnistANDtargetloc_coeff = rew_seek__mnistANDtargetloc_coeff
        self.rew_seek__final_mnistANDtargetloc_coeff = rew_seek__final_mnistANDtargetloc_coeff

        self.rew_hide__taskclassif_coeff = rew_hide__taskclassif_coeff
        self.taskclassif_obs = []
        self.taskclassif_labels = []
        self.taskclassif_obs_all = []
        self.taskclassif_labels_all = []
        self.taskclassif_obs_train_prev = None
        self.taskclassif_labels_train_prev = None

        # How many previously failed observations to retain
        # _prev observations will be used for task classifier in case we either don't have failures or successes
        self.taskclassif_obs_success_prev = []
        self.taskclassif_obs_fail_prev = []
        self.taskclassif_obs_fail_success_hist_size = int(0.25 * taskclassif_pool_size)
        self.taskclassif_adaptive_middle = taskclassif_adaptive_middle
        self.taskclassif_adaptive_middle_regulation_zone = taskclassif_adaptive_middle_regulation_zone

        self.taskclassif_pool_size = taskclassif_pool_size
        self.taskclassif_use_allpoints = taskclassif_use_allpoints

        if isinstance(task_classifier, str):
            task_classifier = task_classifier.lower()
        if task_classifier == 'gp':
            self.task_classifier = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
            self.task_classifier_type = 'gp'
        else:
            self.task_classifier_type = 'ext'
            self.task_classifier = task_classifier

        self.taskclassif_add_goal_as_pos_sampl_num = taskclassif_add_goal_as_pos_sampl_num

        ## Brownian agent parameters
        self.starts_update_every_itr = int(starts_update_every_itr)
        # self.starts_adaptive_update_itr = starts_adaptive_update_itr
        self.brown_itr_min = brown_itr_min
        self.brown_itr_max = brown_itr_max
        self.center_reached_ratio = 0
        self.center_reached_ratio_test = 0
        self.center_reached_ratio_max = center_reached_ratio_max
        self.center_reached_ratio_min = center_reached_ratio_min

        self.brown_adaptive_variance = brown_adaptive_variance
        self.brown_var_min = brown_variance_min
        self.brown_var_control_coeff = brown_var_control_coeff
        self.brown_tmax_adaptive = brown_tmax_adaptive
        self.brown_t_adaptive = brown_t_adaptive
        self.brown_prob_adaptive = brown_prob_middle_adaptive
        self.brown_success_rate_pref = brown_success_rate_pref

        ## Multi seed agent params
        self.brown_seed_agent_period = int(brown_seed_agent_period)

        # Create linear regression object for variance prediction
        self.regr = linear_model.LinearRegression()
        self.success_rates = []
        self.prev_variances = []

        # ----------------------
        self.obs_indx = obs_indx

        self.rew_best_bias = 0
        self.agent_names = ['hide', 'seek']
        print('BatchPolopt: env.action_space.high = ', self.env.action_space.high)
        self.Fxy_max = np.linalg.norm(self.env.action_space.high[2:4], ord=2)
        self.digit_distr_uniform = np.array([1./9.]*9)
        self.entropy_max = self.entropy(self.digit_distr_uniform)

        if self.second_order_update:
            assert self.kl_batch_size == 1
            assert self.n_itr_update == 1

        # Params to keep track of moving average (both intrinsic and external reward) mean/var.
        if self.normalize_reward:
            self._reward_mean = deque(maxlen=self.kl_q_len)
            self._reward_std = deque(maxlen=self.kl_q_len)
        if self.use_kl_ratio:
            self._kl_mean = deque(maxlen=self.kl_q_len)
            self._kl_std = deque(maxlen=self.kl_q_len)

        if self.use_kl_ratio_q:
            # Add Queue here to keep track of N last kl values, compute average
            # over them and divide current kl values by it. This counters the
            # exploding kl value problem.
            self.kl_previous = deque(maxlen=self.kl_q_len)

        pf.print_sec0('POLICY OPTIMIZATION PARAMETERS')
        for key,val in locals().items(): print(key, ': ', val)
        pf.print_sec0_end()

        if self.use_hide_alg == 0:
            self.myplotter = myPlotter(out_dir= self.log_dir + 'graph_log')
        else:
            self.myplotter = myPlotter(out_dir=self.log_dir + 'graph_log', graph_names=['xy_time', 'xy_time_test', 'xy_tasklabels'])

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policies, self.bnn)
        if self.plot:
            plotter.init_plot(self.env, self.policies)

    def scale01(self, x, min_max):
        a = min_max[0]
        b = min_max[1]
        x = np.clip(x, a, b)
        bias = (a + b) / 2
        scale = 1.0 / (b - a)
        return (x - bias) * scale + 0.5

    def shutdown_worker(self):
        pass

    def train(self):
        ###############################################################################
        ## Dynamics related initialization
        batch_size = 1  # Redundant
        n_batches = 5  # Hardcode or annealing scheme \pi_i.

        # MDP observation and action dimensions.
        act_dim = np.prod(self.env.action_space.shape)

        if self.rew_bnn_use:
            logger.log("Building BNN model (eta={}) ...".format(self.eta))
            start_time = time.time()
            ##!!! Requires work
            # Tuple observations should be incorporated
            bnn_obs_shape = self.preproc_obs(self.env.observation_space.low).shape
            obs_dim = np.prod(bnn_obs_shape)
            self.bnn = bnn.BNN(
                obs_shape=bnn_obs_shape,
                n_act=act_dim,
                n_out=obs_dim,
                net_params=self.bnn_params,
                n_batches=n_batches,
                batch_size=batch_size,
                n_samples=self.snn_n_samples,
                prior_sd=self.prior_sd,
                use_reverse_kl_reg=self.use_reverse_kl_reg,
                reverse_kl_reg_factor=self.reverse_kl_reg_factor,
                second_order_update=self.second_order_update,
                learning_rate=self.unn_learning_rate,
                compression=self.compression,
                information_gain=self.information_gain
            )
            logger.log(
            "Model built ({:.1f} sec).".format((time.time() - start_time)))
        else:
            self.bnn = None


        ## Pool is only needed for dynamics, so it is not really necessary to initialize it
        if self.rew_bnn_use and self.use_replay_pool:
            self.pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_shape=self.env.observation_space.shape,
                action_dim=act_dim,
                observation_dtype=self.env.observation_space.low.dtype
            )
        else:
            self.pool = None

        ## Rendering initial policy
        # self.show_rollouts(0)

        ###############################################################################
        ## Initialize the parallel sampler
        self.start_worker()

        ## Initilizing optimization
        self.init_opt(policy_name='seek')

        self.episode_rewards = []
        self.episode_lengths = []
        for itr in range(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)

            ## Obtaining samples with parallel workers
            paths = self.obtain_samples(itr)

            # Re-organizing dimensions:
            # paths[i][hide/seek] into pahts[hide/seek][i]
            paths = self.ld2dl(paths)

            ## Sample processing:
            # - calculating additional rewards
            # - baseline fitting
            # - logging everything
            logger.record_tabular('Iteration', itr)

            seek_samples_data = self.process_samples(itr, paths['seek'],
                                                baseline=self.baselines['seek'],
                                                policy=self.policies['seek'],
                                                name='seek')
            samples_data = {'seek':seek_samples_data}


            ## Fitting the baseline
            logger.log("Fitting baseline seek...")
            self.baselines['seek'].fit(paths['seek'])
            logger.log("Fitted")

            ## Fitting the dynamics
            if self.rew_bnn_use:
                self.train_dynamics(samples_data=samples_data['seek'])

            ## Logging the hell out of it
            # self.env.log_diagnostics(paths)
            self.policies['seek'].log_diagnostics(paths['seek'])
            self.baselines['seek'].log_diagnostics(paths['seek'])

            ## Optimizing policies
            self.optimize_policy(itr, samples_data['seek'], policy_name='seek')

            ## Saving the hell out of it
            logger.log("saving snapshot...")

            # Just constructs a dictionary with all agent-related objects you care to save
            params = self.get_itr_snapshot(itr, samples_data)

            # Compose diagnostics
            diagnostics = {}
            for agent_name in ['seek']:
                diagnostics[agent_name] = {}

                paths = samples_data[agent_name]["paths"]
                if self.store_paths:
                    diagnostics[agent_name]["paths"] = paths

                self.episode_rewards.extend(sum(p["rewards"]) for p in paths)
                self.episode_lengths.extend(len(p["rewards"]) for p in paths)

                # diagnostics[agent_name]["episode_rewards"] = np.array(self.episode_rewards)
                # diagnostics[agent_name]["episode_lengths"] = np.array(self.episode_lengths)
                diagnostics[agent_name]["algo"] = self

            params['diagnostics'] = diagnostics
            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            ## Rendering few rollouts to show results and record video at the end of every few iterations
            self.show_rollouts(itr)

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot()
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                              "continue...")

        ## Cleaning up
        self.shutdown_worker()

    def prepare_dict2save(self, diagnostics=None, paths=None, obs_indx_exclude=None):
        """
        Corrections to save:
        - savemat does not digest tuples, thus observations should be transformed
        :param diagnostics: (dict) dictionary with diagnostics
        :return:
        """
        if diagnostics is not None:
            for agent in diagnostics.keys():
                if isinstance(diagnostics[agent]['paths'][0]['observations'], tuple):
                    for path_i, path in enumerate(diagnostics[agent]['paths']):
                        for i, v in enumerate(diagnostics[agent]['paths'][path_i]['observations']):
                            # print('obs_indx = ', i, ' excluded = ', obs_indx_exclude)
                            if i != obs_indx_exclude:
                                diagnostics[agent]['paths'][path_i]['observations_' + str(i)] = v
                        del diagnostics[agent]['paths'][path_i]['observations']

        if paths is not None:
            if isinstance(paths[0]['observations'], tuple):
                for path_i, path in enumerate(paths):
                    for i, v in enumerate(paths[path_i]['observations']):
                        if i != obs_indx_exclude:
                            paths[path_i]['observations_' + str(i)] = v
                            paths[path_i]['observations_' + str(i)] = v
                    del paths[path_i]['observations']

    def train_hide_seek(self):
        ###############################################################################
        ## Dynamics related initialization
        batch_size = 1  # Redundant
        n_batches = 5  # Hardcode or annealing scheme \pi_i.

        # MDP observation and action dimensions.
        act_dim = np.prod(self.env.action_space.shape)

        if self.rew_bnn_use:
            logger.log("Building BNN model (eta={}) ...".format(self.eta))
            start_time = time.time()
            bnn_obs_shape = self.preproc_obs(self.env.observation_space.low).shape
            obs_dim = np.prod(bnn_obs_shape)
            self.bnn = bnn.BNN(
                obs_shape=bnn_obs_shape,
                n_act=act_dim,
                n_out=obs_dim,
                net_params=self.bnn_params,
                n_batches=n_batches,
                batch_size=batch_size,
                n_samples=self.snn_n_samples,
                prior_sd=self.prior_sd,
                use_reverse_kl_reg=self.use_reverse_kl_reg,
                reverse_kl_reg_factor=self.reverse_kl_reg_factor,
                second_order_update=self.second_order_update,
                learning_rate=self.unn_learning_rate,
                compression=self.compression,
                information_gain=self.information_gain
            )
            logger.log(
                "Model built ({:.1f} sec).".format((time.time() - start_time)))
        else:
            self.bnn = None


        ## Pool is only needed for dynamics, so it is not really necessary to initialize it
        if self.rew_bnn_use and self.use_replay_pool:
            self.pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_shape=self.env.observation_space.shape,
                action_dim=act_dim,
                observation_dtype=self.env.observation_space.low.dtype
            )
        else:
            self.pool = None
        ###############################################################################
        ## Initialize the parallel sampler
        self.start_worker()

        ## Initilizing optimization
        self.init_opt(policy_name='hide')
        self.init_opt(policy_name='seek')

        self.episode_rewards = []
        self.episode_lengths = []
        for itr in range(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)

            ## Obtaining samples with parallel workers
            paths = self.obtain_samples(itr)

            # Re-organizing dimensions:
            # paths[i][hide/seek] into pahts[hide/seek][i]
            paths = self.ld2dl(paths)

            ## Sample processing:
            # - calculating additional rewards
            # - baseline fitting
            # - logging everything
            logger.record_tabular('Iteration', itr)

            # Calculating agent specific rewards
            if self.mode == 'seek_with_digit_action':
                paths = self.hide_rewards(paths=paths)
                paths = self.seek_rewards(paths=paths)
            elif self.mode == 'seek_force_only' or self.mode == 'reach_center_and_stop':
                if self.env.spec.id[:6] == 'Blocks' and self.env.spec.id[:12] != 'BlocksSimple' :
                    paths, taskclassif_diagn = self.hide_rewards_pretrained_classifier(itr=itr, paths=paths)
                else:
                    paths, taskclassif_diagn = self.hide_rewards_taskclassif(itr=itr, paths=paths)
                paths = self.seek_rewards_pretrained_classifier(itr=itr, paths=paths)
            else:
                raise ValueError(self.__class__.__name__ + ': Wrong execution mode:', self.mode)

            hide_samples_data = self.process_samples(itr, paths['hide'],
                                                baseline=self.baselines['hide'],
                                                policy=self.policies['hide'],
                                                name='hide')
            seek_samples_data = self.process_samples(itr, paths['seek'],
                                                baseline=self.baselines['seek'],
                                                policy=self.policies['seek'],
                                                name='seek')
            samples_data = {'hide':hide_samples_data, 'seek':seek_samples_data}


            ## Fitting the baseline
            logger.log("Fitting baseline hide...")
            self.baselines['hide'].fit(paths['hide'])
            logger.log("Fitting baseline seek...")
            self.baselines['seek'].fit(paths['seek'])
            logger.log("Fitted")

            ## Fitting the dynamics
            if self.rew_bnn_use:
                self.train_dynamics(samples_data=samples_data['hide'])
                self.train_dynamics(samples_data=samples_data['seek'])

            ## Logging the hell out of it
            # self.env.log_diagnostics(paths)
            self.policies['hide'].log_diagnostics(paths['hide'])
            self.policies['seek'].log_diagnostics(paths['seek'])

            self.baselines['hide'].log_diagnostics(paths['hide'])
            self.baselines['seek'].log_diagnostics(paths['seek'])

            ## Optimizing policies
            hide_perform_optimization = (self.hide_stop_improve_after is None) or \
                                        (itr < self.hide_stop_improve_after)
            self.optimize_policy(itr, samples_data['hide'], policy_name='hide',
                                 do_optimization = hide_perform_optimization)


            log_seek_opt_vars = self.optimize_policy(itr, samples_data['seek'], policy_name='seek')
            # if itr % self.train_seek_every == 0:
            #     log_seek_opt_vars = self.optimize_policy(itr, samples_data['seek'], policy_name='seek')
            # else:
            #     for var in log_seek_opt_vars.keys():
            #         logger.record_tabular(var, log_seek_opt_vars[var])

            ## Saving the hell out of it
            logger.log("saving snapshot...")

            # Just constructs a dictionary with all agent-related objects you care to save
            params = self.get_itr_snapshot(itr, samples_data)

            # Compose diagnostics
            diagnostics = {}
            for agent_name in self.agent_names:
                diagnostics[agent_name] = {}

                paths = samples_data[agent_name]["paths"]
                if self.store_paths:
                    diagnostics[agent_name]["paths"] = paths

                self.episode_rewards.extend(sum(p["rewards"]) for p in paths)
                self.episode_lengths.extend(len(p["rewards"]) for p in paths)

                # diagnostics[agent_name]["episode_rewards"] = np.array(self.episode_rewards)
                # diagnostics[agent_name]["episode_lengths"] = np.array(self.episode_lengths)

            ## Testing environment
            logger.log('Testing environment ...')
            test_paths = self.test_rollouts_seek()
            self.seek_rewards_pretrained_classifier(itr=itr, paths=test_paths, prefix='test_', test=True)

            ## Saving everything
            # This is diagnostics of the task classifier (assigns rewards based on classif of task complexity)
            diagnostics['hide']['taskclassif'] = taskclassif_diagn
            params['diagnostics'] = diagnostics
            logger.log('Saving mat file ...')
            self.prepare_dict2save(diagnostics=diagnostics, obs_indx_exclude=0)
            self.prepare_dict2save(paths=test_paths['seek'], obs_indx_exclude=0)
            diagnostics = {'train': diagnostics, 'test': test_paths['seek']}
            sio.savemat(self.diagnostics_dir + 'diagnostics_itr%04d' % itr,
                        mdict=diagnostics,
                        do_compression=True)

            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            ## Rendering few rollouts to show results and record video at the end of every few iterations
            logger.log('Showing environment ...')
            # self.show_rollouts_hide_seek(itr)
            self.show_rollouts_seek(itr)

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot(name='seek')
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                              "continue...")

        ## Cleaning up
        self.shutdown_worker()

    def train_seek(self):
        ###############################################################################
        ## Dynamics related initialization
        batch_size = 1  # Redundant
        n_batches = 5  # Hardcode or annealing scheme \pi_i.

        # MDP observation and action dimensions.
        act_dim = np.prod(self.env.action_space.shape)

        logger.log("Building BNN model (eta={}) ...".format(self.eta))
        start_time = time.time()
        if self.rew_bnn_use:
            ##!!! Requires work
            # Tuple observations should be incorporated
            bnn_obs_shape = self.preproc_obs(self.env.observation_space.low).shape
            obs_dim = np.prod(bnn_obs_shape)
            self.bnn = bnn.BNN(
                obs_shape=bnn_obs_shape,
                n_act=act_dim,
                n_out=obs_dim,
                net_params=self.bnn_params,
                n_batches=n_batches,
                batch_size=batch_size,
                n_samples=self.snn_n_samples,
                prior_sd=self.prior_sd,
                use_reverse_kl_reg=self.use_reverse_kl_reg,
                reverse_kl_reg_factor=self.reverse_kl_reg_factor,
                second_order_update=self.second_order_update,
                learning_rate=self.unn_learning_rate,
                compression=self.compression,
                information_gain=self.information_gain
            )
        else:
            self.bnn = None
        logger.log(
            "Model built ({:.1f} sec).".format((time.time() - start_time)))

        ## Pool is only needed for dynamics, so it is not really necessary to initialize it
        if self.rew_bnn_use and self.use_replay_pool:
            self.pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_shape=self.env.observation_space.shape,
                action_dim=act_dim,
                observation_dtype=self.env.observation_space.low.dtype
            )
        else:
            self.pool = None

        ## Rendering initial policy
        self.show_rollouts_seek(0)

        ###############################################################################
        ## Initialize the parallel sampler
        self.start_worker()

        ## Initilizing optimization
        self.init_opt(policy_name='seek')

        self.episode_rewards = []
        self.episode_lengths = []
        for itr in range(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)

            ## Obtaining samples with parallel workers
            paths = self.obtain_samples(itr)

            # Re-organizing dimensions:
            # paths[i][hide/seek] into pahts[hide/seek][i]
            paths = self.ld2dl(paths)

            ## Sample processing:
            # - calculating additional rewards
            # - baseline fitting
            # - logging everything
            logger.record_tabular('Iteration', itr)

            # Calculating agent specific rewards
            if self.mode == 'seek_with_digit_action':
                paths = self.seek_rewards(paths=paths)
            elif self.mode == 'seek_force_only' or self.mode == 'reach_center_and_stop':
                paths = self.seek_rewards_pretrained_classifier(itr=itr, paths=paths)
            else:
                raise ValueError(self.__class__.__name__ + ': Wrong execution mode:', self.mode)

            seek_samples_data = self.process_samples(itr, paths['seek'],
                                                baseline=self.baselines['seek'],
                                                policy=self.policies['seek'],
                                                name='seek')
            samples_data = {'seek':seek_samples_data}


            ## Fitting the baseline
            logger.log("Fitting baseline seek...")
            self.baselines['seek'].fit(paths['seek'])
            logger.log("Fitted")

            ## Fitting the dynamics
            if self.rew_bnn_use:
                self.train_dynamics(samples_data=samples_data['seek'])

            ## Logging the hell out of it
            # self.env.log_diagnostics(paths)
            self.policies['seek'].log_diagnostics(paths['seek'])
            self.baselines['seek'].log_diagnostics(paths['seek'])

            ## Optimizing policies
            self.optimize_policy(itr, samples_data['seek'], policy_name='seek')

            ## Saving the hell out of it
            logger.log("saving snapshot...")

            # Just constructs a dictionary with all agent-related objects you care to save
            params = self.get_itr_snapshot(itr, samples_data)

            # Compose diagnostics
            diagnostics = {}
            for agent_name in ['seek']:
                diagnostics[agent_name] = {}

                paths = samples_data[agent_name]["paths"]
                if self.store_paths:
                    diagnostics[agent_name]["paths"] = paths

                self.episode_rewards.extend(sum(p["rewards"]) for p in paths)
                self.episode_lengths.extend(len(p["rewards"]) for p in paths)

                # diagnostics[agent_name]["episode_rewards"] = np.array(self.episode_rewards)
                # diagnostics[agent_name]["episode_lengths"] = np.array(self.episode_lengths)
                # diagnostics[agent_name]["algo"] = self

            ## Saving everything
            params['diagnostics'] = diagnostics
            logger.log('Saving mat file ...')
            self.prepare_dict2save(diagnostics=diagnostics, obs_indx_exclude=0)
            # self.prepare_dict2save(diagnostics=diagnostics)
            diagnostics = {'train': diagnostics}
            sio.savemat(self.diagnostics_dir + 'diagnostics_itr%04d' % itr,
                        diagnostics,
                        do_compression=True)

            params['diagnostics'] = diagnostics
            logger.log("pickling policies ...")
            logger.save_itr_params(itr, params)
            # logger.save_itr_params(itr, self.policies)
            # pickle.dump(self.policies['seek'], open('_results_temp/test/policies_seek.pkl', 'wb'))
            # joblib.dump(self.policies, '_results_temp/test/policies_jb.pkl')
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            ## Rendering few rollouts to show results and record video at the end of every few iterations
            self.show_rollouts_seek(itr)

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot()
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                              "continue...")

        ## Cleaning up
        self.shutdown_worker()

    def train_dynamics(self, samples_data):
        """
        WARNING: Function is not adapted to take observation tuples
        :param samples_data:
        :return:
        """
        if self.use_replay_pool:
            # Fill replay pool.
            logger.log("Fitting dynamics model using replay pool ...")
            for path in samples_data['paths']:
                path_len = len(path['rewards'])
                for i in range(path_len):
                    obs = path['observations'][i]
                    act = path['actions'][i]
                    rew = path['rewards'][i]
                    term = (i == path_len - 1)
                    self.pool.add_sample(obs, act, rew, term)

            # Now we train the dynamics model using the replay self.pool; only
            # if self.pool is large enough.
            if self.pool.size >= self.min_pool_size:
                obs_mean, obs_std, act_mean, act_std = self.pool.mean_obs_act()
                _inputss = []
                _targetss = []
                for _ in range(self.n_updates_per_sample):
                    batch = self.pool.random_batch(
                        self.pool_batch_size)
                    obs = (batch['observations'] - obs_mean) / \
                          (obs_std + 1e-8)
                    next_obs = (
                                   batch['next_observations'] - obs_mean) / (obs_std + 1e-8)
                    act = (batch['actions'] - act_mean) / \
                          (act_std + 1e-8)

                    # Subsampling obs for dynamics model
                    # if observations are images
                    _inputs = [self.preproc_obs(obs), act]
                    _targets = self.preproc_obs(next_obs)

                    _inputss.append(_inputs)
                    _targetss.append(_targets)

                old_acc = 0.
                for _inputs, _targets in zip(_inputss, _targetss):
                    _out = self.bnn.pred_fn(_inputs[0], _inputs[1])
                    old_acc += np.mean(np.square(_out - _targets))
                old_acc /= len(_inputss)

                for _inputs, _targets in zip(_inputss, _targetss):
                    self.bnn.train_fn(_inputs[0], _inputs[1], _targets)

                new_acc = 0.
                for _inputs, _targets in zip(_inputss, _targetss):
                    _out = self.bnn.pred_fn(_inputs[0], _inputs[1])
                    new_acc += np.mean(np.square(_out - _targets))
                new_acc /= len(_inputss)

                logger.record_tabular(
                    'BNN_DynModelSqLossBefore', old_acc)
                logger.record_tabular(
                    'BNN_DynModelSqLossAfter', new_acc)
                # ----------------

    def show_rollouts(self, iter):
        if iter % glob.video_scheduler.render_every_iterations == 0:
            glob.video_scheduler.record = True
            for i in range(0, glob.video_scheduler.render_rollouts_num):
                rollout(self.env, self.policies, animated=True)
                glob.video_scheduler.record = False

    def show_rollouts_hide_seek(self, iter):
        if iter % glob.video_scheduler.render_every_iterations == 0:
            logger.log('Showing/Recording hide/seek rollouts')
            env = self.env
            glob.video_scheduler.record = True
            if env.spec.id[:6] == 'Blocks':
                # Let's make rendering a bit smoother
                frame_skip_prev = env.env.unwrapped.frame_skip
                env.env.unwrapped.frame_skip = 10

            for i in range(0, glob.video_scheduler.render_rollouts_num):
                rollout_hide_seek(env, self.policies, animated=True, mode=self.mode, hide_tmax=self.hide_tmax)
                # time.sleep(2)
                glob.video_scheduler.record = False
            if env.spec.id[:6] == 'Blocks':
                env.env.unwrapped.frame_skip = frame_skip_prev

    def test_rollouts_seek(self, animated_roolouts_num=0):
        if self.env_test is not None:
            logger.log('Special test env is used ...')
            env = self.env_test
        else:
            env = self.env
        paths = []

        ep_len_lst = []
        for i in range(0, self.test_episodes_num):
            path = rollout_seek(env, self.policies, animated=(i < animated_roolouts_num), mode=None, always_return_paths=True)
            ep_len_lst.append(path['seek']['rewards'].size)
            paths.append(path)
        logger.log('Test episodes done = %d len_avg = %d' % (self.test_episodes_num, np.mean(ep_len_lst)))
        paths = self.ld2dl(paths)
        return paths

    def uniform_rollouts(self, samples_num, animated_roolouts_num=0):
        if self.env_test is not None:
            logger.log('Using special test env for uniform sampler ...')
            env = self.env_test
        else:
            env = self.env
        paths = []

        ep_len_lst = []
        samples_cur = 0
        i = 0
        while samples_cur < samples_num:
            path = rollout_seek(env, self.policies, animated=(i < animated_roolouts_num), mode=None, always_return_paths=True)
            ep_len_lst.append(path['seek']['rewards'].size)
            paths.append(path)
            samples_cur += path['seek']['rewards'].size
            i += 1
        logger.log('Test episodes done = %d len_avg = %d samples=%d' % (self.test_episodes_num, np.mean(ep_len_lst), samples_cur))
        return paths

    def test_init_states(self):
        env = self.env
        paths = []
        animated_roolouts_num = 0
        ep_len_lst = []
        for i in range(0, self.test_init_states_episodes_num):
            path = rollout_seek(env, self.policies, animated=(i < animated_roolouts_num), mode=None, always_return_paths=True)
            ep_len_lst.append(path['seek']['rewards'].size)
            paths.append(path)
        logger.log('Test episodes done for init states = %d len_avg = %d' % (self.test_episodes_num, np.mean(ep_len_lst)))
        paths = self.ld2dl(paths)
        return paths

    def show_rollouts_seek(self, iter):
        # print('BatchPolOpt: render_every_iterations = ', glob.video_scheduler.render_every_iterations, 'iter:', iter)
        if iter % glob.video_scheduler.render_every_iterations == 0:
            glob.video_scheduler.record = True
            env = self.env_test
            if env.spec.id[:6] == 'Blocks' and env.spec.id[:12] != 'BlocksSimple':
                # Let's make rendering a bit smoother
                logger.log('WARNING: Frame skip is changed since Blocks env is used !!!')
                frame_skip_prev = env.env.unwrapped.frame_skip
                env.env.unwrapped.frame_skip = 10
            if self.use_hide is None:
                for i in range(0, glob.video_scheduler.render_rollouts_num):
                    rollout(env, self.policies, animated=True)
            else:
                for i in range(0, glob.video_scheduler.render_rollouts_num):
                    rollout_seek(env, self.policies, animated=True)
            glob.video_scheduler.record = False
            if env.spec.id[:6] == 'Blocks' and env.spec.id[:12] != 'BlocksSimple':
                env.env.unwrapped.frame_skip = frame_skip_prev
            logger.log('Finished rendering rollouts')

    def obs_is_img(self, obs):
        obs_shape = obs.shape
        self._obs_is_img = len(obs_shape) >= 2 and ((obs_shape[0] > 1 and obs_shape[1] > 1) or (len(obs_shape) > 2))
        return self._obs_is_img

    def preproc_obs(self, obs):
        """
        Preprocessign observations for BNN
        :param obs:
        :return:
        """
        if self.obs_is_img(obs):
            if len(obs.shape) == 2:
                return obs[::self.subsamp_bnn_obs_step, ::self.subsamp_bnn_obs_step]
            else:
                return obs[:, ::self.subsamp_bnn_obs_step, ::self.subsamp_bnn_obs_step]
        else:
            return obs

    def init_opt(self):
        """
        Initialize the optimization procedure. If using theano / cgt, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self, name):
        if self.plot:
            plotter.update_plot(self.policies[name], self.max_path_length)

    def sigmoid(self, x, offset=-0.75, scale=5):
        return 1. / (1. + np.exp(-scale*(x-offset)))

    def obtain_samples(self, itr):
        # 현재 policy의 weight값을 불러온다.
        cur_params = {}
        for key in self.policies.keys():
            cur_params[key] = self.policies[key].get_param_values()

        # self.rew_bnn_use = False 일때 초기화
        cur_dynamics_params = None
        reward_mean = None
        reward_std = None
        obs_mean, obs_std, act_mean, act_std = None, None, None, None

        # if self.rew_bnn_use:
        #     cur_dynamics_params = self.bnn.get_param_values()
        #
        # if self.rew_bnn_use and self.normalize_reward:
        #     # Compute running mean/std.
        #     reward_mean = np.mean(np.asarray(self._reward_mean))
        #     reward_std = np.mean(np.asarray(self._reward_std))
        #
        # # Mean/std obs/act based on replay pool.
        # if self.rew_bnn_use:
        #     obs_mean, obs_std, act_mean, act_std = self.pool.mean_obs_act()


        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            dynamics_params=cur_dynamics_params,
            max_samples=self.batch_size,
            max_path_length=self.max_path_length,
            itr=itr,
            normalize_reward=self.normalize_reward,
            reward_mean=reward_mean,
            reward_std=reward_std,
            kl_batch_size=self.kl_batch_size,
            n_itr_update=self.n_itr_update,
            use_replay_pool=self.use_replay_pool,
            obs_mean=obs_mean,
            obs_std=obs_std,
            act_mean=act_mean,
            act_std=act_std,
            second_order_update=self.second_order_update,
            use_hide=self.use_hide,
            use_hide_alg = self.use_hide_alg,
            mode=self.mode,
            show_rollout_chance=self.show_rollout_chance,
            hide_tmax=self.hide_tmax,
        )

        if self.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(
                paths, self.batch_size)
            return paths_truncated

    def ld2dl(self, LD):
        """
        The function converts a list of dictionaries into dictionary of lists
        Necessary to re-organize dimensions, such as paths[i][hide/seek] into paths[hide/seek][i]
        :param LD:
        :return:
        """
        return dict(zip(LD[0], zip(*[d.values() for d in LD])))

    def dl2ld(self, DL):
        """
        The function converts a dictionary of lists into list of dictionaries
        :param DL:
        :return:
        """
        return [dict(zip(DL, t)) for t in zip(*DL.values())]

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def softmax_batch(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def get_digit_distr(self, action):
        if len(action.shape) == 1:
            return np.squeeze(self.softmax(action[4:13]))
        else:
            return self.softmax_batch(action[:, 4:13])

    def hide_rewards(self, paths):
        """
        Calculating additional rewards for hide agent.
        Since rewards of hide agent depends mainly on performance of seek agent
        Seek's information will mainly be used
        :param paths:
        :return:
        """
        ##
        # self.rew_hide__search_time_coeff = 1.
        # self.rew_action_coeff = -1.
        # self.rew_hide__digit_entropy_coeff = 1.
        # self.rew_hide__digit_correct_coeff = 1. #make <0 if we want to penalize correct predicitons by seek
        # self.rew_hide__time_step = -0.01 # Just penalty for taking time steps

        rew_seek_time = []
        rew_action = []
        rew_digit_entropy = []
        rew_digit_correct = []

        path_lengths = []

        for i, path in enumerate(paths['seek']):
            path_lengths.append(paths['hide'][i]['rewards'].size)

            #######################################################################
            ## Time reward (reflects complexity)
            # Options:
            # - reward length seek takes to figure out
            # - reward length difference
            time_len = len(path['rewards'])
            paths['hide'][i]['rew_seek_time'] = self.rew_hide__search_time_coeff * time_len
            paths['hide'][i]['rewards'][-1] += paths['hide'][i]['rew_seek_time']
            rew_seek_time.append(paths['hide'][i]['rew_seek_time'])
            # print('hide_rewards: rew_seek_time ', paths['hide'][i]['rew_seek_time'])

            #######################################################################
            ## Penalty for time
            paths['hide'][i]['rewards'] += self.rew_hide__time_step

            #######################################################################
            ## Penalty for taking actions
            # - penalize norm of Fx,Fy vector of actions

            # action = [x,y,Fx,Fy,stop], in np slicing 2:4 the last indx is not included
            Fxy = paths['hide'][i]['actions'][:, 2:4]
            force_ratio = np.linalg.norm(Fxy, ord=2, axis=1) / self.Fxy_max

            # Squared to penalize large actions more severely
            # Be sure to make rew_action_coeff negative
            path['rew_action'] = self.rew_hide__action_coeff * force_ratio**2
            path['rewards'] += paths['hide'][i]['rew_action']
            rew_action.append(np.mean(paths['hide'][i]['rew_action']))
            # print('hide_rewards: rew_action ', paths['hide'][i]['rew_action'])

            #######################################################################
            ## Reward Seek's ambiguity of digit answer
            # Let's reward high entropy of seek's guesses
            # - when he is unsure about wrong guess - it has positive effect, since it makes it think
            # - when it unsure about good guess - it has positive effect of complicating the task
            digit_distrib = self.get_digit_distr(path['actions'][-1, :])
            digit_guess_entropy = self.entropy(digit_distrib)
            paths['hide'][i]['rew_digit_entropy'] = self.rew_hide__digit_entropy_coeff * digit_guess_entropy / self.entropy_max
            paths['hide'][i]['rewards'][-1] += paths['hide'][i]['rew_digit_entropy']
            rew_digit_entropy.append(paths['hide'][i]['rew_digit_entropy'])
            # print('hide_rewards: rew_digit_entropy ', paths['hide'][i]['rew_digit_entropy'])

            #######################################################################
            ## Reward/penalize Seek's right answer (use coeff to set if it is reward or panalty)
            digit_guess = np.argmax(digit_distrib)
            paths['hide'][i]['rew_digit_correct'] = self.rew_hide__digit_correct_coeff * \
                                                    float(digit_guess == path['env_infos']['digits_in_scene'][-1])
            paths['hide'][i]['rewards'][-1] += paths['hide'][i]['rew_digit_correct']
            rew_digit_correct.append(paths['hide'][i]['rew_digit_correct'])
            # print('hide_rewards: rew_digit_correct ', paths['hide'][i]['rew_digit_correct'])

            #######################################################################
            ## Create original rewards
            paths['hide'][i]['rewards_orig'] = copy.deepcopy(paths['hide'][i]['rewards'])

        logger.record_tabular('hide_ep_len', np.mean(path_lengths))
        logger.record_tabular('hide_ep_len_max', np.max(path_lengths))
        logger.record_tabular('hide_ep_len_min', np.min(path_lengths))
        logger.record_tabular('hide_rew_seek_time', np.mean(rew_seek_time))
        logger.record_tabular('hide_rew_action', np.mean(rew_action))
        logger.record_tabular('hide_rew_digit_entropy', np.mean(rew_digit_entropy))
        logger.record_tabular('hide_rew_digit_correct', np.mean(rew_digit_correct))

        return paths

    @staticmethod
    def get_selfplay_rew(t_hide, t_seek):
        return max(0, t_seek - t_hide)

    @staticmethod
    def get_timelen_reward(t, t_avg, t_max, power=3):
        t = min(t_max, t)
        if t > t_avg:
            return (t_max - t) ** power / (t_max - t_avg) ** power
        else:
            return (t / t_avg) ** power

    @staticmethod
    def get_timelen_reward2(t, t_avg, t_max, power=3):
        t = min(t_max, t)
        if t == t_max:
            return 0
        else:
            return ((t - t_avg + 1) / (t_max - t_avg)) ** power

    @staticmethod
    def get_timelen_reward_with_penalty(t, t_avg, t_max, power=3):
        if t >= t_max or t <= 1:
            return -0.1
        t = min(t_max, t)
        if t > t_avg:
            return (t_max - t) ** power / (t_max - t_avg) ** power
        else:
            return (t / t_avg) ** power

    @staticmethod
    def get_timelen_reward_with_median(t, t_avg, t_max, power=1):
        if t == t_max:
            return 0
        t = min(t_max, t)
        rew = np.sign(t - t_avg) * np.abs(t - t_avg) ** power
        if t > t_avg:
            norm = np.abs((t_max - t_avg) ** power)
        else:
            norm = np.abs((t_avg) ** power)
        return rew / norm

    @staticmethod
    def get_prob_reward(prob, pow=3., middle=0.5):
        middle = min(middle, 1.0)
        middle = max(middle, 0.0)
        norm = np.ones_like(prob)
        norm[prob < middle] = middle
        norm[prob >= middle] = 1.0 - middle
        return -np.abs(((prob - middle) / norm) ** pow)

    @staticmethod
    def get_prob_reward_unnorm(prob, pow=1., middle=0.5):
        middle = min(middle, 1.0)
        middle = max(middle, 0.0)
        norm = np.ones_like(prob)
        norm[prob < middle] = 1.0
        norm[prob >= middle] = -1.0
        return -np.abs(((prob - middle) / norm) ** pow)

    def hide_rewards_pretrained_classifier(self, itr, paths, test=False):
        """
        Calculating additional rewards for hide agent based on performance of seek agent.
        :param paths:
        :return:
        """
        ##
        # self.rew_seek__digit_entropy_coeff = 1.
        # self.rew_seek__digit_correct_coeff = 1.
        # self.rew_seek__time_step = -0.01  # Just penalty for taking time steps

        # true_digit
        # act_min_dist
        # act_min_dist_norm
        # act_force
        # act_force_norm
        # act_force_max
        # act_dist_max
        # rew_mnist
        # pred_digit
        # pred_distr
        # pred_entropy

        # Per episode reward lists
        path_lengths = []
        seek_path_lengths = []

        rew_orig_rewards_eplst = []

        rew_seek_time_eplst = []
        rew_seek_act_force_eplst = []

        rew_mnist_eplst = []
        rew_action_force_eplst = []
        rew_action_dist_eplst= []
        rew_center_reached_eplst = []

        action_force_eplst = []
        action_dist_eplst = []

        x_init_eplst = []
        y_init_eplst = []

        traj_num = 0
        self.timerew_percentile = 75
        self.timelen_success_eplst = [0]

        for i, path in enumerate(paths['hide']):
            traj_num += 1
            seek_time_len = paths['seek'][i]['rewards'].size
            self.timelen_eplst.append(seek_time_len)
            if seek_time_len < self.timelen_max:
                self.timelen_success_eplst.append(seek_time_len)

        ## Calculating success ratio
        traj_success_ratio = float(len(self.timelen_success_eplst)) / float(traj_num)


        ## Calculating average episode timelen for timelen-based rewards
        timelen_hist_size = np.max([self.timelen_avg_hist_size, traj_num])
        self.timelen_eplst = self.timelen_eplst[-timelen_hist_size:]
        if self.adaptive_timelen_avg:
            if self.adaptive_percentile:
                traj_success_ratio_scaled = self.scale01(traj_success_ratio, self.adaptive_percentile_regulation_zone)
                self.timerew_percentile = np.clip(int(traj_success_ratio_scaled * 100), 1, 99)
                self.timelen_avg = np.percentile(self.timelen_success_eplst, self.timerew_percentile)
            else:
                # This mode is used for the previous scripts
                self.timelen_avg = np.percentile(self.timelen_eplst, self.timerew_percentile)

        logger.log('Traj_success_ratio: %d, Success timelen lst: [%s], Percentile: %d, Timelen: %d' % (
        traj_success_ratio, " ".join(str(x) for x in self.timelen_success_eplst), self.timerew_percentile, self.timelen_avg))

        self.taskclassif_labels_all = []
        self.taskclassif_obs_success = []
        self.taskclassif_obs_fail = []

        success_timelen = []

        for i, path in enumerate(paths['hide']):

            path_lengths.append(path['rewards'].size)
            paths['hide'][i]['reward_components'] = {}

            rew_orig_rewards_eplst.append(np.sum(path['rewards']))

            #######################################################################
            ## REWARDS
            #######################################################################
            ## Nullifying the main reward since hide should not depend on them
            paths['hide'][i]['rewards_orig'] = copy.deepcopy(paths['hide'][i]['rewards'])
            paths['hide'][i]['rewards'] = np.zeros_like(paths['hide'][i]['rewards'])

            ## Reward/penalize Seek's right answer (use coeff to set if it is reward or panalty)
            paths['hide'][i]['reward_components']['rew_seek_mnist_digit'] = \
                self.rew_hide__digit_correct_coeff * (2 * float(paths['seek'][i]['env_infos']['digit_revealed'][-1]) - 1.0)

            paths['hide'][i]['rewards'][-1] += paths['hide'][i]['reward_components']['rew_seek_mnist_digit']
            rew_mnist_eplst.append(paths['hide'][i]['reward_components']['rew_seek_mnist_digit'])

            ## Reward for Seek reaching the center
            paths['hide'][i]['reward_components']['rew_center_reached'] = \
                self.rew_hide__center_reached_coeff * (2 * float(paths['seek'][i]['env_infos']['center_reached'][-1]) - 1.0)

            paths['hide'][i]['rewards'][-1] += paths['hide'][i]['reward_components']['rew_center_reached']
            rew_center_reached_eplst.append(paths['hide'][i]['reward_components']['rew_center_reached'])

            ## Reward Seek taking time
            hide_time_len = paths['hide'][i]['rewards'].size
            seek_time_len = paths['seek'][i]['rewards'].size
            seek_path_lengths.append(seek_time_len)

            # These rewards were used for table environemnts
            if self.timelen_reward_fun == 'get_timelen_reward2':
                pf.print_warn('Timelen2 reward function is used')
                paths['hide'][i]['reward_components']['rew_seek_time'] = \
                    self.rew_hide__search_time_coeff * self.get_timelen_reward2(t=seek_time_len,
                                                                                t_avg=self.timelen_avg,
                                                                                t_max=self.timelen_max,
                                                                                power=self.rew_hide__search_time_power)
            elif self.timelen_reward_fun == 'get_timelen_reward_with_penalty':
                paths['hide'][i]['reward_components']['rew_seek_time'] = \
                    self.rew_hide__search_time_coeff * self.get_timelen_reward_with_penalty(t=seek_time_len,
                                                                                t_avg=self.timelen_avg,
                                                                                t_max=self.timelen_max,
                                                                                power=self.rew_hide__search_time_power)
            elif self.timelen_reward_fun == 'get_timelen_reward_with_median':
                # pf.print_warn('get_timelen_reward_with_median reward function is used')
                paths['hide'][i]['reward_components']['rew_seek_time'] = \
                    self.rew_hide__search_time_coeff * self.get_timelen_reward_with_median(t=seek_time_len,
                                                                                           t_avg=self.timelen_avg,
                                                                                           t_max=self.timelen_max,
                                                                                           power=self.rew_hide__search_time_power)
            elif self.timelen_reward_fun == 'get_selfplay_rew':
                paths['hide'][i]['reward_components']['rew_seek_time'] = \
                    self.rew_hide__search_time_coeff * self.get_selfplay_rew(t_hide=hide_time_len,
                                                                             t_seek=seek_time_len)

            else:
                raise ValueError('Unknown time rewarding functioin')

            # These rewards were used for maze environments


            # paths['hide'][i]['reward_components']['rew_seek_time'] = self.rew_hide__search_time_coeff * seek_time_len
            paths['hide'][i]['rewards'][-1] += paths['hide'][i]['reward_components']['rew_seek_time']
            rew_seek_time_eplst.append(paths['hide'][i]['reward_components']['rew_seek_time'])

            ## Reward Seek applying actions
            seek_forces = paths['seek'][i]['env_infos']['act_force_norm']
            seek_force_sum = np.sum(seek_forces)
            paths['hide'][i]['reward_components']['rew_seek_act_force'] = self.rew_hide__search_force_coeff * seek_force_sum
            paths['hide'][i]['rewards'][-1] += paths['hide'][i]['reward_components']['rew_seek_act_force']
            rew_seek_act_force_eplst.append(paths['hide'][i]['reward_components']['rew_seek_act_force'])
            action_force_eplst.append(np.mean(paths['hide'][i]['env_infos']['act_force']))

            #######################################################################
            ## PENALTIES
            #######################################################################
            ## Penalty for applying action far from center of the box
            paths['hide'][i]['reward_components']['rew_act_dist'] = \
                self.rew_hide__act_dist_coeff * paths['hide'][i]['env_infos']['act_min_dist_norm']

            paths['hide'][i]['rewards'] += paths['hide'][i]['reward_components']['rew_act_dist']
            self.check_vec_size(paths['hide'][i]['rewards'], paths['hide'][i]['reward_components']['rew_act_dist'], 'rew_act_dist')

            rew_action_dist_eplst.append(np.sum(paths['hide'][i]['reward_components']['rew_act_dist']))
            action_dist_eplst.append(np.mean(paths['hide'][i]['env_infos']['act_min_dist']))

            #######################################################################
            ## Penalty for Applying Force (taking action)
            force_ratio = paths['hide'][i]['env_infos']['act_force_norm']
            paths['hide'][i]['reward_components']['rew_act_force'] = self.rew_hide__action_coeff * force_ratio ** 2

            paths['hide'][i]['rewards'] += paths['hide'][i]['reward_components']['rew_act_force']
            self.check_vec_size(paths['hide'][i]['rewards'], paths['hide'][i]['reward_components']['rew_act_force'], 'rew_act_force')

            rew_action_force_eplst.append(np.sum(paths['hide'][i]['reward_components']['rew_act_force']))

            #######################################################################
            ## Penalty for time
            paths['hide'][i]['rewards'] += self.rew_hide__time_step

            #######################################################################
            ## Diagnostics for the episode
            x_init_eplst.append(paths['seek'][i]['observations'][self.obs_indx][0][0])
            y_init_eplst.append(paths['seek'][i]['observations'][self.obs_indx][0][1])

            self.taskclassif_obs.append(paths['seek'][i]['observations'][self.obs_indx][0][0:2])

            # Labels assign according to if task is solvable or not.
            # Typically tasks consistently solved within a time budget are solvable
            self.taskclassif_labels.append(int(seek_time_len < (self.timelen_max - 1)))
            # print('seek time len = ', seek_time_len, ' Label = ', self.taskclassif_labels[-1], ' Timelenmax = ', self.timelen_max)
            # print('All observation = ', paths['seek'][i]['observations'][self.obs_indx][:,0:2], 'obs_num', obs_num)

            ## Creating list of all observations
            if self.taskclassif_labels[-1] == 0:
                self.taskclassif_obs_fail.append(paths['seek'][i]['observations'][self.obs_indx][1:,0:2])
                self.taskclassif_obs_fail_prev.append(paths['seek'][i]['observations'][self.obs_indx][1:,0:2])
            else:
                self.taskclassif_obs_success.append(paths['seek'][i]['observations'][self.obs_indx][1:,0:2])
                self.taskclassif_obs_success_prev.append(paths['seek'][i]['observations'][self.obs_indx][1:, 0:2])
                success_timelen.append(seek_time_len)


        ## Adaptive middle point if required based on proportion of successes/failures
        print('Seek path lengths: ', seek_path_lengths)
        obs_success_num = len(self.taskclassif_obs_success)
        obs_fail_num = len(self.taskclassif_obs_fail)
        obs_fail_ratio = float(obs_fail_num) / float(obs_success_num + obs_fail_num)
        if self.taskclassif_adaptive_middle:
            self.rew_hide__taskclassif_middle = self.scale01(obs_fail_ratio,
                                                             self.taskclassif_adaptive_middle_regulation_zone)
            logger.log('Adaptive taskclassif middle: %.4f' % self.rew_hide__taskclassif_middle)


        ## After we iterated through everything we should join everything
        # Variables with _prev are not reset every iteration
        self.taskclassif_obs_fail_prev = self.taskclassif_obs_fail_prev[-self.taskclassif_obs_fail_success_hist_size:]
        self.taskclassif_obs_success_prev = self.taskclassif_obs_success_prev[-self.taskclassif_obs_fail_success_hist_size:]

        if len(self.taskclassif_obs_fail) == 0:
            logger.log('Prev failures used for task classification due to no failures in the current run')
            pf.print_warn('NO FAILURE SAMPLES WERE FOUND: ADDING SOME PREV FAILURES TO RUN CLASSIFIER')
            print('Timelenmax = ', self.timelen_max)
            print('Failure history size = ', self.taskclassif_obs_fail_success_hist_size)
            # self.taskclassif_obs_fail.append(self.taskclassif_obs_success[0])
            self.taskclassif_obs_fail = copy.deepcopy(self.taskclassif_obs_fail_prev)
            # print('Added obs to failures:', self.taskclassif_obs_fail)

        self.taskclassif_obs_fail = np.concatenate(self.taskclassif_obs_fail, axis=0)

        # Adding obviously successful observation, when hide is at the beginning
        # it should help us at the beginning when we can not generate positive samples
        self.taskclassif_obs_success.append(np.expand_dims(paths['hide'][0]['observations'][self.obs_indx][0, 0:2], axis=0))
        self.taskclassif_obs_success = np.concatenate(self.taskclassif_obs_success, axis=0)

        obs_success_num = self.taskclassif_obs_success.shape[0]
        obs_fail_num = self.taskclassif_obs_fail.shape[0]
        logger.log('Successful obs = %d Failure obs = %d' % (obs_success_num, obs_fail_num))


        ## Rewards based on classification of tasks (hard/easy)
        taskclassif_diagn = {}

        if self.rew_hide__taskclassif_coeff is not None and self.rew_hide__taskclassif_coeff != 0:

            if self.taskclassif_use_allpoints:
                '''
                This algorithm samples from all observations from the last iteration, 
                but does not touch previous iterations and does not use outdated points
                '''
                self.taskclassif_obs = self.taskclassif_obs[-traj_num:]
                self.taskclassif_labels = self.taskclassif_labels[-traj_num:]

                sample_obs_add = self.taskclassif_pool_size - traj_num
                if sample_obs_add > 0:
                    '''
                    Here we add remaining samples and balance success/failure samples
                    '''
                    samples_total = obs_success_num + obs_fail_num
                    #Guarantee that we have enough observations to add
                    sample_obs_add = min(sample_obs_add, samples_total)

                    #Adding positive samples as much as we can
                    if self.taskclassif_balance_positive_labels:
                        success_obs_add_num = int(sample_obs_add / 2.0)
                    else:
                        success_obs_add_num = int(sample_obs_add * (1.0 - obs_fail_ratio))

                    success_obs_add_num = min(success_obs_add_num, obs_success_num)
                    logger.log('Successful samples will be added to pile: %d' % success_obs_add_num)
                    taskclassif_obs_train = []
                    taskclassif_labels_train = []

                    success_obs_sampled, success_indices = sample_multidim(array=self.taskclassif_obs_success,
                                                          samp_num=success_obs_add_num)


                    ## Adding goals as samples that are sure successfull
                    if self.taskclassif_add_goal_as_pos_sampl_num > 1:
                        goal_samples = np.tile(np.expand_dims(paths['hide'][0]['observations'][self.obs_indx][0, 0:2], axis=0),
                                               [self.taskclassif_add_goal_as_pos_sampl_num - 1, 1])
                        success_obs_sampled = np.concatenate([success_obs_sampled, goal_samples], axis=0)

                    success_obs_add_num = success_obs_sampled.shape[0]
                    taskclassif_labels_train.extend([1]*success_obs_add_num)

                    #Adding failure samples
                    sample_obs_add_remain = sample_obs_add - success_obs_add_num
                    fail_obs_add_num = min(sample_obs_add_remain, obs_fail_num)
                    fail_obs_sampled, fail_indices = sample_multidim(array=self.taskclassif_obs_fail,
                                                       samp_num=fail_obs_add_num)
                    fail_obs_add_num = fail_obs_sampled.shape[0]
                    logger.log('Fail samples will be added to pile: %d' % fail_obs_add_num)
                    taskclassif_labels_train.extend([0] * fail_obs_add_num)

                    #Combining all samples
                    taskclassif_obs_train = np.concatenate([success_obs_sampled, fail_obs_sampled], axis=0)
                    logger.log('Samples used for task calssif success/fail: %d / %d' % (success_obs_add_num, fail_obs_add_num))

                    #Adding initial position samples
                    # WARING: Looks like it was a bug
                    # taskclassif_obs_train = np.concatenate([self.taskclassif_obs, taskclassif_obs_train])
                    taskclassif_obs_train = np.concatenate([taskclassif_obs_train, self.taskclassif_obs])
                    taskclassif_labels_train.extend(self.taskclassif_labels)
                else:
                    taskclassif_obs_train = self.taskclassif_obs
                    taskclassif_labels_train = self.taskclassif_labels

            else:
                '''
                This algorithm only considers initial observations, 
                thus if you need more observations than the last iteration can provide (from initial observations)
                then it takes observations from past iterations
                '''
                taskclassif_size = np.max([self.taskclassif_pool_size, traj_num])
                self.taskclassif_labels = self.taskclassif_labels[-taskclassif_size:]
                self.taskclassif_obs = self.taskclassif_obs[-taskclassif_size:]

                taskclassif_obs_train = self.taskclassif_obs
                taskclassif_labels_train = self.taskclassif_labels

            # taskclassif_labels_train.append(1)
            # taskclassif_obs_train = np.concatenate([taskclassif_obs_train, paths['hide'][0]['observations'][self.obs_indx][0,0:2]],
            #                                        axis=0)

            # Printing what we are fitting
            taskclassif_obs_train = np.array(taskclassif_obs_train)
            print('Fitted observations shape = ', taskclassif_obs_train.shape)
            logger.log('Samples (obs / lbl) to fit: %d / %d' % (taskclassif_obs_train.shape[0], len(taskclassif_labels_train)))
            # print('Seek Path lengths = ', seek_path_lengths)
            # print('Classes assigned = ', self.taskclassif_labels)

            # Counting success/fail labels
            taskclassif_labels_train_array = np.array(taskclassif_labels_train)
            success_labels_train_num = np.count_nonzero(taskclassif_labels_train_array == 1)
            fail_labels_train_num = np.count_nonzero(taskclassif_labels_train_array == 0)

            if np.unique(taskclassif_labels_train).size < 2:
                logger.log('WARNING: All labels are %d thus classifier will not be re-fitted' % taskclassif_labels_train[0])
            else:
                logger.log('Fitting Task classifier on fail/success/total obs: %d / %d / %d   ...' %
                           (fail_labels_train_num, success_labels_train_num, taskclassif_obs_train.shape[0]))
                self.task_classifier.fit(taskclassif_obs_train, taskclassif_labels_train)

            #Adding extra rewards based on this classifier
            # typically ambiguous tasks considered to be good and should rew_hide__taskclassif_coeffatt no penalty
            task_prob = self.task_classifier.predict_proba(self.taskclassif_obs[-traj_num:])[:, 1]
            # print('Tasks:',self.taskclassif_obs[-traj_num:])
            # print('Task probabilities:', task_prob)
            if self.taskclassif_rew_alg == 'get_prob_reward':
                taskclassif_rewards = self.rew_hide__taskclassif_coeff * self.get_prob_reward(task_prob,
                                                                                              pow=self.rew_hide__taskclassif_power,
                                                                                              middle=self.rew_hide__taskclassif_middle)
            elif self.taskclassif_rew_alg == 'get_prob_reward_unnorm':
                taskclassif_rewards = self.rew_hide__taskclassif_coeff * self.get_prob_reward_unnorm(task_prob,
                                                                                              pow=self.rew_hide__taskclassif_power,
                                                                                              middle=self.rew_hide__taskclassif_middle)
            else:
                raise ValueError('ERROR: Unknown taskclasif reward alg: %s' % self.taskclassif_rew_alg)

            taskclassif_diagn['obs'] = np.array(self.taskclassif_obs)
            taskclassif_diagn['labels'] = np.array(self.taskclassif_labels)
            taskclassif_diagn['pool_size'] = self.taskclassif_pool_size
            taskclassif_diagn['rew_coeff'] = self.rew_hide__taskclassif_coeff
            taskclassif_diagn['traj_num'] = traj_num
            taskclassif_diagn['prob'] = task_prob
            taskclassif_diagn['rewards'] = taskclassif_rewards
            taskclassif_diagn['rewfunc_power'] = self.rew_hide__taskclassif_power
            taskclassif_diagn['rewfunc_middle'] = self.rew_hide__taskclassif_middle


            for i, path in enumerate(paths['hide']):
                paths['hide'][i]['reward_components']['task_prob'] = task_prob[i]
                paths['hide'][i]['reward_components']['taskclassif_rewards'] = taskclassif_rewards[i]
                paths['hide'][i]['rewards'][-1] += paths['hide'][i]['reward_components']['taskclassif_rewards']

            if self.env.spec.id[:7] == 'Reacher':
                xlim = [-0.2, 0.2]
                ylim = [-0.2, 0.2]
            else:
                xlim = [-1, 1]
                ylim = [-1, 1]

            self.myplotter.plot_xy_reward(x=x_init_eplst,
                                          y=y_init_eplst,
                                          r=taskclassif_rewards,
                                          img_name='xy_taskclassrew_itr' + str(itr),
                                          name='xy_taskclassrew',
                                          r_min=-1*self.rew_hide__taskclassif_coeff,
                                          xlim=xlim, ylim=ylim)

            taskclassif_obs_array = np.array(self.taskclassif_obs)
            self.myplotter.plot_xy_reward(x=taskclassif_obs_array[:, 0],
                                          y=taskclassif_obs_array[:, 1],
                                          r=self.taskclassif_labels,
                                          img_name='xy_tasklabels_itr' + str(itr),
                                          name='xy_tasklabels',
                                          r_min=0., r_max=1.,
                                          xlim=xlim, ylim=ylim)

            self.myplotter.plot_xy_reward(x=taskclassif_obs_train[:, 0],
                                          y=taskclassif_obs_train[:, 1],
                                          r=taskclassif_labels_train,
                                          img_name='xy_tasklabels_train_itr' + str(itr),
                                          name='xy_tasklabels_train',
                                          r_min=0., r_max=1.,
                                          marker_size=15,
                                          xlim=xlim, ylim=ylim)
            logger.record_tabular('hide_rew_task_classif', np.mean(taskclassif_rewards))
        else:
            logger.record_tabular('hide_rew_task_classif', 0)

        logger.record_tabular('hide_taskclassif_middle', self.rew_hide__taskclassif_middle)
        logger.record_tabular('hide_ep_len', np.mean(path_lengths))
        logger.record_tabular('hide_ep_len_max', np.max(path_lengths))
        logger.record_tabular('hide_ep_len_min', np.min(path_lengths))
        logger.record_tabular('hide_rew_action_force', np.mean(rew_action_force_eplst))
        logger.record_tabular('hide_rew_action_dist', np.mean(rew_action_dist_eplst))

        logger.record_tabular('hide_rew_orig', np.mean(rew_orig_rewards_eplst))
        logger.record_tabular('hide_rew_mnist', np.mean(rew_mnist_eplst))
        logger.record_tabular('hide_rew_center_reached', np.mean(rew_center_reached_eplst))
        logger.record_tabular('hide_rew_seek_act_force', np.mean(rew_seek_act_force_eplst))
        logger.record_tabular('hide_rew_seek_time', np.mean(rew_seek_time_eplst))


        logger.record_tabular('hide_action_force', np.mean(action_force_eplst))
        logger.record_tabular('hide_action_distance', np.mean(action_dist_eplst))
        logger.record_tabular('seek_timelen_avg', self.timelen_avg)
        # print('BatchPolOpt: HIDE rewards: ')
        # for path in paths['hide']:
        #     print(path['rewards'])

        if self.adaptive_timelen_avg or self.timelen_reward_fun == 'get_timelen_reward2':
            r_min = -1.
        else:
            r_min = -0.1

        if test:
            self.myplotter.plot_xy_timereward(x=x_init_eplst,
                                        y=y_init_eplst,
                                        r=rew_seek_time_eplst,
                                        r_min=None, #r_min=r_min
                                        img_name='xy_timerew_test_itr' + str(itr),
                                        name='xy_timerew_test')
        else:
            self.myplotter.plot_xy_timereward(x=x_init_eplst,
                                        y=y_init_eplst,
                                        r=rew_seek_time_eplst,
                                        r_min=None, #r_min=r_min
                                        img_name='xy_timerew_itr' + str(itr))


        return paths, taskclassif_diagn

    def hide_rewards_taskclassif(self, itr, paths, test=False):
        """
        Calculating additional rewards for hide agent based on performance of seek agent.
        :param paths:
        :return:
        """
        # Per episode reward lists
        path_lengths = []
        seek_path_lengths = []

        rew_orig_rewards_eplst = []

        rew_seek_time_eplst = []
        rew_seek_act_force_eplst = []

        rew_mnist_eplst = []
        rew_action_force_eplst = []
        rew_action_dist_eplst= []
        rew_center_reached_eplst = []

        action_force_eplst = []
        action_dist_eplst = []

        x_init_eplst = []
        y_init_eplst = []

        xy_abs_init_eplst = []

        traj_num = 0
        self.timerew_percentile = 75
        self.timelen_success_eplst = [0]

        if self.env.spec.id[:7] == 'Reacher':
            xlim = [-0.2, 0.2]
            ylim = [-0.2, 0.2]
        else:
            xlim = [-2.4, 2.4]
            ylim = [-2.4, 2.4]

        for i, path in enumerate(paths['hide']):
            traj_num += 1
            seek_time_len = paths['seek'][i]['rewards'].size
            self.timelen_eplst.append(seek_time_len)
            if seek_time_len < self.timelen_max:
                self.timelen_success_eplst.append(seek_time_len)

        ## Calculating success ratio
        traj_success_ratio = float(len(self.timelen_success_eplst)) / float(traj_num)


        ## Calculating average episode timelen for timelen-based rewards
        timelen_hist_size = np.max([self.timelen_avg_hist_size, traj_num])
        self.timelen_eplst = self.timelen_eplst[-timelen_hist_size:]
        if self.adaptive_timelen_avg:
            if self.adaptive_percentile:
                traj_success_ratio_scaled = self.scale01(traj_success_ratio, self.adaptive_percentile_regulation_zone)
                self.timerew_percentile = np.clip(int(traj_success_ratio_scaled * 100), 1, 99)
                self.timelen_avg = np.percentile(self.timelen_success_eplst, self.timerew_percentile)
            else:
                # This mode is used for the previous scripts
                self.timelen_avg = np.percentile(self.timelen_eplst, self.timerew_percentile)

        logger.log('Traj_success_ratio: %d, Success timelen lst: [%s], Percentile: %d, Timelen: %d' % (
        traj_success_ratio, " ".join(str(x) for x in self.timelen_success_eplst), self.timerew_percentile, self.timelen_avg))

        self.taskclassif_labels_all = []
        self.taskclassif_obs_success = []
        self.taskclassif_obs_fail = []
        self.goal_task_features = []

        self.goals_lowdim = []


        success_timelen = []

        for i, path in enumerate(paths['hide']):

            path_lengths.append(path['rewards'].size)
            paths['hide'][i]['reward_components'] = {}

            rew_orig_rewards_eplst.append(np.sum(path['rewards']))

            self.goal_task_features.append(self.env.env.unwrapped.get_task_features(
                paths['hide'][i]['observations'][self.obs_indx])[0,:])
            self.goals_lowdim.append(paths['seek'][i]['env_infos']['goal'][0])

            #######################################################################
            ## REWARDS
            #######################################################################
            ## Nullifying the main reward since hide should not depend on them
            paths['hide'][i]['rewards_orig'] = copy.deepcopy(paths['hide'][i]['rewards'])
            paths['hide'][i]['rewards'] = np.zeros_like(paths['hide'][i]['rewards'])

            ## Reward/penalize Seek's right answer (use coeff to set if it is reward or panalty)
            paths['hide'][i]['reward_components']['rew_seek_mnist_digit'] = \
                self.rew_hide__digit_correct_coeff * (2 * float(paths['seek'][i]['env_infos']['digit_revealed'][-1]) - 1.0)

            paths['hide'][i]['rewards'][-1] += paths['hide'][i]['reward_components']['rew_seek_mnist_digit']
            rew_mnist_eplst.append(paths['hide'][i]['reward_components']['rew_seek_mnist_digit'])

            ## Reward for Seek reaching the center
            paths['hide'][i]['reward_components']['rew_center_reached'] = \
                self.rew_hide__center_reached_coeff * (2 * float(paths['seek'][i]['env_infos']['center_reached'][-1]) - 1.0)

            paths['hide'][i]['rewards'][-1] += paths['hide'][i]['reward_components']['rew_center_reached']
            rew_center_reached_eplst.append(paths['hide'][i]['reward_components']['rew_center_reached'])

            ## Reward Seek taking time
            hide_time_len = paths['hide'][i]['rewards'].size
            seek_time_len = paths['seek'][i]['rewards'].size
            seek_path_lengths.append(seek_time_len)

            # These rewards were used for table environemnts
            if self.timelen_reward_fun == 'get_timelen_reward2':
                pf.print_warn('Timelen2 reward function is used')
                paths['hide'][i]['reward_components']['rew_seek_time'] = \
                    self.rew_hide__search_time_coeff * self.get_timelen_reward2(t=seek_time_len,
                                                                                t_avg=self.timelen_avg,
                                                                                t_max=self.timelen_max,
                                                                                power=self.rew_hide__search_time_power)
            elif self.timelen_reward_fun == 'get_timelen_reward_with_penalty':
                paths['hide'][i]['reward_components']['rew_seek_time'] = \
                    self.rew_hide__search_time_coeff * self.get_timelen_reward_with_penalty(t=seek_time_len,
                                                                                t_avg=self.timelen_avg,
                                                                                t_max=self.timelen_max,
                                                                                power=self.rew_hide__search_time_power)
            elif self.timelen_reward_fun == 'get_timelen_reward_with_median':
                # pf.print_warn('get_timelen_reward_with_median reward function is used')
                paths['hide'][i]['reward_components']['rew_seek_time'] = \
                    self.rew_hide__search_time_coeff * self.get_timelen_reward_with_median(t=seek_time_len,
                                                                                           t_avg=self.timelen_avg,
                                                                                           t_max=self.timelen_max,
                                                                                           power=self.rew_hide__search_time_power)
            elif self.timelen_reward_fun == 'get_selfplay_rew':
                paths['hide'][i]['reward_components']['rew_seek_time'] = \
                    self.rew_hide__search_time_coeff * self.get_selfplay_rew(t_hide=hide_time_len,
                                                                             t_seek=seek_time_len)

            else:
                raise ValueError('Unknown time rewarding functioin')

            # These rewards were used for maze environments


            # paths['hide'][i]['reward_components']['rew_seek_time'] = self.rew_hide__search_time_coeff * seek_time_len
            paths['hide'][i]['rewards'][-1] += paths['hide'][i]['reward_components']['rew_seek_time']
            rew_seek_time_eplst.append(paths['hide'][i]['reward_components']['rew_seek_time'])

            ## Reward Seek applying actions
            seek_forces = paths['seek'][i]['env_infos']['act_force_norm']
            seek_force_sum = np.sum(seek_forces)
            paths['hide'][i]['reward_components']['rew_seek_act_force'] = self.rew_hide__search_force_coeff * seek_force_sum
            paths['hide'][i]['rewards'][-1] += paths['hide'][i]['reward_components']['rew_seek_act_force']
            rew_seek_act_force_eplst.append(paths['hide'][i]['reward_components']['rew_seek_act_force'])
            action_force_eplst.append(np.mean(paths['hide'][i]['env_infos']['act_force']))

            #######################################################################
            ## PENALTIES
            #######################################################################
            ## Penalty for applying action far from center of the box
            paths['hide'][i]['reward_components']['rew_act_dist'] = \
                self.rew_hide__act_dist_coeff * paths['hide'][i]['env_infos']['act_min_dist_norm']

            paths['hide'][i]['rewards'] += paths['hide'][i]['reward_components']['rew_act_dist']
            self.check_vec_size(paths['hide'][i]['rewards'], paths['hide'][i]['reward_components']['rew_act_dist'], 'rew_act_dist')

            rew_action_dist_eplst.append(np.sum(paths['hide'][i]['reward_components']['rew_act_dist']))
            action_dist_eplst.append(np.mean(paths['hide'][i]['env_infos']['act_min_dist']))

            #######################################################################
            ## Penalty for Applying Force (taking action)
            force_ratio = paths['hide'][i]['env_infos']['act_force_norm']
            if self.rew_hide__actcontrol_middle is not None:
                self.rew_hide__action_coeff = np.sign(traj_success_ratio - self.rew_hide__actcontrol_middle) * (traj_success_ratio - self.rew_hide__actcontrol_middle) ** 2
            paths['hide'][i]['reward_components']['rew_act_force'] = self.rew_hide__action_coeff * force_ratio ** 2

            paths['hide'][i]['rewards'] += paths['hide'][i]['reward_components']['rew_act_force']
            self.check_vec_size(paths['hide'][i]['rewards'], paths['hide'][i]['reward_components']['rew_act_force'], 'rew_act_force')

            rew_action_force_eplst.append(np.sum(paths['hide'][i]['reward_components']['rew_act_force']))

            #######################################################################
            ## Penalty for time
            paths['hide'][i]['rewards'] += self.rew_hide__time_step

            #######################################################################
            ## Diagnostics for the episode
            x_init_eplst.append(paths['seek'][i]['env_infos']['xyz_goal_relative_prev'][0][0])
            y_init_eplst.append(paths['seek'][i]['env_infos']['xyz_goal_relative_prev'][0][1])

            xy_abs_init_eplst.append(paths['seek'][i]['env_infos']['xyz_prev'][0][0:2])

            task_features = self.env.env.unwrapped.get_task_features(paths['seek'][i]['observations'][self.obs_indx])
            self.taskclassif_obs.append(task_features[0, :])

            # Labels assign according to if task is solvable or not.
            # Typically tasks consistently solved within a time budget are solvable
            self.taskclassif_labels.append(int(seek_time_len < (self.timelen_max - 1)))

            ## Creating list of all observations
            if self.taskclassif_labels[-1] == 0:
                self.taskclassif_obs_fail.append(task_features[1:, :])
                self.taskclassif_obs_fail_prev.append(task_features[1:, :])
            else:
                self.taskclassif_obs_success.append(task_features[1:, :])
                self.taskclassif_obs_success_prev.append(task_features[1:, :])
                success_timelen.append(seek_time_len)

        self.goal_task_features = np.array(self.goal_task_features)
        print('Goal task features shape:', self.goal_task_features.shape)


        ## Adaptive middle point if required based on proportion of successes/failures
        print('Seek path lengths: ', seek_path_lengths)
        obs_success_num = len(self.taskclassif_obs_success)
        obs_fail_num = len(self.taskclassif_obs_fail)
        obs_fail_ratio = float(obs_fail_num) / float(obs_success_num + obs_fail_num)
        if self.taskclassif_adaptive_middle:
            self.rew_hide__taskclassif_middle = self.scale01(obs_fail_ratio,
                                                             self.taskclassif_adaptive_middle_regulation_zone)
            logger.log('Adaptive taskclassif middle: %.4f' % self.rew_hide__taskclassif_middle)


        ## After we iterated through everything we should join everything
        # Variables with _prev are not reset every iteration
        self.taskclassif_obs_fail_prev = self.taskclassif_obs_fail_prev[-self.taskclassif_obs_fail_success_hist_size:]
        self.taskclassif_obs_success_prev = self.taskclassif_obs_success_prev[-self.taskclassif_obs_fail_success_hist_size:]

        if len(self.taskclassif_obs_fail) == 0:
            logger.log('Prev failures used for task classification due to no failures in the current run')
            pf.print_warn('NO FAILURE SAMPLES WERE FOUND: ADDING SOME PREV FAILURES TO RUN CLASSIFIER')
            print('Timelenmax = ', self.timelen_max)
            print('Failure history size = ', self.taskclassif_obs_fail_success_hist_size)
            # self.taskclassif_obs_fail.append(self.taskclassif_obs_success[0])
            self.taskclassif_obs_fail = copy.deepcopy(self.taskclassif_obs_fail_prev)
            # print('Added obs to failures:', self.taskclassif_obs_fail)

        self.taskclassif_obs_fail = np.concatenate(self.taskclassif_obs_fail, axis=0)

        # Adding obviously successful observation, when hide is at the beginning
        # it should help us at the beginning when we can not generate positive samples
        self.taskclassif_obs_success.append(self.goal_task_features)
        self.taskclassif_obs_success = np.concatenate(self.taskclassif_obs_success, axis=0)

        obs_success_num = self.taskclassif_obs_success.shape[0]
        obs_fail_num = self.taskclassif_obs_fail.shape[0]
        logger.log('Successful obs = %d Failure obs = %d' % (obs_success_num, obs_fail_num))


        ## Rewards based on classification of tasks (hard/easy)
        taskclassif_diagn = {}
        if self.rew_hide__taskclassif_coeff is not None and self.rew_hide__taskclassif_coeff != 0:

            if self.taskclassif_use_allpoints:
                '''
                This algorithm samples from all observations from the last iteration, 
                but does not touch previous iterations and does not use outdated points
                '''
                self.taskclassif_obs = self.taskclassif_obs[-traj_num:]
                self.taskclassif_labels = self.taskclassif_labels[-traj_num:]

                sample_obs_add = self.taskclassif_pool_size - traj_num
                if sample_obs_add > 0:
                    '''
                    Here we add remaining samples and balance success/failure samples
                    '''
                    samples_total = obs_success_num + obs_fail_num

                    if self.taskclassif_balance_all_labels:
                        sample_obs_add = 2 * min(obs_success_num, obs_fail_num)
                        success_obs_add_num = min(obs_success_num, obs_fail_num)
                    else:
                        #Guarantee that we have enough observations to add
                        sample_obs_add = min(sample_obs_add, samples_total)

                        #Adding positive samples as much as we can
                        if self.taskclassif_balance_positive_labels:
                            success_obs_add_num = int(sample_obs_add / 2.0)
                        else:
                            success_obs_add_num = int(sample_obs_add * (1.0 - obs_fail_ratio))

                        success_obs_add_num = min(success_obs_add_num, obs_success_num)


                    logger.log('Successful samples will be added to pile: %d' % success_obs_add_num)
                    taskclassif_obs_train = []
                    taskclassif_labels_train = []

                    success_obs_sampled, success_indices = sample_multidim(array=self.taskclassif_obs_success,
                                                          samp_num=success_obs_add_num)


                    ## Adding goals as samples that are sure successfull
                    if self.taskclassif_add_goal_as_pos_sampl_num > 1:
                        success_obs_sampled = np.concatenate([success_obs_sampled, self.goal_task_features], axis=0)

                    success_obs_add_num = success_obs_sampled.shape[0]
                    taskclassif_labels_train.extend([1]*success_obs_add_num)

                    #Adding failure samples
                    if self.taskclassif_balance_all_labels:
                        fail_obs_add_num = min(success_obs_add_num, obs_fail_num)
                    else:
                        sample_obs_add_remain = sample_obs_add - success_obs_add_num
                        fail_obs_add_num = min(sample_obs_add_remain, obs_fail_num)


                    fail_obs_sampled, fail_indices = sample_multidim(array=self.taskclassif_obs_fail,
                                                       samp_num=fail_obs_add_num)
                    fail_obs_add_num = fail_obs_sampled.shape[0]
                    logger.log('Fail samples will be added to pile: %d' % fail_obs_add_num)
                    taskclassif_labels_train.extend([0] * fail_obs_add_num)

                    #Combining all samples
                    taskclassif_obs_train = np.concatenate([success_obs_sampled, fail_obs_sampled], axis=0)
                    logger.log('Samples used for task classif success/fail: %d / %d' % (success_obs_add_num, fail_obs_add_num))

                    #Adding initial position samples
                    # WARING: Looks like it was a bug
                    # taskclassif_obs_train = np.concatenate([self.taskclassif_obs, taskclassif_obs_train])
                    taskclassif_obs_train = np.concatenate([taskclassif_obs_train, self.taskclassif_obs])
                    taskclassif_labels_train.extend(self.taskclassif_labels)
                else:
                    taskclassif_obs_train = self.taskclassif_obs
                    taskclassif_labels_train = self.taskclassif_labels

            else:
                '''
                This algorithm only considers initial observations, 
                thus if you need more observations than the last iteration can provide (from initial observations)
                then it takes observations from past iterations
                '''
                taskclassif_size = np.max([self.taskclassif_pool_size, traj_num])
                self.taskclassif_labels = self.taskclassif_labels[-taskclassif_size:]
                self.taskclassif_obs = self.taskclassif_obs[-taskclassif_size:]

                taskclassif_obs_train = self.taskclassif_obs
                taskclassif_labels_train = self.taskclassif_labels


            # Printing what we are fitting
            taskclassif_obs_train = np.array(taskclassif_obs_train)
            print('Fitted observations shape = ', taskclassif_obs_train.shape)
            logger.log('Samples (obs / lbl) to fit: %d / %d' % (taskclassif_obs_train.shape[0], len(taskclassif_labels_train)))
            # print('Seek Path lengths = ', seek_path_lengths)
            # print('Classes assigned = ', self.taskclassif_labels)

            # Counting success/fail labels
            taskclassif_labels_train_array = np.array(taskclassif_labels_train)
            success_labels_train_num = np.count_nonzero(taskclassif_labels_train_array == 1)
            fail_labels_train_num = np.count_nonzero(taskclassif_labels_train_array == 0)

            if np.unique(taskclassif_labels_train).size < 2:
                logger.log('WARNING: All labels are %d thus classifier will not be re-fitted' % taskclassif_labels_train[0])
            else:
                logger.log('Fitting Task classifier on fail/success/total obs: %d / %d / %d   ...' %
                           (fail_labels_train_num, success_labels_train_num, taskclassif_obs_train.shape[0]))

                if self.task_classifier_type == 'gp' or self.task_classifier_type is None:
                    self.task_classifier.fit(taskclassif_obs_train, taskclassif_labels_train)
                else:
                    # Samples/Labels from prev iteration will be used as a validation set
                    self.task_classifier.fit(taskclassif_obs_train, taskclassif_labels_train,
                                             self.taskclassif_obs_train_prev, self.taskclassif_labels_train_prev)
                    self.taskclassif_obs_train_prev = copy.deepcopy(taskclassif_obs_train)
                    self.taskclassif_labels_train_prev = copy.deepcopy(taskclassif_labels_train)


            ## Adding extra rewards to hide based on this classifier
            # typically ambiguous tasks considered to be good and should rew_hide__taskclassif_coeffatt no penalty
            logger.log('Predicting observations from the last task ...')
            task_prob = self.task_classifier.predict_proba(np.array(self.taskclassif_obs[-traj_num:]))[:, 1]
            # print('Tasks:',self.taskclassif_obs[-traj_num:])
            # print('Task probabilities:', task_prob)
            if self.taskclassif_rew_alg == 'get_prob_reward':
                taskclassif_rewards = self.rew_hide__taskclassif_coeff * self.get_prob_reward(task_prob,
                                                                                              pow=self.rew_hide__taskclassif_power,
                                                                                              middle=self.rew_hide__taskclassif_middle)
            elif self.taskclassif_rew_alg == 'get_prob_reward_unnorm':
                taskclassif_rewards = self.rew_hide__taskclassif_coeff * self.get_prob_reward_unnorm(task_prob,
                                                                                              pow=self.rew_hide__taskclassif_power,
                                                                                              middle=self.rew_hide__taskclassif_middle)
            else:
                raise ValueError('ERROR: Unknown taskclasif reward alg: %s' % self.taskclassif_rew_alg)


            taskclassif_diagn['obs'] = np.array(self.taskclassif_obs)
            taskclassif_diagn['labels'] = np.array(self.taskclassif_labels)
            taskclassif_diagn['pool_size'] = self.taskclassif_pool_size
            taskclassif_diagn['rew_coeff'] = self.rew_hide__taskclassif_coeff
            taskclassif_diagn['traj_num'] = traj_num
            taskclassif_diagn['prob'] = task_prob
            taskclassif_diagn['rewards'] = taskclassif_rewards
            taskclassif_diagn['rewfunc_power'] = self.rew_hide__taskclassif_power
            taskclassif_diagn['rewfunc_middle'] = self.rew_hide__taskclassif_middle


            for i, path in enumerate(paths['hide']):
                paths['hide'][i]['reward_components']['task_prob'] = task_prob[i]
                paths['hide'][i]['reward_components']['taskclassif_rewards'] = taskclassif_rewards[i]
                paths['hide'][i]['rewards'][-1] += paths['hide'][i]['reward_components']['taskclassif_rewards']

            self.myplotter.plot_xy_reward(x=x_init_eplst,
                                          y=y_init_eplst,
                                          r=taskclassif_rewards,
                                          img_name='xy_taskclassrew_itr' + str(itr),
                                          name='xy_taskclassrew', xlim=xlim, ylim=ylim)

            taskclassif_obs_array = np.array(self.taskclassif_obs)
            self.myplotter.plot_xy_reward(x=x_init_eplst,
                                          y=y_init_eplst,
                                          r=self.taskclassif_labels,
                                          img_name='xy_tasklabels_itr' + str(itr),
                                          name='xy_tasklabels', xlim=xlim, ylim=ylim,
                                          # r_min=0.,
                                          # r_max=1.
                                          )

            self.myplotter.plot_xy_reward(x=taskclassif_obs_train[:, 0],
                                          y=taskclassif_obs_train[:, 1],
                                          r=taskclassif_labels_train,
                                          img_name='xy_tasklabels_train_itr' + str(itr),
                                          name='xy_tasklabels_train',
                                          # r_min=0.,
                                          # r_max=1.,
                                          marker_size=15, xlim=xlim, ylim=ylim)

            self.myplotter.plot_goal_vec(goals=self.goals_lowdim, init_xy=xy_abs_init_eplst,
                                         xlim=xlim, ylim=ylim,
                                         labels=self.taskclassif_labels, img_name='xy_goal_vec_train_itr' + str(itr))

            logger.record_tabular('hide_rew_task_classif', np.mean(taskclassif_rewards))
        else:
            logger.record_tabular('hide_rew_task_classif', 0)

        logger.record_tabular('hide_taskclassif_middle', self.rew_hide__taskclassif_middle)
        logger.record_tabular('hide_ep_len', np.mean(path_lengths))
        logger.record_tabular('hide_ep_len_max', np.max(path_lengths))
        logger.record_tabular('hide_ep_len_min', np.min(path_lengths))
        logger.record_tabular('hide_rew_action_force', np.mean(rew_action_force_eplst))
        logger.record_tabular('hide_rew_action_dist', np.mean(rew_action_dist_eplst))
        logger.record_tabular('hide_rew_action_coeff', np.mean(self.rew_hide__action_coeff))



        logger.record_tabular('hide_rew_orig', np.mean(rew_orig_rewards_eplst))
        logger.record_tabular('hide_rew_mnist', np.mean(rew_mnist_eplst))
        logger.record_tabular('hide_rew_center_reached', np.mean(rew_center_reached_eplst))
        logger.record_tabular('hide_rew_seek_act_force', np.mean(rew_seek_act_force_eplst))
        logger.record_tabular('hide_rew_seek_time', np.mean(rew_seek_time_eplst))


        logger.record_tabular('hide_action_force', np.mean(action_force_eplst))
        logger.record_tabular('hide_action_distance', np.mean(action_dist_eplst))
        logger.record_tabular('seek_timelen_avg', self.timelen_avg)



        if self.adaptive_timelen_avg or self.timelen_reward_fun == 'get_timelen_reward2':
            r_min = -1.
        else:
            r_min = -0.1


        if test:
            self.myplotter.plot_xy_timereward(x=x_init_eplst,
                                        y=y_init_eplst,
                                        r=rew_seek_time_eplst,
                                        r_min=None, #r_min=r_min
                                        img_name='xy_timerew_test_itr' + str(itr),
                                        name='xy_timerew_test',
                                        xlim=xlim, ylim=ylim)
        else:
            self.myplotter.plot_xy_timereward(x=x_init_eplst,
                                        y=y_init_eplst,
                                        r=rew_seek_time_eplst,
                                        r_min=None, #r_min=r_min
                                        img_name='xy_timerew_itr' + str(itr),
                                        xlim=xlim, ylim=ylim)


        return paths, taskclassif_diagn

    def check_vec_size(self, ar_correct, ar_test, name=''):
        if ar_correct.size != ar_test.size:
            raise ValueError(self.__class__.__name__, ':' + name + ' size(%d) does not match correct size (%d)' % (ar_test.size, ar_correct.size))

    def seek_rewards_pretrained_classifier(self, itr, paths, prefix='', test=False):
        """
        Calculating additional rewards for seek agent.
        :param paths:
        :return:
        """
        ##
        # self.rew_seek__digit_entropy_coeff = 1.
        # self.rew_seek__digit_correct_coeff = 1.
        # self.rew_seek__time_step = -0.01  # Just penalty for taking time steps

        # true_digit
        # act_min_dist
        # act_min_dist_norm
        # act_force
        # act_force_norm
        # act_force_max
        # act_dist_max
        # rew_mnist
        # pred_digit
        # pred_distr
        # pred_entropy

        # Per episode reward lists
        rew_orig_rewards_eplst = []
        path_lengths = []
        rew_mnist_eplst = []
        rew_action_force_eplst = []
        rew_action_dist_eplst= []
        rew_center_reached_eplst = []
        center_reached_eplst = []

        action_force_eplst = []
        action_dist_eplst = []
        rew_dist2center_eplst = []
        rew_mnistANDtargetloc_eplst = []
        rew_final_mnistANDtargetloc_eplst = []

        rew_seek_taskclassif_reward = []

        seek_path_confidences_all = []

        x_init_eplst = []
        y_init_eplst = []

        for i, path in enumerate(paths['seek']):
            path_lengths.append(path['rewards'].size)
            paths['seek'][i]['reward_components'] = {}
            paths['seek'][i]['rewards_orig'] = copy.deepcopy(paths['seek'][i]['rewards'])

            rew_orig_rewards_eplst.append(np.sum(path['rewards']))

            rew_mnist_eplst.append(np.sum(paths['seek'][i]['env_infos']['rew_mnist']))

            #######################################################################
            ## Reward for Seek reaching the target
            paths['seek'][i]['reward_components']['rew_center_reached'] = \
                self.rew_seek__center_reached_coeff * (float(paths['seek'][i]['env_infos']['center_reached'][-1]))
            center_reached_eplst.append(float(paths['seek'][i]['env_infos']['center_reached'][-1]))

            paths['seek'][i]['rewards'][-1] += paths['seek'][i]['reward_components']['rew_center_reached']
            rew_center_reached_eplst.append(paths['seek'][i]['reward_components']['rew_center_reached'])

            #######################################################################
            ## More compex reward for at the same time reaching the goal location (typicaly center)
            # and revealing the digit (at any time)
            paths['seek'][i]['reward_components']['rew_mnistANDtargetloc'] = \
                self.rew_seek__mnistANDtargetloc_coeff * (paths['seek'][i]['env_infos']['rew_mnistANDtargetloc'].astype(dtype=np.float32))

            paths['seek'][i]['rewards'] += paths['seek'][i]['reward_components']['rew_mnistANDtargetloc']
            rew_mnistANDtargetloc_eplst.append(np.sum(paths['seek'][i]['reward_components']['rew_mnistANDtargetloc']))

            #######################################################################
            ## Same as prev, but for the final moment only
            paths['seek'][i]['reward_components']['rew_final_mnistANDtargetloc'] = \
                self.rew_seek__final_mnistANDtargetloc_coeff * (float(paths['seek'][i]['env_infos']['rew_mnistANDtargetloc'][-1]))

            paths['seek'][i]['rewards'][-1] += paths['seek'][i]['reward_components']['rew_final_mnistANDtargetloc']
            rew_final_mnistANDtargetloc_eplst.append(paths['seek'][i]['reward_components']['rew_final_mnistANDtargetloc'])


            #######################################################################
            ## Reward for Seek's distance from target
            paths['seek'][i]['reward_components']['dist2target'] = \
                self.rew_seek__dist2target_coeff * (0.5 - paths['seek'][i]['env_infos']['distance2center_norm'])

            paths['seek'][i]['rewards'] += paths['seek'][i]['reward_components']['dist2target']
            rew_dist2center_eplst.append(np.sum(paths['seek'][i]['reward_components']['dist2target']))

            #######################################################################
            ## Penalty for being far from center of box
            paths['seek'][i]['reward_components']['rew_act_dist'] = \
                self.rew_seek__act_dist_coeff * paths['seek'][i]['env_infos']['act_min_dist_norm']

            paths['seek'][i]['rewards'] += paths['seek'][i]['reward_components']['rew_act_dist']
            self.check_vec_size(paths['seek'][i]['rewards'], paths['seek'][i]['reward_components']['rew_act_dist'], 'rew_act_dist')

            rew_action_dist_eplst.append(np.sum(paths['seek'][i]['reward_components']['rew_act_dist']))
            action_dist_eplst.append(np.mean(paths['seek'][i]['env_infos']['act_min_dist']))

            #######################################################################
            ## Penalty for Applying Force (taking action)
            force_ratio = paths['seek'][i]['env_infos']['act_force_norm']
            paths['seek'][i]['reward_components']['rew_act_force_norm'] = self.rew_seek__action_coeff * force_ratio ** 2

            paths['seek'][i]['rewards'] += paths['seek'][i]['reward_components']['rew_act_force_norm']
            self.check_vec_size(paths['seek'][i]['rewards'], paths['seek'][i]['reward_components']['rew_act_force_norm'], 'rew_act_force_norm')

            rew_action_force_eplst.append(np.sum(paths['seek'][i]['reward_components']['rew_act_force_norm']))
            action_force_eplst.append(np.mean(paths['seek'][i]['env_infos']['act_force']))

            #######################################################################
            ## Adding extra rewards to seek based on the same classifier
            # (i.e. seek would pick more reliable paths when it has poor performance)
            if self.rew_seek__taskclassif_coeff is not None and self.rew_seek__taskclassif_coeff != 0 and not test:
                seek_path_taskfeatures = self.env.env.unwrapped.get_task_features(
                    paths['seek'][i]['observations'][self.obs_indx])
                seek_path_confidences = self.task_classifier.predict_proba(seek_path_taskfeatures)[:, 1]
                seek_path_confidences_all.append(seek_path_confidences)

                if self.taskclassif_rew_alg == 'get_prob_reward':
                    seek_path_confidence_rewards = self.rew_seek__taskclassif_coeff * self.get_prob_reward(
                        seek_path_confidences,
                        pow=self.rew_hide__taskclassif_power,
                        middle=self.rew_hide__taskclassif_middle)
                elif self.taskclassif_rew_alg == 'get_prob_reward_unnorm':
                    seek_path_confidence_rewards = self.rew_seek__taskclassif_coeff * self.get_prob_reward_unnorm(
                        seek_path_confidences,
                        pow=self.rew_hide__taskclassif_power,
                        middle=self.rew_hide__taskclassif_middle)
                else:
                    raise ValueError('ERROR: Unknown taskclasif reward alg: %s' % self.taskclassif_rew_alg)

                # Rolling rewards backwards since reward should be assigned for action (i.e. next obs)
                seek_path_confidence_rewards = np.roll(seek_path_confidence_rewards, -1)
                seek_path_confidence_rewards[-1] = 0

                # Adding these rewards to the rewards we have
                paths['seek'][i]['rewards'] += seek_path_confidence_rewards
                paths['seek'][i]['reward_components']['taskclassif_rewards'] = np.sum(seek_path_confidence_rewards)
                rew_seek_taskclassif_reward.append(paths['seek'][i]['reward_components']['taskclassif_rewards'])


            #######################################################################
            ## Penalty for time
            paths['seek'][i]['rewards'] += self.rew_seek__time_step

            #######################################################################
            ## Diagnostics for the episode
            if self.env.spec.id[:6] == 'Reacher':
                x_init_eplst.append(paths['seek'][i]['env_infos']['xyz_goal_relative'][0][0])
                y_init_eplst.append(paths['seek'][i]['env_infos']['xyz_goal_relative'][0][1])
            else:
                x_init_eplst.append(paths['seek'][i]['env_infos']['xyz_prev_normalized'][0][0])
                y_init_eplst.append(paths['seek'][i]['env_infos']['xyz_prev_normalized'][0][1])

        # print('BatchPolOpt: SEEK rewards: ')
        # for path in paths['seek']:
        #     print(path['rewards'])

        if self.rew_seek__taskclassif_coeff is not None and self.rew_seek__taskclassif_coeff != 0 and not test:
            logger.record_tabular('seek_rew_taskclassif', np.mean(rew_seek_taskclassif_reward))

            # seek_path_confidences_all = np.concatenate(seek_path_confidences_all)
            # conf_fig = plt.figure(10)
            # plt.clf()
            # sns.distplot(seek_path_confidences_all, bins=10, kde=False, rug=True)
            # conf_fig.savefig(self.myplotter.out_dir + 'seek_taskclassif_conf_distrib.jpg')
        elif not test:
            logger.record_tabular('seek_rew_taskclassif', 0)

        logger.record_tabular(prefix + 'seek_rew_orig', np.mean(rew_orig_rewards_eplst))
        logger.record_tabular(prefix + 'seek_ep_len' , np.mean(path_lengths))
        logger.record_tabular(prefix + 'seek_ep_len_max', np.max(path_lengths))
        logger.record_tabular(prefix + 'seek_ep_len_min' , np.min(path_lengths))
        logger.record_tabular(prefix + 'seek_rew_mnist' , np.mean(rew_mnist_eplst))
        logger.record_tabular(prefix + 'seek_rew_center_reached' , np.mean(rew_center_reached_eplst))
        logger.record_tabular(prefix + 'seek_center_reached', np.mean(center_reached_eplst))
        logger.record_tabular(prefix + 'seek_rew_dist2target' , np.mean(rew_dist2center_eplst))
        logger.record_tabular(prefix + 'seek_rew_action_force' , np.mean(rew_action_force_eplst))
        logger.record_tabular(prefix + 'seek_rew_action_distance' , np.mean(rew_action_dist_eplst))
        logger.record_tabular(prefix + 'seek_action_force', np.mean(action_force_eplst))
        logger.record_tabular(prefix + 'seek_action_distance', np.mean(action_dist_eplst))
        logger.record_tabular(prefix + 'seek_rew_mnistANDtargetloc' , np.mean(rew_mnistANDtargetloc_eplst))
        logger.record_tabular(prefix + 'seek_rew_final_mnistANDtargetloc_eplst' , np.mean(rew_final_mnistANDtargetloc_eplst))

        if test:
            self.myplotter.plot_xy_time(x=x_init_eplst,
                                        y=y_init_eplst,
                                        t=path_lengths,
                                        t_max=self.timelen_max,
                                        img_name='xy_time_test_itr' + str(itr), name='xy_time_test')
        else:
            self.myplotter.plot_xy_time(x=x_init_eplst,
                                        y=y_init_eplst,
                                        t=path_lengths,
                                        t_max=self.timelen_max,
                                        img_name='xy_time_itr' + str(itr))


        return paths

    def seek_rewards(self, paths):
        """
        Calculating additional rewards for seek agent.
        :param paths:
        :return:
        """
        ##
        # self.rew_seek__digit_entropy_coeff = 1.
        # self.rew_seek__digit_correct_coeff = 1.
        # self.rew_seek__time_step = -0.01  # Just penalty for taking time steps

        rew_digit_correct_final = []
        rew_digit_entropy = []
        digit_entropy_final = []
        digit_prediction_correct_list = []
        digit_prediction_correct_final_list = []
        path_lengths = []
        rew_digit_correct_sums = []

        for i, path in enumerate(paths['seek']):
            path_lengths.append(path['rewards'].size)

            #######################################################################
            ## Reward/penalize Seek's right answer (use coeff to set if it is reward or panalty)
            # --- Final answer
            digit_distrib = self.get_digit_distr(path['actions'][-1, :])
            digit_guess = np.argmax(digit_distrib)
            digit_prediction_correct_final = (digit_guess == path['env_infos']['digits_in_scene'][-1])
            digit_prediction_correct_final_list.append(digit_prediction_correct_final)
            path['rew_digit_correct_final'] = self.rew_seek__final_digit_correct_coeff * \
                                                    float(digit_prediction_correct_final)

            path['rewards'][-1] += path['rew_digit_correct_final']
            rew_digit_correct_final.append(path['rew_digit_correct_final'])

            path['rewards_orig'] = copy.deepcopy(path['rewards'])

            # --- current answers
            digit_distrib_all = self.get_digit_distr(path['actions'])
            digit_guess_all = np.argmax(digit_distrib_all, axis=1)
            digit_prediction_correct = (digit_guess_all == path['env_infos']['digits_in_scene'][-1])
            digit_prediction_correct_list.append(digit_prediction_correct)

            path['rew_digit_correct'] = self.rew_seek__digit_correct_coeff * digit_prediction_correct.astype(dtype=np.float32)
            rew_digit_correct_sums.append(np.sum(path['rew_digit_correct']))

            path['rewards'] += path['rew_digit_correct']


            #######################################################################
            ## Penalty for time
            path['rewards'] += self.rew_seek__time_step

            #######################################################################
            ## Penalize Seek's ambiguity of digit answer
            # - penalize certainty of incorrect guesses
            # - reward certainty of correction guesses
            digit_distrib = self.get_digit_distr(path['actions'][-1, :])
            digit_guess_entropy = self.entropy(digit_distrib)
            digit_guess_entropy_normalized = digit_guess_entropy / self.entropy_max

            # Reward low entropy of correct answers and penalize low entropy of incorrect ones
            if digit_prediction_correct_final:
                entropy_sign = 1.
            else:
                entropy_sign = -1.
            digit_guess_entropy_reward = entropy_sign * (1.0 - digit_guess_entropy_normalized)
            path['rew_digit_entropy'] = self.rew_seek__final_digit_entropy_coeff * digit_guess_entropy_reward
            path['rewards'][-1] += path['rew_digit_entropy']
            rew_digit_entropy.append(path['rew_digit_entropy'])
            digit_entropy_final.append(digit_guess_entropy)

        digit_prediction_correct_allsamples = np.concatenate(digit_prediction_correct_list)
        logger.record_tabular('seek_accuracy_samplewise', np.mean(digit_prediction_correct_allsamples))
        logger.record_tabular('seek_accuracy_final', np.sum(digit_prediction_correct_final_list) / len(paths['seek']))
        logger.record_tabular('seek_ep_len', np.mean(path_lengths))
        logger.record_tabular('seek_rew_digit_correct_final', np.mean(rew_digit_correct_final))
        logger.record_tabular('seek_rew_digit_entropy_final', np.mean(rew_digit_entropy))
        logger.record_tabular('seek_digit_entropy_final', np.mean(digit_entropy_final))
        logger.record_tabular('seek_rew_digit_correct_sum', np.mean(rew_digit_correct_sums))

        return paths

    def entropy(self, y_pred):
        y_pred_corr = y_pred + 1e-8
        y_norm = np.sum(y_pred_corr)
        y_pred_corr = y_pred_corr / y_norm
        entr = -np.sum(y_pred_corr * np.log(y_pred_corr))
        return entr

    def persample_entropy(y_pred):
        y_pred_corr = y_pred + 1e-8
        y_norm = np.tile(np.expand_dims(np.sum(y_pred_corr, axis=1), axis=1), [1, y_pred_corr.shape[1]])
        y_pred_corr = y_pred_corr / y_norm
        return -np.sum(y_pred_corr * np.log(y_pred_corr), axis=1)

    def process_samples(self, itr, paths, baseline, policy, name):
        """
        Sample processing, such as:
        - calculating statistics: mean reward
        - reward normalization
        - calculating additional rewards
        :param itr:
        :param paths:
        :param baseline:
        :param policy:
        :param name:
        :return:
        """
        ########################################################################
        ## Processing original (main) rewards
        reward_main_pathsums = []
        for i in range(len(paths)):
            # Saving original reward
            paths[i]['reward_main'] = paths[i]['rewards']
            reward_main_pathsums.append(np.sum(paths[i]['reward_main']))
        reward_main_avg = np.mean(reward_main_pathsums)
        logger.record_tabular(name + '_RewAvg_Main', reward_main_avg)
        logger.log('Process Samples | reward_main_avg: %f' %reward_main_avg)

        ########################################################################
        ## Reward normalization (can be applied for every agent)
        # reward_main_norm_pathsums = []
        # if self.normalize_reward:
        #     logger.log('Normalizing rewards ...')
        #     # Update reward mean/std Q.
        #     rewards = []
        #     for i in range(len(paths)):
        #         rewards.append(paths[i]['rewards'])
        #     rewards_flat = np.hstack(rewards)
        #     self._reward_mean.append(np.mean(rewards_flat))
        #     self._reward_std.append(np.std(rewards_flat))
        #
        #     # Normalize rewards.
        #     reward_mean = np.mean(np.asarray(self._reward_mean))
        #     reward_std = np.mean(np.asarray(self._reward_std))
        #     for i in range(len(paths)):
        #         paths[i]['rewards'] = (paths[i]['rewards'] - reward_mean) / (reward_std + 1e-8)
        # logger.log('Process Samples| normalize_reward:%d ' %self.normalize_reward)
        ########################################################################
        ## Dynamics related processing (exploration)
        # !!! Requires works
        # if self.rew_bnn_use:
        #     if itr > 0:
        #         kls = []
        #         for i in range(len(paths)):
        #             kls.append(paths[i]['KL'])
        #
        #         kls_flat = np.hstack(kls)
        #
        #         logger.record_tabular(name + '_Expl_MeanKL', np.mean(kls_flat))
        #         logger.record_tabular(name + '_Expl_StdKL', np.std(kls_flat))
        #         logger.record_tabular(name + '_Expl_MinKL', np.min(kls_flat))
        #         logger.record_tabular(name + '_Expl_MaxKL', np.max(kls_flat))
        #
        #
        #         # Perform normalization of the intrinsic rewards.
        #         if self.use_kl_ratio:
        #             if self.use_kl_ratio_q:
        #                 # Update kl Q
        #                 self.kl_previous.append(np.median(np.hstack(kls)))
        #                 previous_mean_kl = np.mean(np.asarray(self.kl_previous))
        #                 for i in range(len(kls)):
        #                     kls[i] = kls[i] / previous_mean_kl
        #
        #
        #         ## INTRINSIC REWARDS
        #         reward_bnn_pathsums = []
        #         for i in range(len(paths)):
        #             paths[i]['reward_bnn'] = self.eta * kls[i]
        #             paths[i]['rewards'] = paths[i]['rewards'] + self.eta * kls[i]
        #             reward_bnn_pathsums.append(np.sum(paths[i]['reward_bnn']))
        #
        #         reward_bnn_avg = np.mean(reward_bnn_pathsums)
        #         logger.record_tabular(name + '_RewAvg_Dyn', reward_bnn_avg)
        #
        #         # Discount eta
        #         self.eta *= self.eta_discount
        #     else:
        #         logger.record_tabular(name + '_Expl_MeanKL', 0.)
        #         logger.record_tabular(name + '_Expl_StdKL', 0.)
        #         logger.record_tabular(name + '_Expl_MinKL', 0.)
        #         logger.record_tabular(name + '_Expl_MaxKL', 0.)
        #         logger.record_tabular(name + '_RewAvg_Dyn', 0.)
        # logger.log('Process Samples| rew_bnn_use:%d ' % (self.rew_bnn_use))

        ########################################################################
        ## BASELINE FOR A PATH
        baseline_values = []
        returns = []
        for path in paths:
            path_baselines = np.append(baseline.predict(path), 0)
            deltas = path["rewards"] + \
                self.discount * path_baselines[1:] - \
                path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(
                path["rewards"], self.discount)
            baseline_values.append(path_baselines[:-1])
            returns.append(path["returns"])

        ########################################################################
        ## BASELINE FOR A PATH
        if not policy.recurrent:
            observations = e2e_tensor_utils.concat_tensor_list(
                [path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list(
                [path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list(
                [path["rewards"] for path in paths])
            advantages = tensor_utils.concat_tensor_list(
                [path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list(
                [path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list(
                [path["agent_infos"] for path in paths])

            if self.center_adv:
                advantages = util.center_advantages(advantages)

            if self.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [
                sum(path["rewards"]) for path in paths]

            # ent = np.mean(policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baseline_values),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
            # logger.log('Process Samples| policy.recurrent: %d' %policy.recurrent)
        ## POLICY is recurrent
        ##!!! Requires work
        # else:
        #     max_path_length = max([len(path["advantages"]) for path in paths])
        #
        #     # make all paths the same length (pad extra advantages with 0)
        #     obs = [path["observations"] for path in paths]
        #     obs = np.array(
        #         [tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])
        #
        #     if self.center_adv:
        #         raw_adv = np.concatenate(
        #             [path["advantages"] for path in paths])
        #         adv_mean = np.mean(raw_adv)
        #         adv_std = np.std(raw_adv) + 1e-8
        #         adv = [
        #             (path["advantages"] - adv_mean) / adv_std for path in paths]
        #     else:
        #         adv = [path["advantages"] for path in paths]
        #
        #     adv = np.array(
        #         [tensor_utils.pad_tensor(a, max_path_length) for a in adv])
        #
        #     actions = [path["actions"] for path in paths]
        #     actions = np.array(
        #         [tensor_utils.pad_tensor(a, max_path_length) for a in actions])
        #
        #     rewards = [path["rewards"] for path in paths]
        #     rewards = np.array(
        #         [tensor_utils.pad_tensor(r, max_path_length) for r in rewards])
        #
        #     agent_infos = [path["agent_infos"] for path in paths]
        #     agent_infos = tensor_utils.stack_tensor_dict_list(
        #         [tensor_utils.pad_tensor_dict(
        #             p, max_path_length) for p in agent_infos]
        #     )
        #
        #     env_infos = [path["env_infos"] for path in paths]
        #     env_infos = tensor_utils.stack_tensor_dict_list(
        #         [tensor_utils.pad_tensor_dict(
        #             p, max_path_length) for p in env_infos]
        #     )
        #
        #     valids = [np.ones_like(path["returns"]) for path in paths]
        #     valids = np.array(
        #         [tensor_utils.pad_tensor(v, max_path_length) for v in valids])
        #
        #     average_discounted_return = \
        #         np.mean([path["returns"][0] for path in paths])
        #
        #     undiscounted_returns = [sum(path["rewards"]) for path in paths]
        #
        #     # ent = np.mean(policy.distribution.entropy(agent_infos))
        #
        #     ev = special.explained_variance_1d(
        #         np.concatenate(baseline_values),
        #         np.concatenate(returns)
        #     )
        #
        #     samples_data = dict(
        #         observations=obs,
        #         actions=actions,
        #         advantages=adv,
        #         rewards=rewards,
        #         valids=valids,
        #         agent_infos=agent_infos,
        #         env_infos=env_infos,
        #         paths=paths,
        #     )

        logger.record_tabular(name + '_AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular(name + '_AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular(name + '_ExplainedVariance', ev)
        logger.record_tabular(name + '_NumTrajs', len(paths))
        # logger.record_tabular(name + '_Entropy', ent)
        # logger.record_tabular(name + '_Perplexity', np.exp(ent))
        logger.record_tabular(name + '_StdReturn', np.std(undiscounted_returns))
        logger.record_tabular(name + '_MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular(name + '_MinReturn', np.min(undiscounted_returns))

        return samples_data

    def train_brownian(self):
        self.bnn = None  # Left for compatibility

        ###############################################################################
        ## Initialize the parallel sampler
        self.start_worker() ###

        ## Initilizing optimization
        self.init_opt(policy_name='seek')   ### npo_comp.py

        # For saving diagnostics
        self.episode_rewards = []
        self.episode_lengths = []

        # 1. Sample Nearby <-- brownian_agent.py
        # - 현재 starts에서 랜덤하게 한개의 start_state를 선택
        # - init_state에서 시작하여 brownian motion을 하며 지나는 점들을 starts에 넣는다
        # - hide rollout, params['start_pool_size']의 크기만큼 starts sample
        logger.log('Re-sampling new start positions ...')
        self.policies['hide'].sample_nearby(animated=False) ## brownian_agent.py
        logger.log('%d new goals populated' % len(self.policies['hide'].starts))

        if self.env.spec.id[:6] == 'Blocks':
            start_scale = 2.4 # Works for maze1_singlegoal only, so be careful
        else:
            start_scale = 1.0

        variance_mean = self.policies['hide'].action_variance_default
        variance = self.policies['hide'].action_variance_default

        for itr in range(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)

            # 2. Obtain Samples (with parallel workers)
            # <-- _worker_collect_one_path(parallel_sampler_comp.py)
            # <-- rollout_brownian(utils.py)
            # starts에서 init_state와 goal을 sample하여 초기화
            # seek rollout, batch_size의 크기보다는 작은 random한 수의 path sample
            paths = self.obtain_samples(itr)    # animated = False

            # Re-organizing dimensions: paths[i][hide/seek] into pahts[hide/seek][i]
            paths = self.ld2dl(paths)
            logger.log('Obtain Samples | %d paths' %len(paths['seek']))
            # print("***************hide_rewards****************")
            # print(self.policies['hide'].rewards)
            # print("*******************************************")

            ## Sample processing:
            # - calculating additional rewards
            # - baseline fitting
            # - logging everything
            logger.record_tabular('Iteration', itr)

            # 3. Seek Reward Brownian <-- batch_polopt_hide_seek.py
            # - seek의 reward를 계산하고(기존의 binary reward에 추가한다) plot한다.
            # - x_init_eplst, y_init_eplst에는 하나의 path의 시작점(normalized)이 들어있다.
            # - plot 'xy_time' : path의 길이(시간)가 t => 길수록 빨간색
            # - plot 'xy_tasklabels' : path의 길이(시간)가 timelen_max를 넘는지 아닌지가 r
            # paths = self.seek_rewards_brownian(itr=itr, paths=paths)

            # 4. Process Samples
            # - advantage, returns 등 계산
            seek_samples_data = self.process_samples(itr, paths['seek'],
                                                     baseline=self.baselines['seek'],
                                                     policy=self.policies['seek'],
                                                     name='seek')
            samples_data = {'seek': seek_samples_data}
            logger.log('Process Samples | total %d samples' % len(samples_data['seek']['rewards']))
            # print("***************seek_rewards****************")
            # for i in range(len(paths['seek'])):
            #     print(paths['seek'][i]['rewards'])
            # print("*******************************************")

            ## Fitting the baseline
            logger.log("Fitting baseline seek...")
            self.baselines['seek'].fit(paths['seek'])
            logger.log("Fitted")

            ## Logging the hell out of it
            self.policies['seek'].log_diagnostics(paths['seek'])
            self.baselines['seek'].log_diagnostics(paths['seek'])

            ## Optimizing policies
            log_seek_opt_vars = self.optimize_policy(itr, samples_data['seek'], policy_name='seek')

            ## Filtering starts according to performance
            # and re-populating starts
            update_now = (itr % self.starts_update_every_itr == 0)
            update_period = self.starts_update_every_itr

            logger.record_tabular('hide_starts_update_period', update_period)
            logger.record_tabular('hide_starts_update_period_max', self.starts_update_every_itr)

            if update_now:
                # - plot goal_reward: starts의 평균 reward값 => 클수록 빨간색
                # - plot goal new/old: g=필터링 한것, r=start, b=start_old
                # print('++++++++++++++++++ plot +++++++++++++++++++')
                # print('- goal reward => ', len(self.policies['hide'].starts))
                # print('strats:', self.policies['hide'].starts)
                # print('rewards:', self.policies['hide'].rewards)
                self.myplotter.plot_goal_rewards(goals=self.policies['hide'].starts,
                                                 rewards=self.policies['hide'].rewards,
                                                 img_name='goal_rewards_itr%03d' % itr,
                                                 scale=start_scale,
                                                 clear=True, env=self.env)

                # 5. Select Starts
                # - starts의 평균 rewards 값이 r_min과 r_max사이에 있는 값만 걸러낸다
                logger.log('Filtering start positions ...')
                self.policies['hide'].select_starts(success_rate=self.center_reached_ratio)
                logger.log('%d goals selected' % len(self.policies['hide'].starts))
                # print('- goal new/old ')
                # print('starts(g):', len(self.policies['hide'].starts))
                self.myplotter.plot_goals(goals=self.policies['hide'].starts, color=[0,1,0], clear=True, env=self.env)

                # 6. Update Variance
                # Eq (2)-1 :
                # - delta_sigma = variance_diff
                # - k_sigma = self.brown_var_control_coeff
                # - r_avg = self.center_reached_ratio
                # - R_pref = self.center_reached_ratio_max
                # Eq (2)-2 :
                # - sigma = variance_mean
                if self.brown_adaptive_variance == 4:
                    variance_diff = self.brown_var_control_coeff * (self.center_reached_ratio - self.center_reached_ratio_max)
                    variance_diff = np.clip(variance_diff, -0.5, 0.5)
                    logger.log('brown: variance change %f' % variance_diff)
                    variance_mean += variance_diff
                    variance_mean = np.clip(variance_mean, a_min=self.brown_var_min, a_max=1.0)
                else:
                    variance_mean = self.policies['hide'].action_variance_default #using default variance provided in the config

                variance = copy.deepcopy(variance_mean)
                logger.log('Adaptive Variance | r_avg: %f' %self.center_reached_ratio)
                logger.log('Adaptive Variance | variance_mean: [%f, %f]' %(variance[0], variance[1]))


                logger.log('Re-sampling new start positions ...')
                self.policies['hide'].sample_nearby(itr=itr, success_rate=self.center_reached_ratio, variance=variance)
                logger.log('Re-sampled %d new goals %d old goals' % (len(self.policies['hide'].starts), len(self.policies['hide'].starts_old)))
                # print('starts(r):', len(self.policies['hide'].starts))
                # print('starts_old(b):', len(self.policies['hide'].starts_old))
                self.myplotter.plot_goals(goals=self.policies['hide'].starts, color=[1, 0, 0], scale=start_scale, env=self.env)
                self.myplotter.plot_goals(goals=self.policies['hide'].starts_old, color=[0, 0, 1], scale=start_scale, img_name='goals', env=self.env)


            logger.record_tabular('brown_samples_num', self.policies['hide'].brownian_samples_num)
            for var_i, var in enumerate(self.policies['hide'].action_variance):
                logger.record_tabular('brown_act_variance_%02d' % var_i, var)

            logger.record_tabular('brown_sampling_temperature', self.policies['hide'].sampling_temperature)

            ## Saving the hell out of it
            logger.log("saving snapshot...")
            # Just constructs a dictionary with all agent-related objects you care to save
            params = self.get_itr_snapshot(itr, samples_data)

            # Compose diagnostics
            diagnostics = {}
            agent_names = ['seek']
            for agent_name in agent_names:
                diagnostics[agent_name] = {}

                paths = samples_data[agent_name]["paths"]
                if self.store_paths:
                    diagnostics[agent_name]["paths"] = paths

                self.episode_rewards.extend(sum(p["rewards"]) for p in paths)
                self.episode_lengths.extend(len(p["rewards"]) for p in paths)

            ###############################################################################
            # 7. Test Rollout
            ## Testing environment
            # params['test_episode_num']만큼의 path sample
            logger.log('Testing environment ...')
            test_paths = self.test_rollouts_seek()
            # self.seek_rewards_brownian(itr=itr, paths=test_paths, prefix='test_', test=True)

            ## Saving everything
            # This is diagnostics of the task classifier (assigns rewards based on classif of task complexity)
            params['diagnostics'] = diagnostics
            logger.log('Saving mat file ...')
            self.prepare_dict2save(diagnostics=diagnostics, obs_indx_exclude=0)
            self.prepare_dict2save(paths=test_paths['seek'], obs_indx_exclude=0)
            diagnostics = {'train': diagnostics, 'test': test_paths['seek']}

            def dict_check_none(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        dict_check_none(v)
                    else:
                        if v is None:
                            print('k,v',k,v)
                        elif isinstance(v, list):
                            for i in v:
                                if v[i] is None:
                                    print('k,v, i', k, v, i)

            # dict_check_none(diagnostics)
            try:
                sio.savemat(self.diagnostics_dir + 'diagnostics_itr%04d' % itr,
                            mdict=diagnostics,
                            do_compression=True)
            except:
                with open(self.log_dir + 'error_itr_%d.txt' % itr, 'w', encoding='utf-8') as f:
                    pf.print2file(diagnostics, file=f)

            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            ## Rendering few rollouts to show results and record video at the end of every few iterations
            # logger.log('Showing environment ...')
            # self.show_rollouts_seek(itr)

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot(name='seek')
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                          "continue...")

        ## Cleaning up
        self.shutdown_worker()

    def regress_variance(self, N=5, desired_success_rate=0.6, img_name=None):
        variance_dimensions = len(self.prev_variances[0])

        success_rates_train = np.array(self.success_rates[-N:]).reshape(-1,1)
        prev_variances = np.array(self.prev_variances)

        # Plot outputs
        plt.figure(100)
        plt.clf()

        variance_array = []

        for i in range(variance_dimensions):
            variances_train = prev_variances[-N:, i].reshape(-1,1)

            # print('Variance regression: var/rates:', variances_train, success_rates_train)

            # Train the model using the training sets
            self.regr.fit(success_rates_train, variances_train)

            # Make predictions using the testing set
            variance_pred = self.regr.predict([desired_success_rate])
            variance_pred_train = self.regr.predict(success_rates_train)

            # The coefficients
            print('Coefficients: \n', self.regr.coef_)
            # The mean squared error
            print("Mean squared error: %.2f"
                  % mean_squared_error(variances_train, variance_pred_train))
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % r2_score(variances_train, variance_pred_train))

            plt.scatter(variances_train, success_rates_train)
            plt.plot(variance_pred_train, success_rates_train, linewidth=3)


            #It returns 2d array, I need to flatten it, otherwise you get list of np.arrays
            variance_array.append(variance_pred.flatten()[0])
        print('Predicted variance mean: ', variance_array)
        plt.xlabel('action variance')
        plt.ylabel('success rate')

        if img_name is not None:
            savedir = self.myplotter.out_dir + 'variance_regression/'
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(savedir + img_name + '.jpg')

        return variance_array

    def train_brownian_with_uniform(self):
        self.bnn = None  # Left for compatibility
        ###############################################################################
        ## Initialize the parallel sampler
        self.start_worker()

        ## Initilizing optimization
        self.init_opt(policy_name='seek')

        # For saving diagnostics
        self.episode_rewards = []
        self.episode_lengths = []

        logger.log('Re-sampling new start positions ...')
        self.policies['hide'].sample_nearby()
        logger.log('%d new goals populated' % len(self.policies['hide'].starts))

        if self.env.spec.id[:6] == 'Blocks':
            start_scale = 2.4 #Works for maze1_singlegoal only, so be careful
        else:
            start_scale = 1.0

        last_update_itr = 0
        update_period = 1
        variance_mean = self.policies['hide'].action_variance_default
        variance = self.policies['hide'].action_variance_default
        first_rollout_after_update = False

        self.batch_size_max = self.batch_size
        self.brown_starts_new_max = self.policies['hide'].starts_new_select_num
        self.brown_starts_old_max = self.policies['hide'].starts_old_select_num

        for itr in range(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)

            ## Obtaining samples with parallel workers
            paths = self.obtain_samples(itr)

            ## Testing environment
            logger.log('Testing environment ...')
            test_paths = self.uniform_rollouts(samples_num=self.batch_size_uniform)

            # Join train and test dictionaries
            paths.extend(copy.deepcopy(test_paths))

            # Re-organizing dimensions:
            # paths[i][hide/seek] into pahts[hide/seek][i]
            paths = self.ld2dl(paths)
            test_paths = self.ld2dl(test_paths)

            ## Sample processing:
            # - calculating additional rewards
            # - baseline fitting
            # - logging everything
            logger.record_tabular('Iteration', itr)

            # Calculating agent specific rewards
            self.seek_rewards_brownian(itr=itr, paths=test_paths, prefix='test_', test=True)
            paths = self.seek_rewards_brownian(itr=itr, paths=paths)

            if self.brown_uniform_anneal:
                self.center_reached_ratio_test_min = 0.2
                self.center_reached_ratio_test_max = 0.8
                self.brown_samples_ratio = np.clip(1.0 - self.center_reached_ratio_test,
                                                 a_min=self.center_reached_ratio_test_min,
                                                 a_max=self.center_reached_ratio_test_max)

                self.batch_size = int(self.batch_size_max * self.brown_samples_ratio)
                self.batch_size_uniform = self.batch_size_max - self.batch_size
                logger.record_tabular('batch_size', self.batch_size)
                logger.record_tabular('batch_size_uniform', self.batch_size_uniform)

                self.policies['hide'].starts_new_select_num = int(self.brown_samples_ratio * self.brown_starts_new_max)
                self.policies['hide'].starts_old_select_num = int(self.brown_samples_ratio * self.brown_starts_old_max)
                logger.record_tabular('brown_starts_new', self.policies['hide'].starts_new_select_num)
                logger.record_tabular('brown_starts_old', self.policies['hide'].starts_old_select_num )

            seek_samples_data = self.process_samples(itr, paths['seek'],
                                                     baseline=self.baselines['seek'],
                                                     policy=self.policies['seek'],
                                                     name='seek')
            samples_data = {'seek': seek_samples_data}

            ## Fitting the baseline
            logger.log("Fitting baseline seek...")
            self.baselines['seek'].fit(paths['seek'])
            logger.log("Fitted")

            ## Logging the hell out of it
            # self.env.log_diagnostics(paths)
            self.policies['seek'].log_diagnostics(paths['seek'])
            self.baselines['seek'].log_diagnostics(paths['seek'])

            ## Optimizing policies
            # self.optimize_policy(itr, samples_data['hide'], policy_name='hide')
            log_seek_opt_vars = self.optimize_policy(itr, samples_data['seek'], policy_name='seek')

            ## Filtering starts according to performance
            # and re-populating starts

            # Automatic adjustment of the update period parameter
            # if self.starts_adaptive_update_itr:
            #     self.starts_update_every_itr = np.ceil(10 * (1.0 - np.clip(self.center_reached_ratio, 0.1, 0.89)))
            #     logger.log('starts_update_every_itr: %d' % self.starts_update_every_itr)
            # logger.record_tabular('hide_starts_update_period', self.starts_update_every_itr)

            update_now = False
            if self.starts_adaptive_update_itr == 1:
                update_now = ((self.center_reached_ratio > self.center_reached_ratio_max) and ((itr - last_update_itr) >= self.brown_itr_min)) or \
                             ((itr - last_update_itr) >= self.brown_itr_max)
            elif self.starts_adaptive_update_itr == 2:
                update_now = ((self.center_reached_ratio_max < self.center_reached_ratio or self.center_reached_ratio < self.center_reached_ratio_min)
                              and ((itr - last_update_itr) >= self.brown_itr_min)) or ((itr - last_update_itr) >= self.brown_itr_max)
            else:
                update_now = (itr % self.starts_update_every_itr == 0)

            update_period = (itr - last_update_itr)

            #Saving training samples for variance regression (prediction)
            if first_rollout_after_update:
                self.success_rates.append(self.center_reached_ratio)
                self.prev_variances.append(variance)
                first_rollout_after_update = False

                if self.brown_tmax_adaptive or self.brown_t_adaptive == 2:
                    #Increases or decreases max temperature (or control gain) of the sampling temperature control
                    # if (self.brown_success_rate_pref - self.center_reached_ratio) > 0:
                    #     self.policies['hide'].sampling_t_max *= 1.2 #increase of temperature leads to more uniform sampling
                    # else:
                    #     self.policies['hide'].sampling_t_max *= 0.83 #decrease of temperature leads to more biased sampling

                    tmax_adapt_difference = 1.5 * (self.brown_success_rate_pref - self.center_reached_ratio)
                    tmax_adapt_difference = np.clip(tmax_adapt_difference, -0.5, 0.5)
                    tmax_adapt_ratio = 1.0 + tmax_adapt_difference
                    logger.log('brown: tmax_adapt_ratio %f', tmax_adapt_ratio)

                    if self.brown_t_adaptive == 2:
                        self.policies['hide'].sampling_temperature *= tmax_adapt_ratio
                        self.policies['hide'].sampling_temperature = np.clip(self.policies['hide'].sampling_temperature, self.policies['hide'].sampling_t_min, 2.0)
                    elif self.brown_tmax_adaptive:
                        self.policies['hide'].sampling_t_max *= tmax_adapt_ratio
                        self.policies['hide'].sampling_t_max = np.clip(self.policies['hide'].sampling_t_max, self.policies['hide'].sampling_t_min, 2.0)

                # if self.brown_prob_adaptive:
                #     prob_adapt_difference = -2.0 * (self.brown_success_rate_pref - self.center_reached_ratio)
                #     prob_adapt_difference = np.clip(prob_adapt_difference, -0.5, 0.5)
                #     prob_adapt_ratio = 1.0 + prob_adapt_difference
                #     logger.log('brown: prob_adapt_ratio %f', prob_adapt_ratio)
                #     self.policies['hide'].prob_middle *= prob_adapt_ratio
                #
                #     self.policies['hide'].prob_middle = np.clip(self.policies['hide'].prob_middle, 0.2, 0.95)

                if self.brown_prob_adaptive:
                    prob_adapt_difference = -1.5 * (self.brown_success_rate_pref - self.center_reached_ratio)
                    prob_adapt_difference = np.clip(prob_adapt_difference, -0.5, 0.5)
                    logger.log('brown: prob_adapt_difference %f', prob_adapt_difference)
                    self.policies['hide'].prob_middle += prob_adapt_difference

                    self.policies['hide'].prob_middle = np.clip(self.policies['hide'].prob_middle,
                                                                self.policies['hide'].prob_min, self.policies['hide'].prob_max)


                if self.brown_adaptive_variance == 2:
                    if (self.brown_success_rate_pref - self.center_reached_ratio) > 0:
                        variance_mean = np.array(self.policies['hide'].action_variance) * 0.8
                    else:
                        variance_mean = np.array(self.policies['hide'].action_variance) * 1.25
                    variance_mean = np.clip(variance_mean, 0.1, 1.0)

                if self.brown_adaptive_variance == 3:
                    variance_diff = 2.0 * (self.center_reached_ratio - self.brown_success_rate_pref)
                    variance_diff = np.clip(variance_diff, -0.5, 0.5)
                    logger.log('brown: variance change %f' % variance_diff)
                    variance_mean += variance_diff
                    variance_mean = np.clip(variance_mean, a_min=0.1, a_max=1.0)
                    # print('!!!!!!!!!!!!!!!!!! brown: variance mean: ', variance_mean, 'dtype', type(variance_mean))


            logger.record_tabular('hide_starts_update_period', update_period)
            logger.record_tabular('brown_sampling_t_max', self.policies['hide'].sampling_t_max)
            logger.record_tabular('brown_prob_middle', self.policies['hide'].prob_middle)

            if update_now:
                self.myplotter.plot_goal_rewards(goals=self.policies['hide'].starts,
                                                 rewards=self.policies['hide'].rewards,
                                                 img_name='goal_rewards_itr%03d' % itr,
                                                 scale=start_scale,
                                                 clear=True, env=self.env)

                logger.log('Filtering start positions ...')
                self.policies['hide'].select_starts(success_rate=self.center_reached_ratio)
                logger.log('%d goals selected' % len(self.policies['hide'].starts))
                self.myplotter.plot_goals(goals=self.policies['hide'].starts, color=[0,1,0], clear=True, env=self.env)

                logger.log('Re-sampling new start positions ...')

                # success_rates_train = self.success_rates[-N:]
                # variances_train = self.prev_variances[-N:]

                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!! brown_adaptive_variance', self.brown_adaptive_variance)
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!! success rates: ', self.success_rates)
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!! prev_variances: ', self.prev_variances)
                if self.brown_adaptive_variance == 1 and len(self.success_rates) >= 2:
                    variance_mean = self.regress_variance(N=5, desired_success_rate=0.6, img_name='variance_regres_%03d' % itr)
                    variance_mean = np.clip(variance_mean, a_min=0.1, a_max=1.0)
                    # logger.log('Choosing new variance: %s', str(variance_mean))
                if self.brown_adaptive_variance == 2 or self.brown_adaptive_variance == 3 or self.brown_tmax_adaptive == 3:
                    pass
                else:
                    variance_mean = self.policies['hide'].action_variance_default #using default variance provided in the config

                # Variance sampling: it is really necessary for regression only
                if self.brown_adaptive_variance == 1:
                    variance = np.random.normal(loc=variance_mean, scale=0.2)
                    variance = np.abs(variance)
                else:
                    variance = copy.deepcopy(variance_mean)

                first_rollout_after_update = True

                print('!!!!!!!!!!!!!!!+++ brown: variance mean: ', variance, 'dtype', type(variance_mean))
                self.policies['hide'].sample_nearby(itr=itr, success_rate=self.center_reached_ratio, variance=variance)
                logger.log('Re-sampled %d new goals %d old goals' % (len(self.policies['hide'].starts), len(self.policies['hide'].starts_old)))
                self.myplotter.plot_goals(goals=self.policies['hide'].starts, color=[1, 0, 0], scale=start_scale, env=self.env)
                self.myplotter.plot_goals(goals=self.policies['hide'].starts_old, color=[0, 0, 1], scale=start_scale, img_name='goals', env=self.env)

                last_update_itr = itr

            logger.record_tabular('brown_samples_num', self.policies['hide'].brownian_samples_num)
            for var_i, var in enumerate(self.policies['hide'].action_variance):
                logger.record_tabular('brown_act_variance_%02d' % var_i, var)

            logger.record_tabular('brown_sampling_temperature', self.policies['hide'].sampling_temperature)

            ## Saving the hell out of it
            logger.log("saving snapshot...")
            # Just constructs a dictionary with all agent-related objects you care to save
            params = self.get_itr_snapshot(itr, samples_data)

            # Compose diagnostics
            diagnostics = {}
            agent_names = ['seek']
            for agent_name in agent_names:
                diagnostics[agent_name] = {}

                paths = samples_data[agent_name]["paths"]
                if self.store_paths:
                    diagnostics[agent_name]["paths"] = paths

                self.episode_rewards.extend(sum(p["rewards"]) for p in paths)
                self.episode_lengths.extend(len(p["rewards"]) for p in paths)

            ## Saving everything
            # This is diagnostics of the task classifier (assigns rewards based on classif of task complexity)
            params['diagnostics'] = diagnostics
            logger.log('Saving mat file ...')
            self.prepare_dict2save(diagnostics=diagnostics, obs_indx_exclude=0)
            self.prepare_dict2save(paths=test_paths['seek'], obs_indx_exclude=0)
            diagnostics = {'train': diagnostics, 'test': test_paths['seek']}

            def dict_check_none(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        dict_check_none(v)
                    else:
                        if v is None:
                            print('k,v',k,v)
                        elif isinstance(v, list):
                            for i in v:
                                if v[i] is None:
                                    print('k,v, i', k, v, i)

            # dict_check_none(diagnostics)
            try:
                sio.savemat(self.diagnostics_dir + 'diagnostics_itr%04d' % itr,
                            mdict=diagnostics,
                            do_compression=True)
            except:
                with open(self.log_dir + 'error_itr_%d.txt' % itr, 'w', encoding='utf-8') as f:
                    pf.print2file(diagnostics, file=f)

            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            ## Rendering few rollouts to show results and record video at the end of every few iterations
            # logger.log('Showing environment ...')
            # self.show_rollouts_seek(itr)

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot(name='seek')
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                          "continue...")

        ## Cleaning up
        self.shutdown_worker()

    def train_brownian_with_goals(self):
        self.bnn = None  # Left for compatibility
        ###############################################################################
        ## Initialize the parallel sampler
        self.start_worker()

        ## Initilizing optimization
        self.init_opt(policy_name='seek')

        # For saving diagnostics
        self.episode_rewards = []
        self.episode_lengths = []

        if self.env.spec.id[:6] == 'Blocks':
            start_scale = 2.4
        else:
            start_scale = 1.0

        for itr in range(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)

            ## Filtering starts according to performance and re-populating starts
            logger.log('Filtering start positions ...')
            starts_finished, samples_required_num = self.policies['hide'].select_starts()
            logger.log('%d goals selected, %d goals required to add' % (len(self.policies['hide'].starts), samples_required_num))
            self.myplotter.plot_goals(goals=self.policies['hide'].starts, color=[0, 1, 0], clear=True, env=self.env)

            if samples_required_num > 0:
                logger.log('Re-sampling new start positions ...')
                self.policies['hide'].sample_nearby(starts_finished, samples_required_num)

            logger.record_tabular('brown_samples_num', self.policies['hide'].brownian_samples_num)

            logger.log('Re-sampled %d new goals | Total: %d goals | %d old goals' % (samples_required_num,
                len(self.policies['hide'].starts), len(self.policies['hide'].starts_old)))
            self.myplotter.plot_goals(goals=self.policies['hide'].starts, color=[1, 0, 0], scale=start_scale, env=self.env)
            self.myplotter.plot_goals(goals=self.policies['hide'].starts_old, color=[0, 0, 1], scale=start_scale, img_name='goals', env=self.env)


            ## Obtaining samples with parallel workers
            paths = self.obtain_samples(itr)

            # Re-organizing dimensions:
            # paths[i][hide/seek] into pahts[hide/seek][i]
            paths = self.ld2dl(paths)

            ## Sample processing:
            # - calculating additional rewards
            # - baseline fitting
            # - logging everything
            logger.record_tabular('Iteration', itr)

            # Calculating agent specific rewards
            paths = self.seek_rewards_brownian(itr=itr, paths=paths)

            seek_samples_data = self.process_samples(itr, paths['seek'],
                                                     baseline=self.baselines['seek'],
                                                     policy=self.policies['seek'],
                                                     name='seek')
            samples_data = {'seek': seek_samples_data}

            ## Fitting the baseline
            logger.log("Fitting baseline seek...")
            self.baselines['seek'].fit(paths['seek'])
            logger.log("Fitted")

            ## Logging the hell out of it
            # self.env.log_diagnostics(paths)
            self.policies['seek'].log_diagnostics(paths['seek'])
            self.baselines['seek'].log_diagnostics(paths['seek'])

            ## Optimizing policies
            # self.optimize_policy(itr, samples_data['hide'], policy_name='hide')
            log_seek_opt_vars = self.optimize_policy(itr, samples_data['seek'], policy_name='seek')

            ## Saving the hell out of it
            logger.log("saving snapshot...")
            # Just constructs a dictionary with all agent-related objects you care to save
            params = self.get_itr_snapshot(itr, samples_data)

            # Compose diagnostics
            diagnostics = {}
            agent_names = ['seek']
            for agent_name in agent_names:
                diagnostics[agent_name] = {}

                paths = samples_data[agent_name]["paths"]
                if self.store_paths:
                    diagnostics[agent_name]["paths"] = paths

                self.episode_rewards.extend(sum(p["rewards"]) for p in paths)
                self.episode_lengths.extend(len(p["rewards"]) for p in paths)

            ## Testing environment
            logger.log('Testing environment ...')
            if itr % glob.video_scheduler.render_every_iterations == 0:
                logger.log('Showing one test rollout ...')
                show_test_rollouts = 1
            else:
                show_test_rollouts = 0
            test_paths = self.test_rollouts_seek(animated_roolouts_num=show_test_rollouts)
            self.seek_rewards_brownian(itr=itr, paths=test_paths, prefix='test_', test=True)

            ## Plotting information related to start generation
            self.myplotter.plot_goal_rewards(goals=self.policies['hide'].starts,
                                             rewards=self.policies['hide'].rewards,
                                             img_name='goal_rewards_itr%03d' % itr,
                                             scale=start_scale,
                                             clear=True, env=self.env)

            ## Saving everything
            # This is diagnostics of the task classifier (assigns rewards based on classif of task complexity)
            params['diagnostics'] = diagnostics
            logger.log('Saving mat file ...')
            self.prepare_dict2save(diagnostics=diagnostics, obs_indx_exclude=0)
            self.prepare_dict2save(paths=test_paths['seek'], obs_indx_exclude=0)
            diagnostics = {'train': diagnostics, 'test': test_paths['seek']}

            try:
                sio.savemat(self.diagnostics_dir + 'diagnostics_itr%04d' % itr,
                            mdict=diagnostics,
                            do_compression=True)
            except:
                with open(self.log_dir + 'error_itr_%d.txt' % itr, 'w', encoding='utf-8') as f:
                    pf.print2file(diagnostics, file=f)

            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            ## Rendering few rollouts to show results and record video at the end of every few iterations
            # logger.log('Showing environment ...')
            # self.show_rollouts_seek(itr)

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot(name='seek')
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                          "continue...")

        ## Cleaning up
        self.shutdown_worker()

    def train_brownian_reverse_repeat_swap(self):
        # self.n_itr = 2

        self.bnn = None  # Left for compatibility
        ###############################################################################
        ## Initialize the parallel sampler
        self.start_worker()

        ## Initilizing optimization
        self.init_opt(policy_name='seek')

        # For saving diagnostics
        self.episode_rewards = []
        self.episode_lengths = []

        if self.env.spec.id[:6] == 'Blocks':
            start_scale = 2.4
        else:
            start_scale = 1.0

        for itr in range(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)

            ## Alternating reverse/repeat modes of optimization
            self.policies['hide'].reverse_mode = (itr % 2 == 0)

            ## Filtering starts according to performance and re-populating starts
            logger.log('Filtering start positions ...')
            starts_finished, samples_required_num = self.policies['hide'].select_starts()
            logger.log('%d goals selected, %d goals required to add' % (len(self.policies['hide'].starts), samples_required_num))
            self.myplotter.plot_goals(goals=self.policies['hide'].starts, color=[0, 1, 0], clear=True, env=self.env)

            if samples_required_num > 0:
                logger.log('Re-sampling new start positions ...')
                self.policies['hide'].sample_nearby(starts_finished, samples_required_num)

            logger.log('Re-sampled %d new goals | Total: %d goals | %d old goals' % (samples_required_num,
                len(self.policies['hide'].starts), len(self.policies['hide'].starts_old)))
            self.myplotter.plot_goals(goals=self.policies['hide'].starts, color=[1, 0, 0], scale=start_scale, env=self.env)
            self.myplotter.plot_goals(goals=self.policies['hide'].starts_old, color=[0, 0, 1], scale=start_scale, img_name='goals', env=self.env)


            ## Obtaining samples with parallel workers
            paths = self.obtain_samples(itr)

            # Re-organizing dimensions:
            # paths[i][hide/seek] into pahts[hide/seek][i]
            paths = self.ld2dl(paths)

            ## Sample processing:
            # - calculating additional rewards
            # - baseline fitting
            # - logging everything
            logger.record_tabular('Iteration', itr)

            # Calculating agent specific rewards
            paths = self.seek_rewards_brownian(itr=itr, paths=paths)

            seek_samples_data = self.process_samples(itr, paths['seek'],
                                                     baseline=self.baselines['seek'],
                                                     policy=self.policies['seek'],
                                                     name='seek')
            samples_data = {'seek': seek_samples_data}

            ## Fitting the baseline
            logger.log("Fitting baseline seek...")
            self.baselines['seek'].fit(paths['seek'])
            logger.log("Fitted")

            ## Logging the hell out of it
            # self.env.log_diagnostics(paths)
            self.policies['seek'].log_diagnostics(paths['seek'])
            self.baselines['seek'].log_diagnostics(paths['seek'])

            ## Optimizing policies
            # self.optimize_policy(itr, samples_data['hide'], policy_name='hide')
            log_seek_opt_vars = self.optimize_policy(itr, samples_data['seek'], policy_name='seek')

            ## Saving the hell out of it
            logger.log("saving snapshot...")
            # Just constructs a dictionary with all agent-related objects you care to save
            params = self.get_itr_snapshot(itr, samples_data)

            # Compose diagnostics
            diagnostics = {}
            agent_names = ['seek']
            for agent_name in agent_names:
                diagnostics[agent_name] = {}

                paths = samples_data[agent_name]["paths"]
                if self.store_paths:
                    diagnostics[agent_name]["paths"] = paths

                self.episode_rewards.extend(sum(p["rewards"]) for p in paths)
                self.episode_lengths.extend(len(p["rewards"]) for p in paths)

            ## Testing environment
            logger.log('Testing environment ...')
            test_paths = self.test_rollouts_seek()
            self.seek_rewards_brownian(itr=itr, paths=test_paths, prefix='test_', test=True)

            ## Plotting information related to start generation
            self.myplotter.plot_goal_rewards(goals=self.policies['hide'].starts,
                                             rewards=self.policies['hide'].rewards,
                                             img_name='goal_rewards_itr%03d' % itr,
                                             scale=start_scale,
                                             clear=True, env=self.env)

            ## Saving everything
            # This is diagnostics of the task classifier (assigns rewards based on classif of task complexity)
            params['diagnostics'] = diagnostics
            logger.log('Saving mat file ...')
            self.prepare_dict2save(diagnostics=diagnostics, obs_indx_exclude=0)
            self.prepare_dict2save(paths=test_paths['seek'], obs_indx_exclude=0)
            diagnostics = {'train': diagnostics, 'test': test_paths['seek']}

            try:
                sio.savemat(self.diagnostics_dir + 'diagnostics_itr%04d' % itr,
                            mdict=diagnostics,
                            do_compression=True)
            except:
                with open(self.log_dir + 'error_itr_%d.txt' % itr, 'w', encoding='utf-8') as f:
                    pf.print2file(diagnostics, file=f)

            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            ## Rendering few rollouts to show results and record video at the end of every few iterations
            # logger.log('Showing environment ...')
            self.show_rollouts_seek(itr)

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot(name='seek')
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                          "continue...")

        print('Sleeping to see the results ...')
        time.sleep(10)

        ## Cleaning up
        self.shutdown_worker()

    def train_brownian_reverse_repeat(self):
        # self.n_itr = 2

        self.bnn = None  # Left for compatibility
        ###############################################################################
        ## Initialize the parallel sampler
        self.start_worker()

        ## Initilizing optimization
        self.init_opt(policy_name='seek')

        # For saving diagnostics
        self.episode_rewards = []
        self.episode_lengths = []

        if self.env.spec.id[:6] == 'Blocks':
            start_scale = 2.4
        else:
            start_scale = 1.0

        for itr in range(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)

            ####################################################################################
            ## REVERSE AGENT
            ## Filtering starts according to performance and re-populating starts
            logger.log('REVERSE AGENT: Filtering start positions ...')
            starts_finished, samples_required_num = self.policies['hide'].reverse_agent.select_starts()
            logger.log('REVERSE AGENT: %d goals selected, %d goals required to add' % (len(self.policies['hide'].reverse_agent.starts), samples_required_num))
            self.myplotter.plot_goals(goals=self.policies['hide'].reverse_agent.starts, color=[0, 1, 0], clear=True, env=self.env, fig_id=11)

            if samples_required_num > 0:
                logger.log('Re-sampling new start positions ...')
                self.policies['hide'].reverse_agent.sample_nearby(starts_finished, samples_required_num)

            logger.log('REVERSE AGENT: Re-sampled %d new goals | Total: %d goals | %d old goals' % (samples_required_num,
                len(self.policies['hide'].reverse_agent.starts), len(self.policies['hide'].reverse_agent.starts_old)))
            self.myplotter.plot_goals(goals=self.policies['hide'].reverse_agent.starts, color=[1, 0, 0], scale=start_scale, env=self.env, fig_id=11)
            self.myplotter.plot_goals(goals=self.policies['hide'].reverse_agent.starts_old, color=[0, 0, 1], scale=start_scale, img_name='reverse_agent_goals', env=self.env, fig_id=11)

            #Rejected samples
            plt.figure(29)
            plt.clf()
            plt.plot(self.policies['hide'].reverse_agent.rejected_starts_vec)

            ####################################################################################
            ## REPEAT AGENT
            ## Filtering starts according to performance and re-populating starts
            logger.log('REPEAT AGENT: Filtering start positions ...')
            starts_finished, samples_required_num = self.policies['hide'].repeat_agent.select_starts()
            logger.log('REPEAT AGENT: %d goals selected, %d goals required to add' % (len(self.policies['hide'].repeat_agent.starts), samples_required_num))
            self.myplotter.plot_goals(goals=self.policies['hide'].repeat_agent.starts, color=[0, 1, 0], clear=True, env=self.env, fig_id=12)

            if samples_required_num > 0:
                logger.log('Re-sampling new start positions ...')
                self.policies['hide'].repeat_agent.sample_nearby(starts_finished, samples_required_num)

            logger.log('REPEAT AGENT: Re-sampled %d new goals | Total: %d goals | %d old goals' % (samples_required_num,
                len(self.policies['hide'].repeat_agent.starts), len(self.policies['hide'].repeat_agent.starts_old)))
            self.myplotter.plot_goals(goals=self.policies['hide'].repeat_agent.starts, color=[1, 0, 0], scale=start_scale, env=self.env, fig_id=12)
            self.myplotter.plot_goals(goals=self.policies['hide'].repeat_agent.starts_old, color=[0, 0, 1], scale=start_scale, img_name='repeat_agent_goals', env=self.env, fig_id=12)

            # Rejected samples
            plt.figure(29)
            plt.plot(self.policies['hide'].repeat_agent.rejected_starts_vec)
            plt.title('Rejected samples num')
            plt.legend(['reverse', 'repeat'])

            ## Obtaining samples with parallel workers
            paths = self.obtain_samples(itr)

            # Re-organizing dimensions:
            # paths[i][hide/seek] into pahts[hide/seek][i]
            paths = self.ld2dl(paths)

            ## Sample processing:
            # - calculating additional rewards
            # - baseline fitting
            # - logging everything
            logger.record_tabular('Iteration', itr)

            # Calculating agent specific rewards
            paths = self.seek_rewards_brownian(itr=itr, paths=paths)

            seek_samples_data = self.process_samples(itr, paths['seek'],
                                                     baseline=self.baselines['seek'],
                                                     policy=self.policies['seek'],
                                                     name='seek')
            samples_data = {'seek': seek_samples_data}

            ## Fitting the baseline
            logger.log("Fitting baseline seek...")
            self.baselines['seek'].fit(paths['seek'])
            logger.log("Fitted")

            ## Logging the hell out of it
            # self.env.log_diagnostics(paths)
            self.policies['seek'].log_diagnostics(paths['seek'])
            self.baselines['seek'].log_diagnostics(paths['seek'])

            ## Optimizing policies
            # self.optimize_policy(itr, samples_data['hide'], policy_name='hide')
            log_seek_opt_vars = self.optimize_policy(itr, samples_data['seek'], policy_name='seek')

            ## Saving the hell out of it
            logger.log("saving snapshot...")
            # Just constructs a dictionary with all agent-related objects you care to save
            params = self.get_itr_snapshot(itr, samples_data)

            # Compose diagnostics
            diagnostics = {}
            agent_names = ['seek']
            for agent_name in agent_names:
                diagnostics[agent_name] = {}

                paths = samples_data[agent_name]["paths"]
                if self.store_paths:
                    diagnostics[agent_name]["paths"] = paths

                self.episode_rewards.extend(sum(p["rewards"]) for p in paths)
                self.episode_lengths.extend(len(p["rewards"]) for p in paths)

            ## Testing environment
            logger.log('Testing environment ...')
            test_paths = self.test_rollouts_seek()
            self.seek_rewards_brownian(itr=itr, paths=test_paths, prefix='test_', test=True)

            ## Plotting information related to start generation
            self.myplotter.plot_goal_rewards(goals=self.policies['hide'].reverse_agent.starts,
                                             rewards=self.policies['hide'].reverse_agent.rewards,
                                             img_name=None,
                                             scale=start_scale,
                                             clear=True, env=self.env, fig_id=14)

            self.myplotter.plot_goal_rewards(goals=self.policies['hide'].repeat_agent.starts,
                                             rewards=self.policies['hide'].repeat_agent.rewards,
                                             img_name='goal_rewards_itr%03d' % itr,
                                             scale=start_scale,
                                             clear=False, env=self.env, fig_id=14)


            ## Saving everything
            # This is diagnostics of the task classifier (assigns rewards based on classif of task complexity)
            params['diagnostics'] = diagnostics
            logger.log('Saving mat file ...')
            self.prepare_dict2save(diagnostics=diagnostics, obs_indx_exclude=0)
            self.prepare_dict2save(paths=test_paths['seek'], obs_indx_exclude=0)
            diagnostics = {'train': diagnostics, 'test': test_paths['seek']}

            try:
                sio.savemat(self.diagnostics_dir + 'diagnostics_itr%04d' % itr,
                            mdict=diagnostics,
                            do_compression=True)
            except:
                with open(self.log_dir + 'error_itr_%d.txt' % itr, 'w', encoding='utf-8') as f:
                    pf.print2file(diagnostics, file=f)

            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            ## Rendering few rollouts to show results and record video at the end of every few iterations
            # logger.log('Showing environment ...')
            self.show_rollouts_seek(itr)

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot(name='seek')
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                          "continue...")

        print('Sleeping to see the results ...')
        time.sleep(10)

        ## Cleaning up
        self.shutdown_worker()

    def train_brownian_multiseed(self):
        # self.n_itr = 2

        self.bnn = None  # Left for compatibility
        ###############################################################################
        ## Initialize the parallel sampler
        self.start_worker()

        ## Initilizing optimization
        self.init_opt(policy_name='seek')

        # For saving diagnostics
        self.episode_rewards = []
        self.episode_lengths = []

        if self.env.spec.id[:6] == 'Blocks':
            start_scale = 2.4
        else:
            start_scale = 1.0

        for itr in range(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)


            ####################################################################################
            ## REPEAT AGENT
            ## Filtering starts according to performance and re-populating starts
            for agent_id in range(len(self.policies['hide'].agents)):

                brownian_samples_num = 0

                logger.log('AGENT %d: Filtering start positions ...' % agent_id)
                starts_finished, samples_required_num = self.policies['hide'].agents[agent_id].select_starts()
                logger.log('AGENT %d: %d goals selected, %d goals required to add' % (agent_id, len(self.policies['hide'].agents[agent_id].starts), samples_required_num))
                self.myplotter.plot_goals(goals=self.policies['hide'].agents[agent_id].starts, color=[0, 1, 0], clear=(agent_id == 0), env=self.env, fig_id=12)

                if samples_required_num > 0:
                    logger.log('AGENT %d: Re-sampling new start positions ...' % agent_id)
                    self.policies['hide'].agents[agent_id].sample_nearby(starts_finished, samples_required_num, clear_figures=(agent_id == 0), itr=itr)

                brownian_samples_num += self.policies['hide'].agents[agent_id].brownian_samples_num

                logger.log('AGENT %d: Re-sampled %d new goals | Total: %d goals | %d old goals' % (agent_id, samples_required_num,
                                                                                                   len(self.policies['hide'].agents[agent_id].starts), len(self.policies['hide'].agents[agent_id].starts_old)))
                self.myplotter.plot_goals(goals=self.policies['hide'].agents[agent_id].starts, color=[1, 0, 0], scale=start_scale, env=self.env, fig_id=12)
                self.myplotter.plot_goals(goals=self.policies['hide'].agents[agent_id].starts_old, color=[0, 0, 1], scale=start_scale, img_name='repeat_agent_goals', env=self.env, fig_id=12)

                # Rejected samples
                plt.figure(29)
                plt.plot(self.policies['hide'].agents[agent_id].rejected_starts_vec)
                plt.title('Rejected samples num')

            plt.legend(['agent %d' % agent_id for agent_id in range(len(self.policies['hide'].agents))])

            logger.record_tabular('brown_samples_num', brownian_samples_num)

            ## Obtaining samples with parallel workers
            paths = self.obtain_samples(itr)

            # Re-organizing dimensions:
            # paths[i][hide/seek] into pahts[hide/seek][i]
            paths = self.ld2dl(paths)

            ## Sample processing:
            # - calculating additional rewards
            # - baseline fitting
            # - logging everything
            logger.record_tabular('Iteration', itr)

            # Calculating agent specific rewards
            paths = self.seek_rewards_brownian(itr=itr, paths=paths)

            seek_samples_data = self.process_samples(itr, paths['seek'],
                                                     baseline=self.baselines['seek'],
                                                     policy=self.policies['seek'],
                                                     name='seek')
            samples_data = {'seek': seek_samples_data}

            ## Fitting the baseline
            logger.log("Fitting baseline seek...")
            self.baselines['seek'].fit(paths['seek'])
            logger.log("Fitted")

            ## Logging the hell out of it
            # self.env.log_diagnostics(paths)
            self.policies['seek'].log_diagnostics(paths['seek'])
            self.baselines['seek'].log_diagnostics(paths['seek'])

            ## Optimizing policies
            # self.optimize_policy(itr, samples_data['hide'], policy_name='hide')
            log_seek_opt_vars = self.optimize_policy(itr, samples_data['seek'], policy_name='seek')

            ## Saving the hell out of it
            logger.log("saving snapshot...")
            # Just constructs a dictionary with all agent-related objects you care to save
            params = self.get_itr_snapshot(itr, samples_data)

            # Compose diagnostics
            diagnostics = {}
            agent_names = ['seek']
            for agent_name in agent_names:
                diagnostics[agent_name] = {}

                paths = samples_data[agent_name]["paths"]
                if self.store_paths:
                    diagnostics[agent_name]["paths"] = paths

                self.episode_rewards.extend(sum(p["rewards"]) for p in paths)
                self.episode_lengths.extend(len(p["rewards"]) for p in paths)

            ## Testing environment
            logger.log('Testing environment ...')
            test_paths = self.test_rollouts_seek()
            self.seek_rewards_brownian(itr=itr, paths=test_paths, prefix='test_', test=True)

            ## Plotting information related to start generation
            for agent_id in range(len(self.policies['hide'].agents)):
                self.myplotter.plot_goal_rewards(goals=self.policies['hide'].agents[agent_id].starts,
                                                 rewards=self.policies['hide'].agents[agent_id].rewards,
                                                 img_name=None,
                                                 scale=start_scale,
                                                 clear=(agent_id == 0), env=self.env, fig_id=14)


            ## Saving everything
            # This is diagnostics of the task classifier (assigns rewards based on classif of task complexity)
            params['diagnostics'] = diagnostics
            logger.log('Saving mat file ...')
            self.prepare_dict2save(diagnostics=diagnostics, obs_indx_exclude=0)
            self.prepare_dict2save(paths=test_paths['seek'], obs_indx_exclude=0)
            diagnostics = {'train': diagnostics, 'test': test_paths['seek']}

            try:
                sio.savemat(self.diagnostics_dir + 'diagnostics_itr%04d' % itr,
                            mdict=diagnostics,
                            do_compression=True)
            except:
                with open(self.log_dir + 'error_itr_%d.txt' % itr, 'w', encoding='utf-8') as f:
                    pf.print2file(diagnostics, file=f)

            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            ## Rendering few rollouts to show results and record video at the end of every few iterations
            # logger.log('Showing environment ...')
            self.show_rollouts_seek(itr)

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot(name='seek')
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                          "continue...")

        print('Sleeping to see the results ...')
        time.sleep(10)

        ## Cleaning up
        self.shutdown_worker()

    def train_brownian_multiseed_swap_every_update_period(self):
        # self.n_itr = 2

        self.bnn = None  # Left for compatibility
        ###############################################################################
        ## Initialize the parallel sampler
        self.start_worker()

        ## Initilizing optimization
        self.init_opt(policy_name='seek')

        # For saving diagnostics
        self.episode_rewards = []
        self.episode_lengths = []

        # Global metric
        self.global_success_rate_prev = None
        self.global_success_rate_diff_vec = []
        updates_num = 0

        if self.env.spec.id[:6] == 'Blocks':
            start_scale = 2.4
        else:
            start_scale = 1.0

        for itr in range(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)

            ## When to update the distribution of start-goal points
            update_now = (itr % self.starts_update_every_itr == 0)
            if update_now:
                ####################################################################################
                ## Adaptive variance (should happen before agent change, otherwise we will update a wrong agent)
                # Excluding itr 0 since otherwise we don't have statistics for the current agent
                if itr != 0 and self.brown_adaptive_variance == 4:
                    variance_mean = np.array(copy.deepcopy(self.policies['hide'].action_variance))
                    variance_diff = self.brown_var_control_coeff * (self.center_reached_ratio - self.center_reached_ratio_max)
                    variance_diff = np.clip(variance_diff, -0.5, 0.5)
                    logger.log('brown: variance change %f' % variance_diff)
                    variance_mean += variance_diff
                    variance_mean = np.clip(variance_mean, a_min=self.brown_var_min, a_max=1.0)
                    self.policies['hide'].action_variance = copy.deepcopy(variance_mean)

                ####################################################################################
                # Making sure I exclude drops in performance due to
                # change in distribution at the first iteration
                update_seed_agent = (updates_num % self.brown_seed_agent_period == 0)
                updates_num += 1 #MUST COME AFTER WE CHECKED update_seed_agent
                self.global_success_rate_prev = None # will force it to exclude first iteration from improvement estimate

                if itr != 0 and update_seed_agent:
                    # Update prev agent before we select the new one
                    # !!!! MUST RUN BEFORE THE NEW AGENT IS SELECTED WITH select_behavior()
                    print('Improvments observed: ', self.global_success_rate_diff_vec)
                    average_improv = np.mean(self.global_success_rate_diff_vec)
                    self.global_success_rate_diff_vec = []
                    self.policies['hide'].update_q(average_improv)
                    print('Last reward: ', average_improv)

                    self.policies['hide'].select_behavior(force=True)
                elif itr == 0:
                    # Need to make sure that internal mechanisms for checking agent first updates are properly engaged
                    self.policies['hide'].select_behavior(force=True)

                cur_agent_id = self.policies['hide'].agent_id

                ####################################################################################
                ## Filtering starts according to performance and re-populating starts
                logger.log('AGENT %d: Filtering start positions ...' % cur_agent_id)
                starts_finished, samples_required_num = self.policies['hide'].agents[cur_agent_id].select_starts()
                logger.log('AGENT %d: %d goals selected, %d goals required to add' %
                           (cur_agent_id, len(self.policies['hide'].agents[cur_agent_id].starts), samples_required_num))
                self.myplotter.plot_goals(goals=self.policies['hide'].agents[cur_agent_id].starts, color=[0, 1, 0], clear=True, env=self.env, fig_id=12)

                if samples_required_num > 0:
                    logger.log('AGENT %d: Re-sampling new start positions ...' % cur_agent_id)
                    self.policies['hide'].agents[cur_agent_id].sample_nearby(starts_finished,
                                                                             samples_required_num,
                                                                             clear_figures=True,
                                                                             itr=itr,
                                                                             variance=self.policies['hide'].action_variance)

                logger.log('AGENT %d: Re-sampled %d new goals | Total: %d goals | %d old goals' % (cur_agent_id, samples_required_num,
                    len(self.policies['hide'].agents[cur_agent_id].starts), len(self.policies['hide'].agents[cur_agent_id].starts_old)))

                for agent_id in range(len(self.policies['hide'].agents)):
                    self.myplotter.plot_goals(goals=self.policies['hide'].agents[agent_id].starts, color=[1, 0, 0],
                                              scale=start_scale, env=self.env, fig_id=12)
                    self.myplotter.plot_goals(goals=self.policies['hide'].agents[agent_id].starts_old, color=[0, 0, 1],
                                              scale=start_scale, img_name='repeat_agent_goals', env=self.env, fig_id=12)

                    # Rejected samples
                    plt.figure(29)
                    plt.plot(self.policies['hide'].agents[agent_id].rejected_starts_vec)
                    plt.title('Rejected samples num')

                # Getting legend
                plt.legend(['agent %d' % agent_id for agent_id in range(len(self.policies['hide'].agents))])

            # Log seed agent IDs
            logger.record_tabular('seed_agent_id', self.policies['hide'].agent_id)

            # Report action variance
            for agent_id in range(len(self.policies['hide'].agents)):
                for var_i, var in enumerate(self.policies['hide'].agents[agent_id].action_variance):
                    logger.record_tabular('brown_act_variance_%02d_%02d' % (agent_id,var_i), var)

            # Log Q function and probabilities
            for agent_i, q_val in enumerate(self.policies['hide'].Q_func):
                logger.record_tabular('seed_agent_q_%02d' % agent_i, q_val)
            for agent_i, p_val in enumerate(self.policies['hide'].choice_prob):
                logger.record_tabular('seed_agent_p_%02d' % agent_i, p_val)

            ## Obtaining samples with parallel workers
            paths = self.obtain_samples(itr)

            # Re-organizing dimensions:
            # paths[i][hide/seek] into pahts[hide/seek][i]
            paths = self.ld2dl(paths)

            ## Sample processing:
            # - calculating additional rewards
            # - baseline fitting
            # - logging everything
            logger.record_tabular('Iteration', itr)

            # Calculating agent specific rewards
            paths = self.seek_rewards_brownian(itr=itr, paths=paths)

            seek_samples_data = self.process_samples(itr, paths['seek'],
                                                     baseline=self.baselines['seek'],
                                                     policy=self.policies['seek'],
                                                     name='seek')
            samples_data = {'seek': seek_samples_data}

            ## Fitting the baseline
            logger.log("Fitting baseline seek...")
            self.baselines['seek'].fit(paths['seek'])
            logger.log("Fitted")

            ## Logging the hell out of it
            # self.env.log_diagnostics(paths)
            self.policies['seek'].log_diagnostics(paths['seek'])
            self.baselines['seek'].log_diagnostics(paths['seek'])

            ## Optimizing policies
            # self.optimize_policy(itr, samples_data['hide'], policy_name='hide')
            log_seek_opt_vars = self.optimize_policy(itr, samples_data['seek'], policy_name='seek')

            ## Saving the hell out of it
            logger.log("saving snapshot...")
            # Just constructs a dictionary with all agent-related objects you care to save
            params = self.get_itr_snapshot(itr, samples_data)

            # Compose diagnostics
            diagnostics = {}
            agent_names = ['seek']
            for agent_name in agent_names:
                diagnostics[agent_name] = {}

                paths = samples_data[agent_name]["paths"]
                if self.store_paths:
                    diagnostics[agent_name]["paths"] = paths

                self.episode_rewards.extend(sum(p["rewards"]) for p in paths)
                self.episode_lengths.extend(len(p["rewards"]) for p in paths)

            ## Testing environment
            logger.log('Testing environment ...')
            test_paths = self.test_rollouts_seek()
            self.seek_rewards_brownian(itr=itr, paths=test_paths, prefix='test_', test=True)

            if self.global_success_rate_prev is not None:
                # self.global_success_rate_prev = self.center_reached_ratio_test
                self.global_success_rate_diff_vec.append(self.center_reached_ratio_test - self.global_success_rate_prev)
            self.global_success_rate_prev = self.center_reached_ratio_test

            ## Plotting information related to start generation
            for agent_id in range(len(self.policies['hide'].agents)):
                self.myplotter.plot_goal_rewards(goals=self.policies['hide'].agents[agent_id].starts,
                                                 rewards=self.policies['hide'].agents[agent_id].rewards,
                                                 img_name=None,
                                                 scale=start_scale,
                                                 clear=(agent_id == 0), env=self.env, fig_id=14)


            ## Saving everything
            # This is diagnostics of the task classifier (assigns rewards based on classif of task complexity)
            params['diagnostics'] = diagnostics
            logger.log('Saving mat file ...')
            self.prepare_dict2save(diagnostics=diagnostics, obs_indx_exclude=0)
            self.prepare_dict2save(paths=test_paths['seek'], obs_indx_exclude=0)
            diagnostics = {'train': diagnostics, 'test': test_paths['seek']}

            try:
                sio.savemat(self.diagnostics_dir + 'diagnostics_itr%04d' % itr,
                            mdict=diagnostics,
                            do_compression=True)
            except:
                with open(self.log_dir + 'error_itr_%d.txt' % itr, 'w', encoding='utf-8') as f:
                    pf.print2file(diagnostics, file=f)

            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

            ## Rendering few rollouts to show results and record video at the end of every few iterations
            # logger.log('Showing environment ...')
            self.show_rollouts_seek(itr)

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot(name='seek')
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                          "continue...")

        print('Sleeping to see the results ...')
        time.sleep(10)

        ## Cleaning up
        self.shutdown_worker()

    def seek_rewards_brownian(self, itr, paths, prefix='', test=False):
        """
        Calculating additional rewards for seek agent.
        :param paths:
        :return:
        """
        # Per episode reward lists
        rew_orig_rewards_eplst = []
        path_lengths = []
        rew_mnist_eplst = []
        rew_action_force_eplst = []
        rew_action_dist_eplst = []
        rew_center_reached_eplst = []
        center_reached_eplst = []

        action_force_eplst = []
        action_dist_eplst = []
        rew_dist2center_eplst = []
        rew_mnistANDtargetloc_eplst = []
        rew_final_mnistANDtargetloc_eplst = []

        x_init_eplst = []
        y_init_eplst = []

        start_state_taskclassif_labels = []

        for i, path in enumerate(paths['seek']):
            # print("========================== path %d ==============================" %i)
            path_lengths.append(path['rewards'].size)
            paths['seek'][i]['reward_components'] = {}
            paths['seek'][i]['rewards_orig'] = copy.deepcopy(paths['seek'][i]['rewards'])

            rew_orig_rewards_eplst.append(np.sum(path['rewards']))

            # rew_mnist_eplst.append(np.sum(paths['seek'][i]['env_infos']['rew_mnist']))

            #######################################################################
            ## Reward for Seek reaching the target
            paths['seek'][i]['reward_components']['rew_center_reached'] = \
                self.rew_seek__center_reached_coeff * (float(paths['seek'][i]['env_infos']['center_reached'][-1]))
            center_reached_eplst.append(float(paths['seek'][i]['env_infos']['center_reached'][-1]))

            paths['seek'][i]['rewards'][-1] += paths['seek'][i]['reward_components']['rew_center_reached']
            rew_center_reached_eplst.append(paths['seek'][i]['reward_components']['rew_center_reached'])

            # print("+++++++++++++++++ rew_center_reached ++++++++++++++++++++")
            # print(paths['seek'][i]['rewards'])

            #######################################################################
            ## More compex reward for at the same time reaching the goal location (typicaly center)
            # and revealing the digit (at any time)
            # paths['seek'][i]['reward_components']['rew_mnistANDtargetloc'] = \
            #     self.rew_seek__mnistANDtargetloc_coeff * (
            #     paths['seek'][i]['env_infos']['rew_mnistANDtargetloc'].astype(dtype=np.float32))
            #
            # paths['seek'][i]['rewards'] += paths['seek'][i]['reward_components']['rew_mnistANDtargetloc']
            # rew_mnistANDtargetloc_eplst.append(
            #     np.sum(paths['seek'][i]['reward_components']['rew_mnistANDtargetloc']))

            # print("+++++++++++++++++ rew_mnistANDtargetloc ++++++++++++++++++++")
            # print(paths['seek'][i]['rewards'])

            #######################################################################
            ## Same as prev, but for the final moment only
            # paths['seek'][i]['reward_components']['rew_final_mnistANDtargetloc'] = \
            #     self.rew_seek__final_mnistANDtargetloc_coeff * (
            #     float(paths['seek'][i]['env_infos']['rew_mnistANDtargetloc'][-1]))
            #
            # paths['seek'][i]['rewards'][-1] += paths['seek'][i]['reward_components']['rew_final_mnistANDtargetloc']
            # rew_final_mnistANDtargetloc_eplst.append(
            #     paths['seek'][i]['reward_components']['rew_final_mnistANDtargetloc'])

            # print("+++++++++++++++++ rew_final_mnistANDtargetloc ++++++++++++++++++++")
            # print(paths['seek'][i]['rewards'])

            #######################################################################
            ## Reward for Seek's distance from target
            # paths['seek'][i]['reward_components']['dist2target'] = \
            #     self.rew_seek__dist2target_coeff * (0.5 - paths['seek'][i]['env_infos']['distance2center_norm'])
            #
            # paths['seek'][i]['rewards'] += paths['seek'][i]['reward_components']['dist2target']
            # rew_dist2center_eplst.append(np.sum(paths['seek'][i]['reward_components']['dist2target']))

            # print("+++++++++++++++++ distance2center_norm ++++++++++++++++++++")
            # print(paths['seek'][i]['rewards'])

            #######################################################################
            ## Penalty for being far from center of box
            # paths['seek'][i]['reward_components']['rew_act_dist'] = \
            #     self.rew_seek__act_dist_coeff * paths['seek'][i]['env_infos']['act_min_dist_norm']
            #
            # paths['seek'][i]['rewards'] += paths['seek'][i]['reward_components']['rew_act_dist']
            # self.check_vec_size(paths['seek'][i]['rewards'], paths['seek'][i]['reward_components']['rew_act_dist'],
            #                     'rew_act_dist')

            # rew_action_dist_eplst.append(np.sum(paths['seek'][i]['reward_components']['rew_act_dist']))
            # action_dist_eplst.append(np.mean(paths['seek'][i]['env_infos']['act_min_dist']))

            # print("+++++++++++++++++ rew_act_dist ++++++++++++++++++++")
            # print(paths['seek'][i]['rewards'])

            #######################################################################
            ## Penalty for Applying Force (taking action)
            # force_ratio = paths['seek'][i]['env_infos']['act_force_norm']
            # paths['seek'][i]['reward_components'][
            #     'rew_act_force_norm'] = self.rew_seek__action_coeff * force_ratio ** 2

            # paths['seek'][i]['rewards'] += paths['seek'][i]['reward_components']['rew_act_force_norm']
            # self.check_vec_size(paths['seek'][i]['rewards'],
            #                     paths['seek'][i]['reward_components']['rew_act_force_norm'], 'rew_act_force_norm')

            # rew_action_force_eplst.append(np.sum(paths['seek'][i]['reward_components']['rew_act_force_norm']))
            # action_force_eplst.append(np.mean(paths['seek'][i]['env_infos']['act_force']))

            # print("+++++++++++++++++ rew_act_force_norm ++++++++++++++++++++")
            # print(paths['seek'][i]['rewards'])

            #######################################################################
            ## Penalty for time
            paths['seek'][i]['rewards'] += self.rew_seek__time_step

            # print("+++++++++++++++++ rew_seek__time_step ++++++++++++++++++++")
            # print(paths['seek'][i]['rewards'])

            #######################################################################
            ## Diagnostics for the episode
            # x_init_eplst, y_init_eplst에는 하나의 path의 시작점(normalized)이 들어있다.
            x_init_eplst.append(paths['seek'][i]['env_infos']['xyz_prev_normalized'][0][0])
            y_init_eplst.append(paths['seek'][i]['env_infos']['xyz_prev_normalized'][0][1])

            # Labels assign according to if task is solvable or not.
            # Typically tasks consistently solved within a time budget are solvable
            # 만약 path의 길이가 self.timelen_max보다 작을 경우 1 그렇지 않으면 0으로 구분한다.
            seek_time_len = paths['seek'][i]['rewards'].size
            start_state_taskclassif_labels.append(int(seek_time_len < (self.timelen_max - 1)))

        if not test:
            self.center_reached_ratio = np.mean(center_reached_eplst)
        else:
            self.center_reached_ratio_test = np.mean(center_reached_eplst)

        logger.record_tabular(prefix + 'seek_rew_orig', np.mean(rew_orig_rewards_eplst))
        logger.record_tabular(prefix + 'seek_ep_len', np.mean(path_lengths))
        logger.record_tabular(prefix + 'seek_ep_len_max', np.max(path_lengths))
        logger.record_tabular(prefix + 'seek_ep_len_min', np.min(path_lengths))

        # logger.record_tabular(prefix + 'seek_rew_mnist', np.mean(rew_mnist_eplst))
        logger.record_tabular(prefix + 'seek_rew_center_reached', np.mean(rew_center_reached_eplst))
        logger.record_tabular(prefix + 'seek_center_reached', np.mean(center_reached_eplst))
        # logger.record_tabular(prefix + 'seek_rew_dist2target', np.mean(rew_dist2center_eplst))
        # logger.record_tabular(prefix + 'seek_rew_action_force', np.mean(rew_action_force_eplst))
        # logger.record_tabular(prefix + 'seek_rew_action_distance', np.mean(rew_action_dist_eplst))
        # logger.record_tabular(prefix + 'seek_action_force', np.mean(action_force_eplst))  #### not zero
        # logger.record_tabular(prefix + 'seek_action_distance', np.mean(action_dist_eplst))
        # logger.record_tabular(prefix + 'seek_rew_mnistANDtargetloc', np.mean(rew_mnistANDtargetloc_eplst))
        # logger.record_tabular(prefix + 'seek_rew_final_mnistANDtargetloc_eplst',
        #                       np.mean(rew_final_mnistANDtargetloc_eplst))

        if self.env.spec.id[:7] == 'Reacher':
            xlim = [-0.22, 0.22]
            ylim = [-0.22, 0.22]
        else:
            xlim = [-1, 1]
            ylim = [-1, 1]

        if test:
            # env_test를 이용할때,,
            # normalized된 path의 시작점이 x,y값 path의 길이(시간)가 t이다.
            # path의 길이가 길면 red, 짧으면 blue
            self.myplotter.plot_xy_time(x=x_init_eplst,
                                        y=y_init_eplst,
                                        t=path_lengths,
                                        t_max=self.timelen_max,
                                        img_name='xy_time_test_itr' + str(itr), name='xy_time_test',
                                        xlim=xlim, ylim=ylim)
        else:
            # normalized된 path의 시작점이 x,y값 path의 길이(시간)가 t이다.
            # path의 길이가 길면 red, 짧으면 blue
            # print("+++++++++++++++++ plot ++++++++++++++++++++")
            # print('x:', x_init_eplst)
            # print('y:', y_init_eplst)
            # print('xy_time:', path_lengths)
            # print('xy_reward:', start_state_taskclassif_labels)
            # print("+++++++++++++++++++++++++++++++++++++++++++")
            self.myplotter.plot_xy_time(x=x_init_eplst,
                                        y=y_init_eplst,
                                        t=path_lengths,
                                        t_max=self.timelen_max,
                                        img_name='xy_time_itr' + str(itr),
                                        xlim=xlim, ylim=ylim)

            # normalized된 path의 시작점이 x,y값 path의 길이(시간)가 timelen_max를 넘는지 아닌지가 r이다.
            # 만약 path의 길이가 self.timelen_max보다 작을 경우 red(1), 그렇지 않으면 blue(0)
            self.myplotter.plot_xy_reward(x=x_init_eplst,
                                          y=y_init_eplst,
                                          r=start_state_taskclassif_labels,
                                          img_name='xy_tasklabels_itr' + str(itr),
                                          name='xy_tasklabels',
                                          r_min=0., r_max=1.,
                                          xlim=xlim, ylim=ylim)

        return paths