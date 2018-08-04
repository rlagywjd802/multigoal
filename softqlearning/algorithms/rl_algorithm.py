import os
import abc
import copy
import gtimer as gt
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from rllab.misc import logger
from rllab.algos.base import Algorithm

from softqlearning.misc.utils import deep_clone
from softqlearning.misc import tf_utils
from softqlearning.misc.sampler import rollouts

from multigoal.utils import print_format as pf

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
        # print('x,y sizes', x.size, y.size)
        # print('goals:', goals)
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
        min_color = np.tile([0., 0., 1.], [samples, 1])
        max_color = np.tile([1., 0., 0.], [samples, 1])
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


class RLAlgorithm(Algorithm):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            sampler,
            n_epochs=1000,
            n_train_repeat=1,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_render=False,

            ##
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            batch_size_uniform=None,  # for brownian agent mixing uniform and brownian sampling
            brown_uniform_anneal=False,  # brownian training: annealing batch size for brownian agent
            max_path_length=500,
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
            test_episodes_num=25,

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
            mode='seek_force_only',
            use_hide=None,
            use_hide_alg=0,
            rew_hide__search_time_coeff=0.01,  # 1.
            rew_hide__action_coeff=-0.01,  # -1.
            rew_seek__action_coeff=-0.01,  # -1.
            rew_hide__digit_entropy_coeff=1,  # 1.
            rew_hide__digit_correct_coeff=1,  # 1. #make <0 if we want to penalize correct predicitons by seek
            rew_hide__time_step=-0.01,  # -0.01 # Just penalty for taking time steps
            rew_hide__act_dist_coeff=-0.05,
            rew_hide__search_force_coeff=0.1,
            rew_hide__center_reached_coeff=0.,
            rew_seek__taskclassif_coeff=None,
            rew_seek__final_digit_entropy_coeff=1,  # 1.
            rew_seek__digit_entropy_coeff=0.01,  # 1.
            rew_seek__final_digit_correct_coeff=1,  # 1.
            rew_seek__digit_correct_coeff=0.01,  # 1.
            rew_seek__time_step=-0.01,  # -0.01  # Just penalty for taking time steps
            rew_seek__act_dist_coeff=-0.05,
            rew_seek__center_reached_coeff=0.,
            rew_seek__dist2target_coeff=0.,
            rew_seek__mnistANDtargetloc_coeff=0.,
            rew_seek__final_mnistANDtargetloc_coeff=0.,
            train_seek_every=1.,
            timelen_max=10,
            timelen_avg=4,
            timelen_reward_fun='get_timelen_reward_with_penalty',
            adaptive_timelen_avg=False,
            adaptive_percentile=False,
            adaptive_percentile_regulation_zone=[0.0, 1.0],
            timelen_avg_hist_size=100,
            task_classifier='gp',
            rew_hide__search_time_power=3,
            rew_hide__taskclassif_coeff=None,
            rew_hide__taskclassif_power=3,
            rew_hide__taskclassif_middle=0.25,
            rew_hide__actcontrol_middle=None,  # action control coeff offset. If None == turned off
            taskclassif_adaptive_middle=False,
            taskclassif_adaptive_middle_regulation_zone=[0.0, 1.0],
            taskclassif_pool_size=100,
            taskclassif_use_allpoints=True,
            taskclassif_balance_positive_labels=True,
            taskclassif_add_goal_as_pos_sampl_num=1,
            taskclassif_rew_alg='get_prob_reward',
            taskclassif_balance_all_labels=False,
            hide_stop_improve_after=None,
            hide_tmax=None,
            starts_update_every_itr=5,
            starts_adaptive_update_itr=False,
            center_reached_ratio_max=0.8,
            center_reached_ratio_min=0.5,
            brown_adaptive_variance=None,
            brown_variance_min=0.1,
            brown_var_control_coeff=2.0,
            brown_tmax_adaptive=False,
            brown_t_adaptive=None,
            brown_prob_middle_adaptive=False,
            brown_success_rate_pref=0.6,
            brown_seed_agent_period=1,
            brown_itr_min=1,
            brown_itr_max=10,
            obs_indx=1,
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_render (`int`): Whether or not to render the evaluation
                environment.
        """
        self.sampler = sampler

        self._n_itr = n_itr
        self._n_train_repeat = n_train_repeat
        self._epoch_length = epoch_length

        self._eval_n_episodes = eval_n_episodes
        self._eval_render = eval_render

        # self.env = None
        # self.policy = None
        self.pool = None

        #####################################################
        # TRPO
        self.env_test = env_test
        self.log_dir = logger.get_snapshot_dir()
        if self.log_dir[-1] != '/':
            self.log_dir += '/'

        self.diagnostics_dir = self.log_dir + 'diagnostics_log/'
        if not os.path.exists(self.diagnostics_dir):
            os.makedirs(self.diagnostics_dir)

        self.diagnostics = {}
        self.diagnostics['iter'] = []
        self.test_episodes_num = test_episodes_num
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.batch_size_uniform = batch_size_uniform
        self.brown_uniform_anneal = brown_uniform_anneal
        self.max_path_length = max_path_length
        # self.discount = discount
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
        self.timelen_avg = timelen_avg
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
        self.taskclassif_balance_positive_labels = taskclassif_balance_positive_labels  # balance positive labels
        self.taskclassif_balance_all_labels = taskclassif_balance_all_labels
        self.taskclassif_rew_alg = taskclassif_rew_alg
        # ----------------------------
        # Rewards
        self.mode = mode

        self.use_hide = use_hide
        self.use_hide_alg = use_hide_alg
        self.rew_hide__search_time_coeff = rew_hide__search_time_coeff  # 1.

        self.rew_hide__actcontrol_middle = rew_hide__actcontrol_middle
        self.rew_hide__action_coeff = rew_hide__action_coeff  # -1.
        self.rew_hide__digit_entropy_coeff = rew_hide__digit_entropy_coeff  # 1.
        self.rew_hide__digit_correct_coeff = rew_hide__digit_correct_coeff  # 1. #make <0 if we want to penalize correct predicitons by seek
        self.rew_hide__time_step = rew_hide__time_step  # -0.01 # Just penalty for taking time steps
        self.rew_hide__act_dist_coeff = rew_hide__act_dist_coeff  # Coeff for punishing large actions
        self.rew_hide__search_force_coeff = rew_hide__search_force_coeff  # Reward hide for seek taking actions coeff
        self.rew_hide__center_reached_coeff = rew_hide__center_reached_coeff

        self.rew_seek__taskclassif_coeff = rew_seek__taskclassif_coeff
        self.rew_seek__action_coeff = rew_seek__action_coeff  # -1.
        self.rew_seek__final_digit_entropy_coeff = rew_seek__final_digit_entropy_coeff  # 1.
        self.rew_seek__digit_entropy_coeff = rew_seek__digit_entropy_coeff  # 1.
        self.rew_seek__digit_correct_coeff = rew_seek__digit_correct_coeff  # 1.
        self.rew_seek__final_digit_correct_coeff = rew_seek__final_digit_correct_coeff  # 1.
        self.rew_seek__time_step = rew_seek__time_step  # -0.01  # Just penalty for taking time steps
        self.rew_seek__act_dist_coeff = rew_seek__act_dist_coeff  # Coeff for punishing large actions
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

        # if isinstance(task_classifier, str):
        #     task_classifier = task_classifier.lower()
        # if task_classifier == 'gp':
        #     self.task_classifier = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
        #     self.task_classifier_type = 'gp'
        # else:
        #     self.task_classifier_type = 'ext'
        #     self.task_classifier = task_classifier

        self.taskclassif_add_goal_as_pos_sampl_num = taskclassif_add_goal_as_pos_sampl_num

        ## Brownian agent parameters
        self.starts_update_every_itr = int(starts_update_every_itr)
        self.starts_adaptive_update_itr = starts_adaptive_update_itr
        # self.brown_itr_min = brown_itr_min
        # self.brown_itr_max = brown_itr_max
        self.center_reached_ratio = 0
        self.center_reached_ratio_test = 0
        self.center_reached_ratio_max = center_reached_ratio_max
        self.center_reached_ratio_min = center_reached_ratio_min

        self.brown_adaptive_variance = brown_adaptive_variance
        self.brown_var_min = brown_variance_min
        self.brown_var_control_coeff = brown_var_control_coeff
        # self.brown_tmax_adaptive = brown_tmax_adaptive
        # self.brown_t_adaptive = brown_t_adaptive
        # self.brown_prob_adaptive = brown_prob_middle_adaptive
        # self.brown_success_rate_pref = brown_success_rate_pref

        ## Multi seed agent params
        # self.brown_seed_agent_period = int(brown_seed_agent_period)

        # Create linear regression object for variance prediction
        # self.regr = linear_model.LinearRegression()
        # self.success_rates = []
        # self.prev_variances = []

        # ----------------------
        self.obs_indx = obs_indx

        self.rew_best_bias = 0
        self.agent_names = ['hide', 'seek']
        # self.Fxy_max = np.linalg.norm(self.env.action_space.high[2:4], ord=2)
        # self.digit_distr_uniform = np.array([1. / 9.] * 9)
        # self.entropy_max = self.entropy(self.digit_distr_uniform)

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
        for key, val in locals().items(): print(key, ': ', val)
        pf.print_sec0_end()

        if self.use_hide_alg == 0:
            self.myplotter = myPlotter(out_dir=self.log_dir + 'graph_log')
        else:
            self.myplotter = myPlotter(out_dir=self.log_dir + 'graph_log',
                                       graph_names=['xy_time', 'xy_time_test', 'xy_tasklabels'])

    def _train(self, env, policy, pool):
        """Perform RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            pool (`PoolBase`): Sample pool to add samples to
        """
        self._init_training()
        self.sampler.initialize(env, policy, pool)

        evaluation_env = deep_clone(env) if self._eval_n_episodes else None

        with tf_utils.get_default_session().as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(
                    range(self._n_epochs + 1), save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    self.sampler.sample()
                    if not self.sampler.batch_ready():
                        continue
                    gt.stamp('sample')

                    for i in range(self._n_train_repeat):
                        self._do_training(
                            iteration=t + epoch * self._epoch_length,
                            batch=self.sampler.random_batch())
                    gt.stamp('train')

                self._evaluate(policy, evaluation_env)
                gt.stamp('eval')

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)

                time_itrs = gt.get_times().stamps.itrs
                time_eval = time_itrs['eval'][-1]
                time_total = gt.get_times().total
                time_train = time_itrs.get('train', [0])[-1]
                time_sample = time_itrs.get('sample', [0])[-1]

                logger.record_tabular('time-train', time_train)
                logger.record_tabular('time-eval', time_eval)
                logger.record_tabular('time-sample', time_sample)
                logger.record_tabular('time-total', time_total)
                logger.record_tabular('epoch', epoch)

                self.sampler.log_diagnostics()

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

            self.sampler.terminate()

    def _train_brownian(self, env, policies, pool):
        """Perform RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            pool (`PoolBase`): Sample pool to add samples to
        """

        ## Initilizing optimization
        self._init_training()

        # Initialize the sampler
        self.sampler.initialize(env, policies, pool)

        # For saving diagnostics
        self.episode_rewards = []
        self.episode_lengths = []

        # 1. Sample Nearby <-- brownian_agent.py
        # - 현재 starts에서 랜덤하게 한개의 start_state를 선택
        # - init_state에서 시작하여 brownian motion을 하며 지나는 점들을 starts에 넣는다
        # - hide rollout, params['start_pool_size']의 크기만큼 starts sample
        logger.log('Re-sampling new start positions ...')
        policies['hide'].sample_nearby(animated=False)
        logger.log('%d new goals populated' % len(policies['hide'].starts))

        if env.spec.id[:6] == 'Blocks':
            start_scale = 2.4 #Works for maze1_singlegoal only, so be careful
            xlim = [-1.0, 1.0]
            ylim = [-1.0, 1.0]
        else:
            start_scale = 1.0
            xlim = [-0.22, 0.22]
            ylim = [-0.22, 0.22]

        variance_mean = policies['hide'].action_variance_default
        variance = policies['hide'].action_variance_default

        with tf_utils.get_default_session().as_default():
            total_episode_length = 0
            for itr in range(self._n_itr):
                logger.push_prefix('itr #%d | ' % itr)
                logger.log("Sample a path") ## should be batch_size
                x_init_eplst = []
                y_init_eplst = []
                path_lengths = []
                start_state_taskclassif_labels = []
                for n in range(20):
                    for t in range(self.max_path_length):
                        # 2. Obtain Samples
                        # <-- sample_sql(sampler.py)
                        # starts에서 init_state와 goal을 sample하여 초기화
                        # seek rollout, 20개의 path sample
                        # - x_init_eplst, y_init_eplst에는 하나의 path의 시작점(normalized)이 들어있다.
                        # - plot 'xy_time' : path의 길이(시간)가 t => 길수록 빨간색
                        # - plot 'xy_tasklabels' : path의 길이(시간)가 timelen_max를 넘는지 아닌지가 r
                        start_pose, done, path_length = self.sampler.sample_sql(animated=False) ## add reward shaping and plotting
                        if done:
                            print("---------------------->done: path_length:", path_length)
                            x_init_eplst.append(start_pose[0][0]/2.4)
                            y_init_eplst.append(start_pose[0][1]/2.4)
                            path_lengths.append(path_length)
                            start_state_taskclassif_labels.append(int(path_length < (self.timelen_max - 1)))
                            total_episode_length += path_length
                            break
                        if not self.sampler.batch_ready():
                            continue
                        else:
                            print("############################### ready")
                        for i in range(self._n_train_repeat):
                            print("---------------------->training")
                            self._do_training(
                                iteration=t + total_episode_length,
                                batch=self.sampler.random_batch())

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

                update_now = (itr % (self.starts_update_every_itr*10) == 0)
                update_period = self.starts_update_every_itr

                # Saving training samples for variance regression (prediction)
                logger.record_tabular('hide_starts_update_period', update_period)
                logger.record_tabular('hide_starts_update_period_max', self.starts_update_every_itr)

                if update_now:
                    self.myplotter.plot_goal_rewards(goals=policies['hide'].starts,
                                                     rewards=policies['hide'].rewards,
                                                     img_name='goal_rewards_itr%03d' % itr,
                                                     scale=start_scale,
                                                     clear=True, env=env)

                    logger.log('Filtering start positions ...')
                    policies['hide'].select_starts(success_rate=self.center_reached_ratio)
                    print("*************** hide_starts ****************")
                    print(policies['hide'].starts)
                    print("*******************************************")
                    logger.log('%d goals selected' % len(policies['hide'].starts))
                    self.myplotter.plot_goals(goals=policies['hide'].starts, color=[0, 1, 0], clear=True,
                                              env=env)

                    logger.log('Re-sampling new start positions ...')

                    # Update Variance
                    if self.brown_adaptive_variance == 4:
                        variance_diff = self.brown_var_control_coeff * (
                                    self.center_reached_ratio - self.center_reached_ratio_max)
                        variance_diff = np.clip(variance_diff, -0.5, 0.5)
                        logger.log('brown: variance change %f' % variance_diff)
                        variance_mean += variance_diff
                        variance_mean = np.clip(variance_mean, a_min=self.brown_var_min, a_max=1.0)
                    else:
                        variance_mean = self.policies['hide'].action_variance_default  # using default variance provided in the config

                    variance = copy.deepcopy(variance_mean)
                    logger.log('Adaptive Variance | r_avg: %f' % self.center_reached_ratio)
                    logger.log('Adaptive Variance | variance_mean: [%f, %f]' % (variance[0], variance[1]))

                    # print('!!!!!!!!!!!!!!!+++ brown: variance mean: ', variance, 'dtype', type(variance_mean))
                    policies['hide'].sample_nearby(itr=itr, success_rate=self.center_reached_ratio,
                                                        variance=variance, animated=False)
                    logger.log('Re-sampled %d new goals %d old goals' % (
                    len(policies['hide'].starts), len(policies['hide'].starts_old)))
                    self.myplotter.plot_goals(goals=policies['hide'].starts, color=[1, 0, 0], scale=start_scale,
                                              env=env)
                    self.myplotter.plot_goals(goals=policies['hide'].starts_old, color=[0, 0, 1],
                                              scale=start_scale, img_name='goals', env=env)

                # Test environment
                # self._eval_n_episodes만큼 rollout
                paths = self._evaluate(policies, self.env_test)
                x_init_eplst = [path['observations'][0][0] for path in paths]
                y_init_eplst = [path['observations'][0][1] for path in paths]
                path_lengths = [path['rewards'].size for path in paths]

                # env_test를 이용할때,,
                # normalized된 path의 시작점이 x,y값 path의 길이(시간)가 t이다.
                # path의 길이가 길면 red, 짧으면 blue
                self.myplotter.plot_xy_time(x=x_init_eplst,
                                            y=y_init_eplst,
                                            t=path_lengths,
                                            t_max=self.timelen_max,
                                            img_name='xy_time_test_itr' + str(itr), name='xy_time_test',
                                            xlim=xlim, ylim=ylim)

                params = self.get_snapshot(itr)
                logger.save_itr_params(itr, params)

                # time_itrs = gt.get_times().stamps.itrs
                # time_eval = time_itrs['eval'][-1]
                # time_total = gt.get_times().total
                # time_train = time_itrs.get('train', [0])[-1]
                # time_sample = time_itrs.get('sample', [0])[-1]
                #
                # logger.record_tabular('time-train', time_train)
                # logger.record_tabular('time-eval', time_eval)
                # logger.record_tabular('time-sample', time_sample)
                # logger.record_tabular('time-total', time_total)
                logger.record_tabular('itr', itr)

                self.sampler.log_diagnostics()

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

            self.sampler.terminate()


    def _evaluate(self, policies, evaluation_env):
        """Perform evaluation for the current policy."""

        if self._eval_n_episodes < 1:
            return

        # TODO: max_path_length should be a property of environment.
        paths = rollouts(evaluation_env, policies['seek'], self.sampler._max_path_length,
                         self._eval_n_episodes)

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]

        logger.record_tabular('return-average', np.mean(total_returns))
        logger.record_tabular('return-min', np.min(total_returns))
        logger.record_tabular('return-max', np.max(total_returns))
        logger.record_tabular('return-std', np.std(total_returns))
        logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
        logger.record_tabular('episode-length-min', np.min(episode_lengths))
        logger.record_tabular('episode-length-max', np.max(episode_lengths))
        logger.record_tabular('episode-length-std', np.std(episode_lengths))

        evaluation_env.log_diagnostics(paths)
        if self._eval_render:
            evaluation_env.render(paths)

        if self.sampler.batch_ready():
            batch = self.sampler.random_batch()
            self.log_diagnostics(batch)

        return paths

    @abc.abstractmethod
    def log_diagnostics(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self):
        raise NotImplementedError
