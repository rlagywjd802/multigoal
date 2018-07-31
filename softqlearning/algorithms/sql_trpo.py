import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.misc.overrides import overrides

from multigoal.softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from multigoal.softqlearning.misc import tf_utils
from .rl_algorithm import RLAlgorithm

from scipy import io as sio
from multigoal.utils import print_format as pf

from rllab_utils.sampler import parallel_sampler_comp as parallel_sampler

EPS = 1e-6


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


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


class SQL(RLAlgorithm):
    """Soft Q-learning (SQL).

    Example:
        See `examples/mujoco_all_sql.py`.

    Reference:
        [1] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine,
        "Reinforcement Learning with Deep Energy-Based Policies," International
        Conference on Machine Learning, 2017. https://arxiv.org/abs/1702.08165
    """

    def __init__(
            self,
            base_kwargs,
            env,
            pool,
            qf,
            policy,
            plotter=None,
            policy_lr=1E-3,
            qf_lr=1E-3,
            value_n_particles=16,
            td_target_update_interval=1,
            kernel_fn=adaptive_isotropic_gaussian_kernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            discount=0.99,
            reward_scale=1,
            use_saved_qf=False,
            use_saved_policy=False,
            save_full_state=False,
            train_qf=True,
            train_policy=True,

            ## TRPO
            policies,
            baselines,
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
            obs_indx=1,  # made it 1 since originally Blocks was using 1
            **kwargs
    ):
        """
        Args:
            base_kwargs (dict): Dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.
            env (`rllab.Env`): rllab environment object.
            pool (`PoolBase`): Replay buffer to add gathered samples to.
            qf (`NNQFunction`): Q-function approximator.
            policy: (`rllab.NNPolicy`): A policy function approximator.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            qf_lr (`float`): Learning rate used for the Q-function approximator.
            value_n_particles (`int`): The number of action samples used for
                estimating the value of next state.
            td_target_update_interval (`int`): How often the target network is
                updated to match the current Q-function.
            kernel_fn (function object): A function object that represents
                a kernel function.
            kernel_n_particles (`int`): Total number of particles per state
                used in SVGD updates.
            kernel_update_ratio ('float'): The ratio of SVGD particles used for
                the computation of the inner/outer empirical expectation.
            discount ('float'): Discount factor.
            reward_scale ('float'): A factor that scales the raw rewards.
                Useful for adjusting the temperature of the optimal Boltzmann
                distribution.
            use_saved_qf ('boolean'): If true, use the initial parameters provided
                in the Q-function instead of reinitializing.
            use_saved_policy ('boolean'): If true, use the initial parameters provided
                in the policy instead of reinitializing.
            save_full_state ('boolean'): If true, saves the full algorithm
                state, including the replay buffer.
        """

        #####################################################
        # SQL
        super(SQL, self).__init__(**base_kwargs)

        self.env = env
        self.pool = pool
        self.qf = qf
        self.policy = policy
        self.plotter = plotter

        self._qf_lr = qf_lr
        self._policy_lr = policy_lr
        self._discount = discount
        self._reward_scale = reward_scale

        self._value_n_particles = value_n_particles
        self._qf_target_update_interval = td_target_update_interval

        self._kernel_fn = kernel_fn
        self._kernel_n_particles = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio

        self._save_full_state = save_full_state
        self._train_qf = train_qf
        self._train_policy = train_policy

        self._observation_dim = self.env.observation_space.flat_dim
        self._action_dim = self.env.action_space.flat_dim

        self._create_placeholders()

        self._training_ops = []
        self._target_ops = []

        self._create_td_update()
        self._create_svgd_update()
        self._create_target_ops()

        if use_saved_qf:
            saved_qf_params = qf.get_param_values()
        if use_saved_policy:
            saved_policy_params = policy.get_param_values()

        self._sess = tf_utils.get_default_session()
        self._sess.run(tf.global_variables_initializer())

        if use_saved_qf:
            self.qf.set_param_values(saved_qf_params)
        if use_saved_policy:
            self.policy.set_param_values(saved_policy_params)


        #####################################################
        # TRPO
        self.log_dir = logger.get_snapshot_dir()
        if self.log_dir[-1] != '/':
            self.log_dir += '/'

        self.diagnostics_dir = self.log_dir + 'diagnostics_log/'
        if not os.path.exists(self.diagnostics_dir):
            os.makedirs(self.diagnostics_dir)

        self.diagnostics = {}
        self.diagnostics['iter'] = []
        # self.env = env
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
        print('BatchPolopt: env.action_space.high = ', self.env.action_space.high)
        self.Fxy_max = np.linalg.norm(self.env.action_space.high[2:4], ord=2)
        self.digit_distr_uniform = np.array([1. / 9.] * 9)
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
        for key, val in locals().items(): print(key, ': ', val)
        pf.print_sec0_end()

        if self.use_hide_alg == 0:
            self.myplotter = myPlotter(out_dir=self.log_dir + 'graph_log')
        else:
            self.myplotter = myPlotter(out_dir=self.log_dir + 'graph_log',
                                       graph_names=['xy_time', 'xy_time_test', 'xy_tasklabels'])

    def _create_placeholders(self):
        """Create all necessary placeholders."""

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observations')

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='next_observations')

        self._actions_pl = tf.placeholder(
            tf.float32, shape=[None, self._action_dim], name='actions')

        self._next_actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._action_dim], name='next_actions')

        self._rewards_pl = tf.placeholder(
            tf.float32, shape=[None], name='rewards')

        self._terminals_pl = tf.placeholder(
            tf.float32, shape=[None], name='terminals')

    def _create_td_update(self):
        """Create a minimization operation for Q-function update."""

        with tf.variable_scope('target'):
            # The value of the next state is approximated with uniform samples.
            target_actions = tf.random_uniform(
                (1, self._value_n_particles, self._action_dim), -1, 1)
            q_value_targets = self.qf.output_for(
                observations=self._next_observations_ph[:, None, :],
                actions=target_actions)
            assert_shape(q_value_targets, [None, self._value_n_particles])

        self._q_values = self.qf.output_for(
            self._observations_ph, self._actions_pl, reuse=True)
        assert_shape(self._q_values, [None])

        # Equation 10:
        next_value = tf.reduce_logsumexp(q_value_targets, axis=1)
        assert_shape(next_value, [None])

        # Importance weights add just a constant to the value.
        next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))
        next_value += self._action_dim * np.log(2)

        # \hat Q in Equation 11:
        ys = tf.stop_gradient(self._reward_scale * self._rewards_pl + (
            1 - self._terminals_pl) * self._discount * next_value)
        assert_shape(ys, [None])

        # Equation 11:
        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values)**2)

        if self._train_qf:
            td_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
                loss=bellman_residual, var_list=self.qf.get_params_internal())
            self._training_ops.append(td_train_op)

        self._bellman_residual = bellman_residual

    def _create_svgd_update(self):
        """Create a minimization operation for policy update (SVGD)."""

        actions = self.policy.actions_for(
            observations=self._observations_ph,
            n_action_samples=self._kernel_n_particles,
            reuse=True)
        assert_shape(actions,
                     [None, self._kernel_n_particles, self._action_dim])

        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.
        n_updated_actions = int(
            self._kernel_n_particles * self._kernel_update_ratio)
        n_fixed_actions = self._kernel_n_particles - n_updated_actions

        fixed_actions, updated_actions = tf.split(
            actions, [n_fixed_actions, n_updated_actions], axis=1)
        fixed_actions = tf.stop_gradient(fixed_actions)
        assert_shape(fixed_actions, [None, n_fixed_actions, self._action_dim])
        assert_shape(updated_actions,
                     [None, n_updated_actions, self._action_dim])

        svgd_target_values = self.qf.output_for(
            self._observations_ph[:, None, :], fixed_actions, reuse=True)

        # Target log-density. Q_soft in Equation 13:
        squash_correction = tf.reduce_sum(
            tf.log(1 - fixed_actions**2 + EPS), axis=-1)
        log_p = svgd_target_values + squash_correction

        grad_log_p = tf.gradients(log_p, fixed_actions)[0]
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)
        grad_log_p = tf.stop_gradient(grad_log_p)
        assert_shape(grad_log_p, [None, n_fixed_actions, 1, self._action_dim])

        kernel_dict = self._kernel_fn(xs=fixed_actions, ys=updated_actions)

        # Kernel function in Equation 13:
        kappa = tf.expand_dims(kernel_dict["output"], dim=3)
        assert_shape(kappa, [None, n_fixed_actions, n_updated_actions, 1])

        # Stein Variational Gradient in Equation 13:
        action_gradients = tf.reduce_mean(
            kappa * grad_log_p + kernel_dict["gradient"], reduction_indices=1)
        assert_shape(action_gradients,
                     [None, n_updated_actions, self._action_dim])

        # Propagate the gradient through the policy network (Equation 14).
        gradients = tf.gradients(
            updated_actions,
            self.policy.get_params_internal(),
            grad_ys=action_gradients)

        surrogate_loss = tf.reduce_sum([
            tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self.policy.get_params_internal(), gradients)
        ])

        if self._train_policy:
            optimizer = tf.train.AdamOptimizer(self._policy_lr)
            svgd_training_op = optimizer.minimize(
                loss=-surrogate_loss,
                var_list=self.policy.get_params_internal())
            self._training_ops.append(svgd_training_op)

    def _create_target_ops(self):
        """Create tensorflow operation for updating the target Q-function."""
        if not self._train_qf:
            return

        source_params = self.qf.get_params_internal()
        target_params = self.qf.get_params_internal(scope='target')

        self._target_ops = [
            tf.assign(tgt, src)
            for tgt, src in zip(target_params, source_params)
        ]

    # TODO: do not pass, policy, and pool to `__init__` directly.
    def train(self):
        self._train(self.env, self.policy, self.pool)

    @overrides
    def _init_training(self):
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, iteration, batch):
        """Run the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(batch)
        self._sess.run(self._training_ops, feed_dict)

        if iteration % self._qf_target_update_interval == 0 and self._train_qf:
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        feeds = {
            self._observations_ph: batch['observations'],
            self._actions_pl: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_pl: batch['rewards'],
            self._terminals_pl: batch['terminals'],
        }

        return feeds

    @overrides
    def log_diagnostics(self, batch):
        """Record diagnostic information.

        Records the mean and standard deviation of Q-function and the
        squared Bellman residual of the  s (mean squared Bellman error)
        for a sample batch.

        Also call the `draw` method of the plotter, if plotter is defined.
        """

        feeds = self._get_feed_dict(batch)
        qf, bellman_residual = self._sess.run(
            [self._q_values, self._bellman_residual], feeds)

        logger.record_tabular('qf-avg', np.mean(qf))
        logger.record_tabular('qf-std', np.std(qf))
        logger.record_tabular('mean-sq-bellman-error', bellman_residual)

        self.policy.log_diagnostics(batch)
        if self.plotter:
            self.plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SQL algorithm.

        If `self._save_full_state == True`, returns snapshot including the
        replay buffer. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, and environment instances.
        """

        state = {
            'epoch': epoch,
            'policy': self.policy,
            'qf': self.qf,
            'env': self.env,
        }

        if self._save_full_state:
            state.update({'replay_buffer': self.pool})

        return state

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policies, self.bnn)
        if self.plot:
            plotter.init_plot(self.env, self.policies)

    def init_opt(self):

    def train_brownian(self):
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

        variance_mean = self.policies['hide'].action_variance_default
        variance = self.policies['hide'].action_variance_default

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

            ## Filtering starts according to performance and re-populating starts
            # Automatic adjustment of the update period parameter

            update_now = (itr % self.starts_update_every_itr == 0)
            update_period = self.starts_update_every_itr

            #Saving training samples for variance regression (prediction)
            logger.record_tabular('hide_starts_update_period', update_period)
            logger.record_tabular('hide_starts_update_period_max', self.starts_update_every_itr)
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

                if self.brown_adaptive_variance == 4:
                    variance_diff = self.brown_var_control_coeff * (self.center_reached_ratio - self.center_reached_ratio_max)
                    variance_diff = np.clip(variance_diff, -0.5, 0.5)
                    logger.log('brown: variance change %f' % variance_diff)
                    variance_mean += variance_diff
                    variance_mean = np.clip(variance_mean, a_min=self.brown_var_min, a_max=1.0)

                # Variance sampling: it is really necessary for regression only
                variance = copy.deepcopy(variance_mean)

                # print('!!!!!!!!!!!!!!!+++ brown: variance mean: ', variance, 'dtype', type(variance_mean))
                self.policies['hide'].sample_nearby(itr=itr, success_rate=self.center_reached_ratio, variance=variance)
                logger.log('Re-sampled %d new goals %d old goals' % (len(self.policies['hide'].starts), len(self.policies['hide'].starts_old)))
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

            ## Testing environment
            logger.log('Testing environment ...')
            test_paths = self.test_rollouts_seek()
            self.seek_rewards_brownian(itr=itr, paths=test_paths, prefix='test_', test=True)

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
            logger.log('Showing environment ...')
            self.show_rollouts_seek(itr)

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot(name='seek')
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                          "continue...")

        ## Cleaning up
        self.shutdown_worker()

