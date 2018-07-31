import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import rllab.plotter as plotter

## env - BlocksSimpleXYQ
import multigoal.env_blocks.blocks_simple as bsmp
## agent -- BrownianAgent
from multigoal.rllab_utils.algos.brownian_agent import brownianAgent
## policy -- GausianMultiObsPolicy
import multigoal.rllab_utils.policies.gaussian_multiobs_policy as gaus_pol
## baseline -- GaussianConvBaseline
from multigoal.rllab_utils.baselines.baselines import GaussianConvBaseline
## optimization -- TRPO
from multigoal.rllab_utils.algos.trpo_comp import TRPO
## parallel sampler
from multigoal.sampler import parallel_sampler_comp as parallel_sampler

# ===========================
#   Plotter
# ===========================
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


# ===========================
#   BatchPolopt
# ===========================
class BatchPolopt():
    """
        Base class for batch sampling-based policy optimization methods.
        This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """
    def __init__(self, env, policies, env_test, env_mode='seek_force_only', use_hide=None, use_hide_alg=0):
        self.env = env
        self.env_test = env_test
        self.mode = mode
        self.use_hide = use_hide
        self.use_hide_alg = use_hide_alg
        if self.use_hide_alg == 0:
            self.myplotter = myPlotter(out_dir= self.log_dir + 'graph_log')
        else:
            self.myplotter = myPlotter(out_dir=self.log_dir + 'graph_log', graph_names=['xy_time', 'xy_time_test', 'xy_tasklabels'])

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policies, self.bnn)
        if self.plot:
            plotter.init_plot(self.env, self.policies)

    def init_opt(self):
        ## NPO
        pass

    def obtain_samples(self, itr):
        cur_params = {}
        for key in self.policies.keys():
            cur_params[key] = self.policies[key].get_param_values()

        cur_dynamics_params = None
        if self.rew_bnn_use:
            cur_dynamics_params = self.bnn.get_param_values()

        reward_mean = None
        reward_std = None
        if self.rew_bnn_use and self.normalize_reward:
            # Compute running mean/std.
            reward_mean = np.mean(np.asarray(self._reward_mean))
            reward_std = np.mean(np.asarray(self._reward_std))

        # Mean/std obs/act based on replay pool.
        obs_mean, obs_std, act_mean, act_std = None, None, None, None
        if self.rew_bnn_use:
            obs_mean, obs_std, act_mean, act_std = self.pool.mean_obs_act()

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
            use_hide_alg=self.use_hide_alg,
            mode=self.mode,
            show_rollout_chance=self.show_rollout_chance,
            hide_tmax=self.hide_tmax
        )
        if self.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(
                paths, self.batch_size)
            return paths_truncated

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
                self.rew_seek__mnistANDtargetloc_coeff * (
                paths['seek'][i]['env_infos']['rew_mnistANDtargetloc'].astype(dtype=np.float32))

            paths['seek'][i]['rewards'] += paths['seek'][i]['reward_components']['rew_mnistANDtargetloc']
            rew_mnistANDtargetloc_eplst.append(
                np.sum(paths['seek'][i]['reward_components']['rew_mnistANDtargetloc']))

            #######################################################################
            ## Same as prev, but for the final moment only
            paths['seek'][i]['reward_components']['rew_final_mnistANDtargetloc'] = \
                self.rew_seek__final_mnistANDtargetloc_coeff * (
                float(paths['seek'][i]['env_infos']['rew_mnistANDtargetloc'][-1]))

            paths['seek'][i]['rewards'][-1] += paths['seek'][i]['reward_components']['rew_final_mnistANDtargetloc']
            rew_final_mnistANDtargetloc_eplst.append(
                paths['seek'][i]['reward_components']['rew_final_mnistANDtargetloc'])

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
            self.check_vec_size(paths['seek'][i]['rewards'], paths['seek'][i]['reward_components']['rew_act_dist'],
                                'rew_act_dist')

            rew_action_dist_eplst.append(np.sum(paths['seek'][i]['reward_components']['rew_act_dist']))
            action_dist_eplst.append(np.mean(paths['seek'][i]['env_infos']['act_min_dist']))

            #######################################################################
            ## Penalty for Applying Force (taking action)
            force_ratio = paths['seek'][i]['env_infos']['act_force_norm']
            paths['seek'][i]['reward_components'][
                'rew_act_force_norm'] = self.rew_seek__action_coeff * force_ratio ** 2

            paths['seek'][i]['rewards'] += paths['seek'][i]['reward_components']['rew_act_force_norm']
            self.check_vec_size(paths['seek'][i]['rewards'],
                                paths['seek'][i]['reward_components']['rew_act_force_norm'], 'rew_act_force_norm')

            rew_action_force_eplst.append(np.sum(paths['seek'][i]['reward_components']['rew_act_force_norm']))
            action_force_eplst.append(np.mean(paths['seek'][i]['env_infos']['act_force']))

            #######################################################################
            ## Penalty for time
            paths['seek'][i]['rewards'] += self.rew_seek__time_step

            #######################################################################
            ## Diagnostics for the episode
            x_init_eplst.append(paths['seek'][i]['env_infos']['xyz_prev_normalized'][0][0])
            y_init_eplst.append(paths['seek'][i]['env_infos']['xyz_prev_normalized'][0][1])

            # Labels assign according to if task is solvable or not.
            # Typically tasks consistently solved within a time budget are solvable
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
        logger.record_tabular(prefix + 'seek_rew_mnist', np.mean(rew_mnist_eplst))
        logger.record_tabular(prefix + 'seek_rew_center_reached', np.mean(rew_center_reached_eplst))
        logger.record_tabular(prefix + 'seek_center_reached', np.mean(center_reached_eplst))
        logger.record_tabular(prefix + 'seek_rew_dist2target', np.mean(rew_dist2center_eplst))
        logger.record_tabular(prefix + 'seek_rew_action_force', np.mean(rew_action_force_eplst))
        logger.record_tabular(prefix + 'seek_rew_action_distance', np.mean(rew_action_dist_eplst))
        logger.record_tabular(prefix + 'seek_action_force', np.mean(action_force_eplst))
        logger.record_tabular(prefix + 'seek_action_distance', np.mean(action_dist_eplst))
        logger.record_tabular(prefix + 'seek_rew_mnistANDtargetloc', np.mean(rew_mnistANDtargetloc_eplst))
        logger.record_tabular(prefix + 'seek_rew_final_mnistANDtargetloc_eplst',
                              np.mean(rew_final_mnistANDtargetloc_eplst))

        if self.env.spec.id[:7] == 'Reacher':
            xlim = [-0.22, 0.22]
            ylim = [-0.22, 0.22]
        else:
            xlim = [-1, 1]
            ylim = [-1, 1]

        print('-------------------->test:', test)
        if test:
            self.myplotter.plot_xy_time(x=x_init_eplst,
                                        y=y_init_eplst,
                                        t=path_lengths,
                                        t_max=self.timelen_max,
                                        img_name='xy_time_test_itr' + str(itr), name='xy_time_test',
                                        xlim=xlim, ylim=ylim)
        else:
            self.myplotter.plot_xy_time(x=x_init_eplst,
                                        y=y_init_eplst,
                                        t=path_lengths,
                                        t_max=self.timelen_max,
                                        img_name='xy_time_itr' + str(itr),
                                        xlim=xlim, ylim=ylim)

            self.myplotter.plot_xy_reward(x=x_init_eplst,
                                          y=y_init_eplst,
                                          r=start_state_taskclassif_labels,
                                          img_name='xy_tasklabels_itr' + str(itr),
                                          name='xy_tasklabels',
                                          r_min=0., r_max=1.,
                                          xlim=xlim, ylim=ylim)

        return paths

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

        ########################################################################
        ## Reward normalization (can be applied for every agent)
        reward_main_norm_pathsums = []
        if self.normalize_reward:
            logger.log('Normalizing rewards ...')
            # Update reward mean/std Q.
            rewards = []
            for i in range(len(paths)):
                rewards.append(paths[i]['rewards'])
            rewards_flat = np.hstack(rewards)
            self._reward_mean.append(np.mean(rewards_flat))
            self._reward_std.append(np.std(rewards_flat))

            # Normalize rewards.
            reward_mean = np.mean(np.asarray(self._reward_mean))
            reward_std = np.mean(np.asarray(self._reward_std))
            for i in range(len(paths)):
                paths[i]['rewards'] = (paths[i]['rewards'] - reward_mean) / (reward_std + 1e-8)

        ########################################################################
        ## Dynamics related processing (exploration)
        # !!! Requires works
        if self.rew_bnn_use:
            if itr > 0:
                kls = []
                for i in range(len(paths)):
                    kls.append(paths[i]['KL'])

                kls_flat = np.hstack(kls)

                logger.record_tabular(name + '_Expl_MeanKL', np.mean(kls_flat))
                logger.record_tabular(name + '_Expl_StdKL', np.std(kls_flat))
                logger.record_tabular(name + '_Expl_MinKL', np.min(kls_flat))
                logger.record_tabular(name + '_Expl_MaxKL', np.max(kls_flat))

                # Perform normalization of the intrinsic rewards.
                if self.use_kl_ratio:
                    if self.use_kl_ratio_q:
                        # Update kl Q
                        self.kl_previous.append(np.median(np.hstack(kls)))
                        previous_mean_kl = np.mean(np.asarray(self.kl_previous))
                        for i in range(len(kls)):
                            kls[i] = kls[i] / previous_mean_kl

                ## INTRINSIC REWARDS
                reward_bnn_pathsums = []
                for i in range(len(paths)):
                    paths[i]['reward_bnn'] = self.eta * kls[i]
                    paths[i]['rewards'] = paths[i]['rewards'] + self.eta * kls[i]
                    reward_bnn_pathsums.append(np.sum(paths[i]['reward_bnn']))

                reward_bnn_avg = np.mean(reward_bnn_pathsums)
                logger.record_tabular(name + '_RewAvg_Dyn', reward_bnn_avg)

                # Discount eta
                self.eta *= self.eta_discount

            else:
                logger.record_tabular(name + '_Expl_MeanKL', 0.)
                logger.record_tabular(name + '_Expl_StdKL', 0.)
                logger.record_tabular(name + '_Expl_MinKL', 0.)
                logger.record_tabular(name + '_Expl_MaxKL', 0.)
                logger.record_tabular(name + '_RewAvg_Dyn', 0.)

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

            ent = np.mean(policy.distribution.entropy(agent_infos))

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

        ## POLICY is recurrent
        ##!!! Requires work
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = np.array(
                [tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])

            if self.center_adv:
                raw_adv = np.concatenate(
                    [path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [
                    (path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.array(
                [tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = np.array(
                [tensor_utils.pad_tensor(a, max_path_length) for a in actions])

            rewards = [path["rewards"] for path in paths]
            rewards = np.array(
                [tensor_utils.pad_tensor(r, max_path_length) for r in rewards])

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(
                    p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(
                    p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = np.array(
                [tensor_utils.pad_tensor(v, max_path_length) for v in valids])

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baseline_values),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        logger.record_tabular(name + '_AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular(name + '_AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular(name + '_ExplainedVariance', ev)
        logger.record_tabular(name + '_NumTrajs', len(paths))
        logger.record_tabular(name + '_Entropy', ent)
        logger.record_tabular(name + '_Perplexity', np.exp(ent))
        logger.record_tabular(name + '_StdReturn', np.std(undiscounted_returns))
        logger.record_tabular(name + '_MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular(name + '_MinReturn', np.min(undiscounted_returns))

        return samples_data

    def show_rollouts_seek(self, iter):
        print('BatchPolOpt: render_every_iterations = ', glob.video_scheduler.render_every_iterations, 'iter:', iter)
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

    def update_plot(self, name):
        if self.plot:
            plotter.update_plot(self.policies[name], self.max_path_length)

    def train_brownian(self):

        ## Initializing the parallel sampler
        self.start_worker()

        ## Initilizing optimization
        self.init_opt(policy_name='seek')

        # For saving diagnostics
        self.episode_rewards = []
        self.episode_lengths = []

        print('Re-sampling new start positions ...')
        self.policies['hide'].sample_nearby()
        print('%d new goals populated' % len(self.policies['hide'].starts))

        variance_mean = self.policies['hide'].action_variance_default
        variance = self.policies['hide'].action_variance_default

        for itr in range(self.n_itr):
            print("itr #%d | " %itr)

            ## Obtaining samples with parallel workers
            paths = self.obtain_samples(itr)    ################# 1
            # Re-organizing dimensions:
            paths = self.ld2dl(paths)

            ## Sample processing:
            paths = self.seek_rewards_browninan(itr=itr, paths=paths)   ################# 2
            seek_samples_data = self.process_samples(itr, paths['seek'],
                                                     baseline=self.baselines['seek'],
                                                     policy=self.policies['seek'],
                                                     name='seek')   ################# 3
            samples_data = {'seek': seek_samples_data}

            ## Fitting the baseline
            print("Fitting baseline seek...")
            self.baselines['seek'].fit(paths['seek'])
            print("Fitted")

            ## Optimizing Policies
            log_seek_opt_vars = self.optimize_policy(itr, samples_data['seek'], policy_name='seek')    ################# 4


            ## Filtering starts according to performance and repopulating starts

            # Automatic adjustment of the update period parameter
            # Set the update period according to the accracy --> Set to Fixed step K
            ## For every K steps, 'hide' resamples
            if itr%20 == 0:
                self.myplotter.plot_goal_rewards(goals=self.policies['hide'].starts,
                                                 rewards=self.policies['hide'].rewards,
                                                 img_name='goal_rewards_itr%03d' % itr,
                                                 scale=start_scale,
                                                 clear=True, env=self.env)
                print('Filtering start positions ...')
                self.policies['hide'].select_starts(success_rate=self.center_reached_ratio)
                print('%d goals selected' % len(self.policies['hide'].starts))
                self.myplotter.plot_goals(goals=self.policies['hide'].starts, color=[0,1,0], clear=True, env=self.env)

                print('Re-sampling new start positions ...')

                ## Adapt Variance
                variance_diff = self.brown_var_control_coeff * (
                            self.center_reached_ratio - self.center_reached_ratio_max)
                variance_diff = np.clip(variance_diff, -0.5, 0.5)
                print('brown: variance change %f' % variance_diff)
                variance_mean += variance_diff
                variance_mean = np.clip(variance_mean, a_min=self.brown_var_min, a_max=1.0)
                variance = copy.deepcopy(variance_mean)

                self.policies['hide'].sample_nearby(itr=itr, success_rate=self.center_reached_ratio, variance=variance)
                print('Re-sampled %d new goals %d old goals' % (len(self.policies['hide'].starts), len(self.policies['hide'].starts_old)))
                self.myplotter.plot_goals(goals=self.policies['hide'].starts, color=[1,0,0], scale=start_scale, env=self.env)
                self.myplotter.plot_goals(goals=self.policies['hide'].starts_old, color=[0,0,1], scale=start_scale, image_name='goals', env=self.env)

            ## Testing environment
            print('Testing environment ...')
            test_paths = self.test_rollouts_seek()  ################ 5
            self.seek_rewards_brownian(itr=itr, paths=test_paths, prefix='test_', test=True)

            ## Rendering few rollouts to show results and record video at the end of every few iterations
            print('Showing environment ...')
            self.show_rollouts_seek(itr)  ################ 6

            ## Plotting the hell out of it
            if self.plot:
                self.update_plot(name='seek')  ################ 7


# ===========================
#   main
# ===========================
def main():

    ##############################################
    ## Parameters
    blocks_multigoal = True
    timelen_max = 100
    blocks_simple_xml = "blocks_simple_maze1.xml"
    target = [-1.0, 0.0]
    # use_hide=Ture, use_hide_alg=1
    # train_mode = 0  # 'multiple_static_goals':False, 'brown_multiple_static_goals_mode':0
    # mode = 'reach_center_and_stop'
    # pickled_mode = False
    r_min = 0.3
    r_max = 0.9
    task_classifier = 'gp'
    print('Blocks target: ', target)

    ##############################################
    ## Environment
    env = bsmp.BlocksSimpleXYQ(multi_goal=blocks_multigoal,
                               time_limit=timelen_max,
                               env_config=blocks_simple_xml,
                               goal=target)

    ##############################################
    ## Policies
    goal_init = env.pose2goal(target)
    print('goal_init: ', goal_init)
    hide_policy = brownianAgent(env=env, r_min=r_min, r_max=r_max)
    seek_policy = gaus_pol.GaussianMultiObsPolicy()
    policies = {'hide': hide_policy, 'seek': seek_policy}

    ##############################################
    ## Baselines
    hide_baseline = None
    seek_baseline = GaussianConvBaseline(env,)
    baselines = {'hide': hide_baseline, 'seek': seek_baseline}

    ##############################################
    ## Optimization : TRPO <- NPO <- BatchPolopt
    # optimizer <- ConjugateGradientOptimizer
    algo = TRPO(env=env, policies=policies, baselines=baselines)
    print('Running the experiment ...')
    algo.train_brownian()

if __name__ == '__main__':
    main()
