import numpy as np
import copy
import time
import os
# import sys

# from rllab.misc import tensor_utils
from multigoal.rllab_utils.misc import utils
# from rllab_utils.misc import tensor_utils as e2e_tensor_utils
# from rllab_utils.misc.glob_config import glob_config
import rllab.misc.logger as logger

import e2eap_training.classif.keras_binary_classifier as nn_classif
import matplotlib.pyplot as plt

## Things to try and compare
# removing vs not removing samples for estimation
# creating independent pool of new samples with resampling from previously provided pool

class brownianAgent(object):
    def __init__(self, env, mode,
                 r_min, r_max,
                 action_variance,
                 start_pool_size, step_limit=1,
                 starts_new_num=200, starts_old_num=100,
                 goal_states=[], starts_old_max_size=100000,
                 multigoal=False, starts_new_select_prob=0.6,
                 N_min=1, N_window=5,
                 use_classifier_sampler=False, classif_label_alg=1, starts_min2classify=2,
                 prob_weight_power=1.0, classif_weight_func=0, obs_indx=0,
                 plot=True, out_dir=None,
                 sampling_temperature=1.0, sampling_adaptive_temperature=None,
                 sampling_t_min=0.1, sampling_t_max=2.0, sampling_func_pow=1.0,
                 sampling_prob_min=0.2, sampling_prob_max=0.95
                 ):
        self.env = env
        self.r_min = r_min
        self.r_max = r_max
        self.sample_pool_size = start_pool_size
        self.mode = mode
        self.step_limit = step_limit
        self.starts_new_select_num = starts_new_num
        self.starts_old_select_num = starts_old_num
        self.goal_states = copy.deepcopy(goal_states)
        self.init_goals = copy.deepcopy(goal_states)
        self.starts_old_max_size = starts_old_max_size
        self.reverse_mode = True
        self._multigoal = multigoal
        self.starts_new_select_prob = starts_new_select_prob

        self.action_variance_default = action_variance
        self.action_variance = action_variance

        self.starts = copy.deepcopy(self.goal_states)
        self.starts_old = copy.deepcopy(self.goal_states)
        # Need val to be a list since for every rollout we have a reward and we calculate average at the end
        self.reset_rewards(val=[1.0])

        self.reset()

        self.brownian_samples_num = 0

        self.plot = plot
        self.itr = 0

        ##################################################
        ## Classifier relevant variables
        self.out_dir = out_dir
        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        # self.hard_easy_border = 0.5
        self.hard_easy_border = copy.deepcopy(self.r_min)
        self.N_min = N_min
        self.N_window = N_window

        self.classif_label_alg = classif_label_alg
        self.starts_min2classify = starts_min2classify

        # Classifier good/bad(rejected) goals. It is used for adaptive sampling of new goals
        self.use_classifier_sampler = use_classifier_sampler
        self.prob_weight_power = prob_weight_power
        if use_classifier_sampler:
            self.task_classifier = nn_classif.kerasBinaryClassifier(
                feat_size=env.observation_space.components[obs_indx].high.size, patience=5)
        self.classif_label_alg = classif_label_alg
        self.classif_w_func = classif_weight_func

        self.sampling_temperature = sampling_temperature
        self.sampling_adaptive_temperature = sampling_adaptive_temperature
        self.sampling_t_min = sampling_t_min
        self.sampling_t_max = sampling_t_max
        self.sampling_func_pow = sampling_func_pow

        self.prob_middle = 0.5
        self.prob_max = sampling_prob_max
        self.prob_min = sampling_prob_min

        self.good_starts_for_classifier = []
        self.good_obs_for_classifier = []
        self.bad_starts_for_classifier = []
        self.bad_obs_for_classifier = []

        self.rejected_starts_vec = []
        self.starts_rejected_total = 0

        self.hard_easy_border_cur = self.hard_easy_border

    def reset(self):
        self.action_prev = np.zeros_like(self.env.action_space.low)

    def select_behavior(self):
        pass

    def sample_one_start(self):
        """
        This function will sample single start from the current start pool
        No removing samples as in the original paper
        :return:
        """
        id = np.random.randint(low=0, high=len(self.starts))
        state = self.starts[id]
        # logger.log("sample one start : from %d starts, choose %d th" % (len(self.starts), id))
        return state, id

    def sample_one_goal(self):
        if self._multigoal:
            # p의 확률로 starts를 고르고 1-p의 확률로 starts_old를 고른다.
            # 고른 pool에서 random하게 한개를 고른다.
            pools = [self.starts_old, self.starts]
            pool_id = np.random.binomial(1, p=self.starts_new_select_prob)
            pool2sample = pools[pool_id]
            id = np.random.randint(low=0, high=len(pool2sample))
            state = pool2sample[id]
            # logger.log("sample one goal : chose %dth from pool%d(total %d)" %(id, pool_id, len(pool2sample)))
        else:
            id = np.random.randint(low=0, high=len(self.init_goals))
            state = self.init_goals[id]
        return state, id

    def reset_rewards(self , val=[]):
        if not isinstance(val, list):
            val = [val]
        self.rewards = [copy.deepcopy(val) for x in range(len(self.starts))]

    def select_starts(self, r_min=None, r_max=None, success_rate=None):
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

        ## Keeping the good starts only
        starts_good = []
        self.good_starts_for_classifier = []
        self.good_obs_for_classifier = []
        self.bad_starts_for_classifier = []
        self.bad_obs_for_classifier = []

        ## Pre-checking: if there is simply not enough hard starts or too many
        # then we adjust percentile of rewards
        starts_finished = copy.deepcopy(self.starts)
        reward_avg_list = []
        reward_avg_legit_list = []
        good_starts_count = 0
        logger.log('Select Starts| N_min:%d' % self.N_min)
        logger.log('Select Starts| N_window:%d' % self.N_window)
        logger.log('Select Starts| hard_easy_border:%f' % self.hard_easy_border)
        logger.log('Select Starts| r_min:%f' % self.r_min)
        logger.log('Select Starts| r_max:%f' % self.r_max)

        # N_min=1, N_window=5, hard_easy_border=0.3
        for id, start in enumerate(self.starts):
            ## Getting labels for the classifier
            N_cur = len(self.rewards[id])
            # N_cur(rewards의 사이즈)가 1보다 크거나 같으면 = 그 point에서 시작한 적이 있으면
            if N_cur >= min(1, self.N_min):
                reward_avg_classif = np.mean(self.rewards[id][-self.N_window:])
                reward_avg_list.append(reward_avg_classif)
                if reward_avg_classif < self.hard_easy_border:
                    good_starts_count+=1
                if 0.0 < reward_avg_classif < 1.0:
                    reward_avg_legit_list.append(reward_avg_classif)
        logger.log('Select Starts| good_starts_count:%d' %good_starts_count)

        good_starts_min_ratio = 0.1
        good_starts_max_ratio = 0.8
        logger.log('Select Starts| use_classifier_sampler:%d' %self.use_classifier_sampler)
        # good_starts의 비율이 good_starts_min_ratio(0.1)보다 작을 경우
        # good_starts의 비율이 good_starts_max_ratio(0.8)보다 클 경우
        if self.use_classifier_sampler:
            if float(good_starts_count) / len(self.starts) < good_starts_min_ratio:
                self.hard_easy_border_cur = np.percentile(reward_avg_list, good_starts_min_ratio * 100)
                logger.log('WARNING: Not enough hard examples: adjusted border %.3f' % self.hard_easy_border_cur)
            elif float(good_starts_count) / len(self.starts) > good_starts_max_ratio:
                self.hard_easy_border_cur = np.percentile(reward_avg_list, good_starts_max_ratio * 100)
                if self.hard_easy_border_cur == 0.0 and len(reward_avg_legit_list) > 0:
                    self.hard_easy_border_cur = np.max(reward_avg_legit_list)
                else:
                    self.hard_easy_border_cur = copy.deepcopy(self.r_min)
                logger.log('WARNING: Too many hard examples: adjusted border %.3f' % self.hard_easy_border_cur)
            else:
                self.hard_easy_border_cur = self.hard_easy_border
        logger.log('Select Starts| hard_easy_border_cur:%f' % self.hard_easy_border_cur)


        total_sample = 0
        starts_rejected = []
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        for id, start in enumerate(self.starts):
            ## Getting labels for the classifier
            N_cur = len(self.rewards[id])
            total_sample += N_cur
            print('#id:', id, ' pose:', start[0][:2],' N_cur:', N_cur,)
            if N_cur >= min(1, self.N_min):
                reward_avg_classif = np.mean(self.rewards[id][-self.N_window:])
                if reward_avg_classif < self.hard_easy_border_cur:
                    self.good_starts_for_classifier.append(start)
                    self.good_obs_for_classifier.append(self.env.env.unwrapped.state2obs(start))
                    # print('\t\treward_avg_classif:', reward_avg_classif,'  good start!')
                else:
                    self.bad_starts_for_classifier.append(start)
                    self.bad_obs_for_classifier.append(self.env.env.unwrapped.state2obs(start))
                    # print('\t\treward_avg_classif:', reward_avg_classif, '  bad start!')
            # 사실 앞부분 다 필요없고 이부분만 사용됨
            if len(self.rewards[id]) > 0:
                reward_avg = np.mean(self.rewards[id])
                if r_min < reward_avg < r_max:
                    starts_good.append(start)
                    print('\t\tavg reward: ', reward_avg, '  starts_good')
                else:
                    starts_rejected.append(start)
                    print('\t\tavg reward: ', reward_avg, '  starts_rejected')


        self.starts_rejected_total += len(starts_rejected)
        self.rejected_starts_vec.append(self.starts_rejected_total)

        self.starts = starts_good

        # Only good starts go to the old goals
        # Here there is no uniqueness traction
        self.starts_old.extend(self.starts)
        self.starts_old = self.starts_old[-self.starts_old_max_size:]

        self.reset_rewards()
        self.reset_rewards()

        # print('*************** starts ****************')
        # print(self.starts)
        # print('************* starts_old **************')
        # print(self.starts_old[0][0])
        # print('***************************************')
        print('total_sample:', total_sample)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return starts_finished, (self.starts_new_select_num + self.starts_old_select_num)

    def is_multigoal(self):
        return self._multigoal

    def get_param_values(self):
        return None

    def set_param_values(self, val):
        pass

    def get_action(self, obs=None):
        action = self.action_prev \
                 + np.random.normal(loc=np.zeros_like(self.env.action_space.low),
                                    scale=self.action_variance)
        return action, {'step_limit': self.step_limit}

    def select_old_starts(self):
        ## Adding good old starts for re-training on them
        starts_old_select_num = min(self.starts_old_select_num, len(self.starts_old))
        starts_old_selected_ids = np.random.choice(len(self.starts_old), starts_old_select_num, replace=False)
        starts_old_selected = [self.starts_old[id] for id in starts_old_selected_ids]
        return starts_old_selected

    def sample_nearby(self, starts_finished=None, starts_new_select_num=None, itr=None, clear_figures=True, success_rate=None, variance=None, animated=False):
        """
        Samples a bunch of new initial conditions around the existing good set of initial conditions
        :param starts_new_select_num: left for compatibility with my version of the alg
        :return:
        """
        # Sometimes after filtering no starts are left to populate from
        # thus it would make sense to re-populate from the old starts instead

        if variance is None:
            self.action_variance = copy.deepcopy(self.action_variance_default)
        else:
            self.action_variance = copy.deepcopy(variance)

        old_starts_added = False
        if len(self.starts) == 0:
            # starts_old_selected = self.select_old_starts()
            starts_old_selected = self.starts_old[-self.starts_old_select_num:]
            self.starts.extend(starts_old_selected)
            old_starts_added = True
        samples_before = len(self.starts)
        logger.log('Sample Nearby | initial starts: %d' % np.array(self.starts).shape[0])

        while len(self.starts) < self.sample_pool_size:
            # 현재 starts에서 랜덤하게 한개의 start_state를 선택
            start_state, id = self.sample_one_start()
            # init_state에서 시작하여 done이 될때까지 brownian motion을 한다(sample one path)
            path = utils.rollout_hide(env=self.env,
                               agents={'hide': self},
                               animated=animated,
                               always_return_paths=True,
                               mode=self.mode, hide_tmax= self.step_limit,
                               init_state=start_state, init_goal=start_state, return_states_as_list=True)
            self.starts.extend(path['states'][1:]) #Excluding the first state since it is already in the pile
        logger.log('Sample Nearby | after rollout_hide, starts: %d' % np.array(self.starts).shape[0])

        # TODO: brownian motion plot

        self.brownian_samples_num += (len(self.starts) - samples_before)
        starts_new_select_num = min(self.starts_new_select_num, len(self.starts))

        # use_classifier_sampler == False
        # starts에서 starts_new_num만큼만 뽑는다.
        starts_new_selected_ids = np.random.choice(len(self.starts), starts_new_select_num, replace=False)
        starts_new_selected = [self.starts[id] for id in starts_new_selected_ids]
        self.starts = starts_new_selected
        logger.log('Sample Nearby | randomly select starts, starts: %d' % np.array(self.starts).shape[0])

        # starts_old에서 starts_old_num만큼만 뽑아서 starts에 더한다.
        # logger.log('Sample Nearby | initial starts_old: %d', np.array(self.starts_old).shape[0])
        if not old_starts_added:
            starts_old_selected = self.select_old_starts()
            self.starts.extend(starts_old_selected)
        logger.log('Sample Nearby | randomly select starts_old, starts_old: %d' % np.array(self.starts_old).shape[0])
        logger.log('Sample Nearby | randomly select starts_old and add to starts, starts: %d' % np.array(self.starts).shape[0])

        # self.rewards 크기를 starts크기에 맞춘다.
        self.reset_rewards()

        return self.brownian_samples_num

    # Not using
    def resample_starts(self, oversampled_starts, starts_new_select_num, itr=None, clear_figures=True, success_rate=None):
        # Calculating precondition
        if self.classif_label_alg == 0:
            # classif_precond = min(len(self.starts_rejected_obs), len(self.starts_old_obs)) >= self.starts_min2classify
            raise NotImplementedError
        elif self.classif_label_alg == 1:
            classif_precond = min(len(self.bad_obs_for_classifier),
                                  len(self.good_obs_for_classifier)) >= self.starts_min2classify
            logger.log('Sampling classifier: Good/Bad starts num: %d/%d' % (
            len(self.good_starts_for_classifier), len(self.bad_starts_for_classifier)))
            logger.log('Sampling classifier precondition %d' % classif_precond)
        else:
            classif_precond = False

        if self.use_classifier_sampler and classif_precond:

            if self.classif_label_alg == 0:
                raise NotImplementedError
                # print('!!!!!!!!!!!!!!!!!!! WARNIGN: Old classifier labeling alg is used')
                # # Balancing samples
                # good_balanced, bad_balanced, good_id, bad_id = self.balance_samples(
                #     self.starts_old_obs[-self.classif_bufsize:],
                #     self.starts_rejected_obs[-self.classif_bufsize:])
                #
                # starts_old_buf = self.starts_old[-self.classif_bufsize:]
                # starts_rej_buf = self.starts_rejected[-self.classif_bufsize:]

            elif self.classif_label_alg == 1:
                # Balancing samples
                good_balanced, bad_balanced, good_id, bad_id = self.balance_samples(self.good_obs_for_classifier,
                                                                                    self.bad_obs_for_classifier)

                starts_old_buf = self.good_starts_for_classifier
                starts_rej_buf = self.bad_starts_for_classifier

            # Training a classifier
            train_obs = copy.deepcopy(good_balanced)
            train_obs += bad_balanced

            train_starts = [starts_old_buf[id] for id in good_id]
            train_starts += [starts_rej_buf[id] for id in bad_id]

            train_labels = []
            train_labels += [1] * len(good_balanced)
            train_labels += [0] * len(bad_balanced)

            logger.log('Fitting Sampling classifier on good/bad obs: %d / %d ...' %
                       (len(good_balanced), len(bad_balanced)))

            self.task_classifier.fit(train_obs, train_labels)

            # Getting observations for oversampled starts
            oversampled_starts_obs = [self.env.env.unwrapped.state2obs(start) for start in oversampled_starts]

            # Extracting probability of the start being good
            samp_good_prob = self.task_classifier.predict_proba(np.array(oversampled_starts_obs))[:, 1]

            if self.classif_w_func == 0:
                samp_good_prob_adjusted = np.power(samp_good_prob, self.prob_weight_power)
            elif self.classif_w_func == 1:
                samp_good_prob_adjusted = self.entr2(p=samp_good_prob)
            elif self.classif_w_func == 2:
                if self.sampling_adaptive_temperature == 1:
                    if success_rate is not None:
                        # High success rate leads to low temperatures and more biased sampling
                        self.sampling_temperature = (1.0 - success_rate) * (
                        self.sampling_t_max - self.sampling_t_min) + self.sampling_t_min
                    else:
                        print('ERROR: please provide success rate !!!!')
                        raise ValueError
                    logger.log('Sampling temperature: %f' % self.sampling_temperature)
                samp_good_prob_adjusted = self.softmax(samp_good_prob / self.sampling_temperature)

            elif self.classif_w_func == 3:
                if self.sampling_adaptive_temperature == 1:
                    if success_rate is not None:
                        # High success rate leads to low temperatures and more biased sampling
                        self.sampling_temperature = self.get_entr_temperature(success_rate=success_rate,
                                                                              middle=0.6,
                                                                              Tmax=self.sampling_t_max,
                                                                              Tmin=self.sampling_t_min)
                    else:
                        print('ERROR: please provide success rate !!!!')
                        raise ValueError
                    logger.log('Sampling temperature: %f' % self.sampling_temperature)
                samp_good_prob_adjusted = self.softmax(self.entr2(p=samp_good_prob) / self.sampling_temperature)

            if self.classif_w_func == 4:
                # Function regualting peak value
                # If the success rate is high it moves the middle toward hard examples
                # If the success rate is low it moves the middle toward easy examples
                samp_good_prob_adjusted = self.get_weights(prob=samp_good_prob, middle=success_rate, pow=self.sampling_func_pow)

            if self.classif_w_func == 5:
                # Function regualting peak value of probability of samples
                # The middle should be set externally
                assert self.prob_middle is not None
                samp_good_prob_adjusted = self.get_weights(prob=samp_good_prob, middle=self.prob_middle, pow=self.sampling_func_pow)
                samp_good_prob_adjusted = self.softmax(samp_good_prob_adjusted / self.sampling_temperature)

            samp_good_prob_adjusted = samp_good_prob_adjusted + 1e-8
            samp_good_prob_weights = samp_good_prob_adjusted / np.sum(samp_good_prob_adjusted)

            # Sampling according to the confidence of how likely new goals to be good ones
            try:
                starts_new_selected_ids = np.random.choice(len(oversampled_starts), starts_new_select_num,
                                                           replace=False,
                                                           p=samp_good_prob_weights)
            except:
                print('ERROR: Weights/Probabilities: ', samp_good_prob_weights, samp_good_prob)
                raise ValueError
            starts_new_selected = [oversampled_starts[id] for id in starts_new_selected_ids]
            starts_new_selected_conf = [samp_good_prob[id] for id in starts_new_selected_ids]

            # Plotting
            if self.plot:
                if itr is None:
                    itr = self.itr

                train_samp_conf = self.task_classifier.predict_proba(np.array(train_obs))
                self.plot_starts(starts=train_starts, rewards=train_samp_conf,
                                 img_name='sampclassif_train_xy_conf_itr%03d' % itr, fig_id=20, env=self.env,
                                 out_dir=self.out_dir, clear=clear_figures)
                self.plot_starts(starts=train_starts, rewards=train_labels,
                                 img_name='sampclassif_train_xy_labels_itr%03d' % itr, fig_id=21, env=self.env,
                                 out_dir=self.out_dir, clear=clear_figures)
                self.plot_starts(starts=oversampled_starts, rewards=samp_good_prob_weights / np.max(samp_good_prob_weights),
                                 img_name='sampclassif_oversampled_weights_itr%03d' % itr, fig_id=22, env=self.env,
                                 out_dir=self.out_dir, clear=clear_figures)
                self.plot_starts(starts=starts_new_selected, rewards=starts_new_selected_conf,
                                 img_name='sampclassif_selected_xy_conf_itr%03d' % itr, fig_id=23, env=self.env,
                                 out_dir=self.out_dir, clear=clear_figures)

        else:
            # Otherwise sample uniformly
            starts_new_selected_ids = np.random.choice(len(oversampled_starts), starts_new_select_num, replace=False)
            starts_new_selected = [oversampled_starts[id] for id in starts_new_selected_ids]

        return starts_new_selected

    def balance_samples(self, good, bad):
        min_samples = min(len(good), len(bad))
        good_id = np.random.choice(len(good), min_samples, replace=False)
        bad_id = np.random.choice(len(bad), min_samples, replace=False)
        good_balanced = [good[id] for id in good_id]
        bad_balanced = [bad[id] for id in bad_id]
        return good_balanced, bad_balanced, good_id, bad_id

    @staticmethod
    def get_entr_temperature(success_rate, pow=1., middle=0.6, Tmax=1.0, Tmin=0.1):
        middle = min(middle, 1.0)
        middle = max(middle, 0.0)
        norm = np.ones_like(success_rate)
        norm[success_rate < middle] = middle
        norm[success_rate >= middle] = 1.0 - middle
        weights = (-np.abs(((success_rate - middle) / norm) ** pow) + 1.0) * Tmax
        weights = np.clip(weights, a_min=Tmin, a_max=Tmax)
        return weights

    def log2(self, x):
        return np.log(x) / np.log(2)

    def entr2(self, p):
        p = np.clip(p, 0.0001, 0.9999)
        return -p * self.log2(p) - (1.0 - p) * self.log2(1.0 - p)

    @staticmethod
    def get_weights(prob, pow=1., middle=0.5):
        """
        The function focuses on hard or easy examples based on probability
        :param pow:
        :param middle:
        :return:
        """
        middle = min(middle, 1.0)
        middle = max(middle, 0.0)
        norm = np.ones_like(prob)
        norm[prob < middle] = 1.0
        norm[prob >= middle] = -1.0
        weights = -np.abs(((prob - middle) / norm) ** pow) + 1.0
        return weights

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

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0)