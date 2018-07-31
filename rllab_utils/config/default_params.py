#!/usr/bin/env python
import yaml
import sys

def trpo_hide_seek_default_params():
    params = {}

    params['env_name'] = "Blocks-v1"
    params['multiple_static_goals'] = False
    #########################################################
    ## Blocks ENV specific
    params['reward_interval'] = {}
    params['reward_interval']['dist']  = None
    params['reward_interval']['px']    = None
    params['reward_interval']['force'] = None
    params['reward_interval']['mnist'] = [0, None]
    params['env_norm'] = True
    params['classif_framework'] = 'keras' # keras, tf, dummy
    params['classif_snapshot'] = 'classif/keras_classifier.h5' #keras_classifier_oneorient.h5 , keras_classifier.h5
    params['vis_force'] = True #For blocks env only
    params['blocks_yaml_config'] = 'config/blocks_config_hide_seek_table5x.yaml'
    params['blocks_yaml_config_test'] = 'config/blocks_config_seek_table5x.yaml'
    params['blocks_simple_xml'] = 'blocks_simple.xml'
    params['blocks_multigoal'] = False

    #########################################################
    ## Reacher env specific
    params['reacher_denserew'] = False
    params['reacher_multigoal'] = True

    #########################################################
    ## TRAINING PARAMS
    params['batch_size'] = 3000
    params['batch_size_uniform'] = None
    params['brown_uniform_anneal'] = False
    params['iterations'] = 500
    params['train_seek_every'] = 1 #Train seek every how many iterations of hide (teacher)
    params['trpo_kl_step'] = 0.01


    #########################################################
    ## REWARD PARAMETERS
    params['mode'] = 'reach_center_and_stop'
    params['norm_reward'] = False
    # seek_with_digit_action (for learning classifer from scratch as part of action space)
    # seek_force_only (for pretrained classifier)
    params['use_hide'] = True #True - use hide/seek rollouts; False -use seek rollout; None - use standard rollouts
    params['use_hide_alg'] = 0 #0 - my alg, 1 - brownian motion teacher
    params['use_stop_action'] = False
    params['target'] = [0, 0]
    params['init_states'] = []
    params['test_init_states'] = []

    params['timelen_max'] = 100
    params['timelen_avg'] = None #35 in original table5x
    params['hide_tmax'] = 15
    params['timelen_avg_hist_size'] = 50

    params['rew_hide__search_time_coeff'] = 1.  # 1.
    params['rew_hide__search_time_power'] = 1.
    params['adaptive_timelen_avg'] = True
    params['adaptive_percentile'] = True
    params['timelen_reward_fun'] = 'get_timelen_reward_with_median'
    # Options:
    #   get_timelen_reward_with_penalty
    #   get_timelen_reward_with_median
    params['taskclassif_rew_alg'] = 'get_prob_reward'
    # Options:
    # get_prob_reward
    # get_prob_reward_unnorm

    params['rew_hide__taskclassif_middle'] = 0.5
    params['taskclassif_adaptive_middle'] = True
    params['rew_hide__taskclassif_power'] = 1
    params['rew_hide__taskclassif_coeff'] = 1.
    params['task_classifier'] = 'gp'
    params['taskclassif_pool_size'] = 300
    params['taskclassif_use_allpoints'] = True
    params['taskclassif_balance_all_labels'] = False
    params['hide_stop_improve_after'] = None

    params['rew_hide__actcontrol_middle'] = None #action control turned off
    params['rew_hide__action_coeff'] = 0.0 #-0.01 #-0.015  # -1.
    params['rew_seek__action_coeff'] = 0.0 #-0.01 #-0.015  # -1.

    params['rew_hide__search_force_coeff'] = 0. #0.1
    params['rew_hide__digit_entropy_coeff'] = 0.  # 0.2
    params['rew_hide__digit_correct_coeff'] = 0.  # -1. make <0 if we want to penalize not correct predictions by seek
    params['rew_hide__time_step'] = -0.01  # -0.01 Just penalty for taking time steps
    params['rew_hide__act_dist_coeff'] = -0.1

    params['rew_seek__taskclassif_coeff'] = None #This coeff does not matter if rew_hide__taskclassif_coeff is None or == 0
    params['rew_seek__final_digit_entropy_coeff'] = 1. #
    params['rew_seek__digit_entropy_coeff'] = 0.01  #
    params['rew_seek__final_digit_correct_coeff'] = 0. # 1.
    params['rew_seek__digit_correct_coeff'] = 0.0  # 0.01
    params['rew_seek__time_step'] = -0.01  # Just penalty for taking time steps
    params['rew_seek__act_dist_coeff'] = 0.0 #-0.1
    params['rew_seek__dist2target_coeff'] = 0.0 #0.1 for dense rewards
    params['rew_seek__center_reached_coeff'] = 1.


    #########################################################
    ## HIDE POLICY AND BASELINE
    hide_obs_net_params = []
    hide_obs_net_params.append({})

    hide_obs_net_params[0]['conv_filters'] = []
    hide_obs_net_params[0]['conv_filter_sizes'] = []
    hide_obs_net_params[0]['conv_strides'] = []
    hide_obs_net_params[0]['conv_pads'] = []
    hide_obs_net_params[0]['hidden_sizes'] = [32]
    hide_obs_net_params[0]['hidden_nonlinearity'] = 'rectify'
    hide_obs_net_params[0]['output_nonlinearity'] = 'rectify'

    params['hide_obs_net_params'] = hide_obs_net_params

    hide_fuse_net_params= {}
    hide_fuse_net_params['hidden_sizes'] = [32,16]
    hide_fuse_net_params['hidden_nonlinearity'] = 'rectify'
    hide_fuse_net_params['output_nonlinearities'] = 'tanh' # I normalize digit distribution later
    hide_fuse_net_params['output_dimensions'] = None

    params['hide_fuse_net_params'] = hide_fuse_net_params

    ## Baseline network
    hide_baseline_net_params = {}
    hide_baseline_net_params['conv_filters'] = []
    hide_baseline_net_params['conv_filter_sizes'] = []
    hide_baseline_net_params['conv_strides'] = []
    hide_baseline_net_params['conv_pads'] = []
    hide_baseline_net_params['hidden_sizes'] = [32, 32, 16]
    hide_baseline_net_params['batchsize'] = 32
    hide_baseline_net_params['init_std'] = 1.0
    hide_baseline_net_params['step_size'] = 0.01 #KL divergence constraint for each iteration
    hide_baseline_net_params['optimizer'] = 'LbfgsOptimizer'
    hide_baseline_net_params['max_opt_itr'] = 20
    hide_baseline_net_params['use_trust_region'] = False

    params['hide_baseline_net_params'] = hide_baseline_net_params

    #########################################################
    ## Brownian agent parameters
    params['r_min'] = 0.1
    params['r_max'] = 0.9
    params['start_pool_size'] = 1000
    params['starts_new_num']  = 135
    params['starts_old_num']  = 65
    params['starts_old_max_size'] = 10000
    params['starts_update_every_itr'] = 5
    params['starts_adaptive_update_itr'] = None
    params['brown_center_reached_ratio_max'] = 0.7
    params['brown_center_reached_ratio_min'] = 0.1
    params['brown_act_variance'] = 0.5
    params['brown_variance_min'] = 0.1
    params['brown_var_control_coeff'] = 2.0
    params['brown_sample_alg'] = 1 #0 - re-sample only from good starts; 1 - resample from pile of good starts and newely sampled starts
    params['brown_itr_min'] = 1
    params['brown_itr_max'] = 10

    params['brown_sampling_temperature'] = 1.0
    params['brown_sampling_adaptive_temperature'] = None
    params['brown_sampling_prob_middle_adaptive'] = False
    params['brown_sampling_prob_min'] = 0.2
    params['brown_sampling_prob_max'] = 0.95
    params['brown_sampling_t_min'] = 0.1
    params['brown_sampling_t_max'] = 2.0
    params['brown_adaptive_variance'] = None
    params['brown_sampling_func_pow'] = 1.0

    params['starts_new_select_prob'] = 0.6

    ## For multigoal only
    params['brown_N_min'] = 5
    params['brown_N_max'] = 30
    params['brown_N_window'] = 10
    params['brown_oversample_times'] = 5
    params['brown_use_classif_sampler'] = False
    params['brown_prob_weight_power'] = 1.0
    # power for the power fuction of weights (see brown_classif_weight_func)
    params['brown_classif_bufsize'] = None
    params['brown_classif_label_alg'] = 0
    # 0 - old alg, where positive labels are from old starts and negative from rejected
    # 1 - alg where positive labels are from current starts only r_min < r < r_max
    params['brown_classif_weight_func'] = 0
    # 0 - power function, i.e. w = pow(p, n)
    # 1 - entropy function, i.e. w = -p * log2(p) - (1-p)*log2(1-p)
    # 2 - softmax function
    # 3 - softmax(entropy) function

    # mode
    # hide_tmax
    # use_hide_alg == 'brownian'
    # target

    params['brown_tmax_adaptive'] = False
    params['brown_success_rate_pref'] = 0.7

    ## For multi init states only
    params['brown_goal_states_proportions'] = None
    # None - do not apply adjustment of num of starts old/new
    #        i.e whatever starts_new_num/starts_old_num states will be applied to every agent
    #        If proportions are specified than whatever starts_new_num/starts_old_num states will be devided
    params['brown_goal_states_agent_probabilities'] = None
    # None - uniform probabilities
    # unif - uniform as well.

    params['brown_goal_states_prop_same_as_prob'] = True
    # True: brown_goal_states_proportions = copy.deepcopy(brown_goal_states_agent_probabilities)
    # Thus if you want samples to be devided and matched set 'unif' instead of None

    params['brown_multiple_static_goals_mode'] = 0
    # 0 == mixing samples from different state seed agents according to distrib 'brown_goal_states_proportions'
    # 1 == swapping state seed agent every update period according to distribution 'brown_goal_states_proportions'

    params['brown_goal_states_agent_alg'] = None
    # None - probailities are static
    # 1 - probabilities are updated using moving average Q function
    params['brown_goal_states_q_alpha'] = 0.1
    params['brown_goal_states_q_temperature'] = 0.1
    params['brown_seed_agent_period'] = 1

    #########################################################
    ## SEEK POLICY AND BASELINE
    seek_obs_net_params = []
    seek_obs_net_params.append({})

    seek_obs_net_params[0]['conv_filters'] = []
    seek_obs_net_params[0]['conv_filter_sizes'] = []
    seek_obs_net_params[0]['conv_strides'] = []
    seek_obs_net_params[0]['conv_pads'] = []
    seek_obs_net_params[0]['hidden_sizes'] = [32]
    seek_obs_net_params[0]['hidden_nonlinearity'] = 'rectify'
    seek_obs_net_params[0]['output_nonlinearity'] = 'rectify'

    params['seek_obs_net_params'] = seek_obs_net_params

    seek_fuse_net_params= {}
    seek_fuse_net_params['hidden_sizes'] = [32,16]
    seek_fuse_net_params['hidden_nonlinearity'] = 'rectify'
    seek_fuse_net_params['output_nonlinearities'] = 'tanh' # I normalize digit distribution later
    seek_fuse_net_params['output_dimensions'] = None

    params['seek_fuse_net_params'] = seek_fuse_net_params

    ## Baseline network
    seek_baseline_net_params = {}
    seek_baseline_net_params['conv_filters'] = []
    seek_baseline_net_params['conv_filter_sizes'] = []
    seek_baseline_net_params['conv_strides'] = []
    seek_baseline_net_params['conv_pads'] = []
    seek_baseline_net_params['hidden_sizes'] = [32, 32, 16]
    seek_baseline_net_params['batchsize'] = 32
    seek_baseline_net_params['init_std'] = 1.0
    seek_baseline_net_params['step_size'] = 0.01 #KL divergence constraint for each iteration
    seek_baseline_net_params['optimizer'] = 'LbfgsOptimizer'
    seek_baseline_net_params['max_opt_itr'] = 20
    seek_baseline_net_params['use_trust_region'] = False

    params['seek_baseline_net_params'] = seek_baseline_net_params

    ########################################################################
    ## Dynamics network parameters
    ##!!! WORK
    bnn_params = {}

    bnn_params['conv_filters'] = [32,32,32]
    bnn_params['conv_filter_sizes'] = [4,4,4]
    bnn_params['conv_strides'] = [2,2,2]
    bnn_params['conv_pads'] = ['valid','valid','valid']
    # CNN/MLP shared parameters
    bnn_params['hidden_sizes'] = (32, 32)
    bnn_params['hidden_types'] = (1,1,1) #For hidden layers + output layer: Probabilistic layer (1) or deterministic layer (0)
    bnn_params['hidden_nonlinearity'] = 'rectify'
    bnn_params['output_nonlinearity'] = 'linear'

    params['bnn_params'] = bnn_params

    params['replay_pool_size'] = 200000  # For bnn only
    params['eta'] = 0.0001 * 0.25 #Init val: 0.0001 : Reward coefficient for weighting KL divergence reward

    # Original parameters
    # unn_n_hidden = [32] #single layer 32 neurons
    # unn_layers_type = [1, 1] #hidden and output layers both bayesian

    ##########################################################################
    ## AUX params
    params['record_video'] = False
    params['show_rollout_chance'] = 0.02

    params['render_every_iterations'] = 100
    params['render_rollouts_num'] = 3
    params['test_episodes_num'] = 20


    return params


def main(argv):
    params_trpo_hide_seek = trpo_hide_seek_default_params()
    params_all_log_file = 'trpo_seek_reacher__noactpenalty.yaml'
    with open(params_all_log_file, 'w') as yaml_file:
        yaml_file.write(yaml.dump(params_trpo_hide_seek, default_flow_style=False))

if __name__ == '__main__':
    main(sys.argv)
