#!/usr/bin/env python
import argparse
import sys
import os
import datetime
import itertools
import os.path as osp
import uuid
import copy

import numpy as np

import dateutil.tz
import yaml

import gym
import gym_blocks

from rllab.misc.ext import set_seed
from rllab.misc.instrument import stub, run_experiment_lite
import rllab.misc.logger as logger
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer

import multigoal.rllab_utils.envs.gym_env as gym_env
import multigoal.rllab_utils.policies.gaussian_multiobs_policy as gaus_pol
from multigoal.rllab_utils.baselines.baselines import GaussianConvBaseline
import multigoal.rllab_utils.envs.globals as glob
from multigoal.rllab_utils.misc.glob_config import glob_config

import multigoal.env_blocks.blocks_simple as bsmp
import multigoal.env_utils.env_wrappers as env_wrap
import multigoal.rllab_utils.config.default_params as def_par
import e2eap_training.classif.keras_binary_classifier as nn_classif
from multigoal.rllab_utils.algos.brownian_agent import brownianAgent

########################################################################
## COMMENTS:
# - multi/single goal env is a bit confusing. So both agent and env can be single/multi goal
# when env is singlegoal it pretty much ignores what hide agent is providing and sets the same goal all the time
# on the other hand we have to prevent that from happening for the hide agent that trains to spread from some init seed state (repeat hide agent)
# thus for the repeat hide we have to provide multigoal env for training, and the agent itself should guarantee that it uses single goal for the
# reverse counterpart (env stops taking care of that), but for the test env everything should be as it is

########################################################################
## TODO
# - segmentation fault while saving a snapshot
# - pickling not working for BlocksEnv
# - can not run more than 1 thread if monitoring (might be problem because I am rendering)

########################################################################
## ARGUMENTS
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("params_def", help='yaml file with default settings of parameters')
parser.add_argument("logdir", default='_results_temp/trpo_hide_seek_temp', help='Directory to log into')
parser.add_argument("--seeds", '-s', default='1234', help='list of seeds to use separated by comma (or a single seed w/o comma)')
parser.add_argument("--param_name", '-p', help='hyperparameter name')
parser.add_argument("--param_val", '-pv', help='hyperparameter values')
args = parser.parse_args()

########################################################################
## PARAMETERS (non grid)
# Loading parameters not specified in the arguments
print('Reading parameter file %s ...' % args.params_def)
params = def_par.trpo_hide_seek_default_params()
yaml_stream = open(args.params_def, 'r')
params_new = yaml.load(yaml_stream)
params.update(params_new)
print('###############################################################')
print('### PARAMETERS')
print(params)

########################################################################
## Aux Parameters
pickled_mode = False
snapshot_mode = 'none' #Options: all, last, none
tabular_log_file = 'progress.csv'
text_log_file = 'debug.log'
params_log_file = 'params.json'
params_all_log_file = 'params.yaml'
plot = False
log_tabular_only = False

n_parallel = 1
glob_config.dtype = np.float32

glob.video_scheduler.render_every_iterations = params['render_every_iterations']
glob.video_scheduler.render_rollouts_num = params['render_rollouts_num']


rand_id = str(uuid.uuid4())[:5]
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
exp_name = 'experiment_%s_%s' % (timestamp, rand_id)


###########################################################################
# This function makes everything a stub object.
# It should be called in pickled mode
if pickled_mode:
    stub(globals())

###########################################################################
## GRID PARAMETERS
# seeds = list(range(2))
mdp = params['env_name']

seeds = [int(x) for x in args.seeds.split(',')]

if args.param_name is None:
    # Just running with  default settings
    param_name = None
    param_values = []
else:
    param_name = [x for x in args.param_name.split(',')]
    if args.param_val is None:
        raise ValueError('No values provided for param %s' % param_name)
    else:
        try:
            param_values = [[float(y) for y in x.split(',')] for x in args.param_val.split(',,')]
        except:
            param_values = [[y for y in x.split(',')] for x in args.param_val.split(',,')]

############################################################################
## POSTPROCESSING OF PARAMETERS
if args.logdir[-1] != '/':
    args.logdir += '/'

if params['hide_baseline_net_params']['optimizer'] == 'LbfgsOptimizer':
    params['hide_baseline_net_params']['optimizer'] = LbfgsOptimizer(max_opt_itr=params['hide_baseline_net_params']['max_opt_itr'])
    params['hide_baseline_net_params'].pop('max_opt_itr', None)
else:
    raise ValueError('Unknown optimizer: %s', params['hide_baseline_net_params']['optimizer'])

if params['seek_baseline_net_params']['optimizer'] == 'LbfgsOptimizer':
    params['seek_baseline_net_params']['optimizer'] = LbfgsOptimizer(max_opt_itr=params['seek_baseline_net_params']['max_opt_itr'])
    params['seek_baseline_net_params'].pop('max_opt_itr', None)
else:
    raise ValueError('Unknown optimizer: %s', params['seek_baseline_net_params']['optimizer'])


## All possible combinations of hyperparameters
param_values.append(seeds)
param_cart_product = itertools.product(*param_values)

# param_cart_product = itertools.product(
#     param_values, seeds
# )


# for param_val_cur, seed in param_cart_product:
for param_tuple in param_cart_product:
    seed = param_tuple[-1]
    param_tuple = param_tuple[:-1]
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('PARAMETERS TUPLE: ', param_name, param_tuple, ' SEED: ', seed)

    if param_name is not None:
        log_dir = args.logdir
        if len(param_name) == 1:
            params[param_name[0]] = param_tuple[0]
            log_dir += (param_name[0] + '/' + str(param_tuple[0]) + '/')
        else:
            for par_i, par in enumerate(param_name):
                params[par] = param_tuple[par_i]
                if par_i == 0:
                    log_dir += (par + '_' + str(param_tuple[par_i]))
                else:
                    log_dir += ('__' + par + '_' + str(param_tuple[par_i]))
            log_dir += '/'

        log_dir += (str(seed) + '/')
    else:
        log_dir = args.logdir + str(seed) + '/'
    log_dir_errors = log_dir + 'errors/'
    if not os.path.isdir(log_dir_errors):
        os.makedirs(log_dir_errors)

    params['cmd'] = " ".join(sys.argv)
    params['seed'] = seed

    ###########################################################################
    ## ENVIRONMENT
    if params['multiple_static_goals']:
        assert isinstance(params['target'][0], list)
        if params['brown_multiple_static_goals_mode'] == 0:
            logger.log('WARNING: Training with multiple goals is engaged. Mixing samples mode')
            train_mode = 2
        if params['brown_multiple_static_goals_mode'] == 1:
            logger.log('WARNING: Training with multiple goals is engaged. Swapping agents mode')
            train_mode = 3
    elif params['init_states'] is not None and len(params['init_states']) != 0:
        logger.log('WARNING: Training with multiple init conditions is engaged')
        train_mode = 1
    else:
        train_mode = 0

    if train_mode != 2:
        goal_for_env = params['target']
    elif train_mode == 2 or train_mode == 3:
        goal_for_env = params['target'][0] #Envs just don't know how to treat multiple goals. Agent takes care of it
    else:
        raise NotImplementedError

    ## Blocks task
    if mdp == 'BlocksSimpleXYQ-v0':
        print('Blocks target: ', params['target'])
        multigoal_agent = params['blocks_multigoal']
        if params['use_hide_alg'] == 2 and train_mode != 0:
            logger.log('WARNING: Multi goal env is used for training to ensure that repeat agent can use variety of goals')
            env = bsmp.BlocksSimpleXYQ(multi_goal=True,  #needed for the repeat agent. Reverse has to take care of env being single goaled itself
                                       time_limit=params['timelen_max'],
                                       env_config=params['blocks_simple_xml'],
                                       goal=goal_for_env)

            env_test = bsmp.BlocksSimpleXYQ(multi_goal=params['blocks_multigoal'],
                                            time_limit=params['timelen_max'],
                                            env_config=params['blocks_simple_xml'],
                                            goal=goal_for_env)
        else:
            env = bsmp.BlocksSimpleXYQ(multi_goal=params['blocks_multigoal'],
                                       time_limit=params['timelen_max'],
                                       env_config=params['blocks_simple_xml'],
                                       goal=goal_for_env)
            env_test = env

    else:
        env = gym.make(params['env_name'])
        env_test = gym.make(params['env_name'])


    if env.spec.id[:12] == 'BlocksSimple':
        obs_indx = 0
    elif env.spec.id[:6] == 'Blocks':
        obs_indx = 1
    else:
        obs_indx = 0

    # Applying wrapping
    # use_distance2center_stop_criteria == False here, because we have to change
    # this parameter depending on if it is hide or seek running
    env = env_wrap.obsTupleWrap(env, add_action_to_obs=False)
    env_test = env_wrap.obsTupleWrap(env_test, add_action_to_obs=False)

    # Wrapping gym env
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)

    if params['record_video']:
        env = gym_env.GymEnv(env,
                             video_schedule=glob.video_scheduler.video_schedule,
                             log_dir=log_dir)
        env_test = gym_env.GymEnv(env_test, video_schedule=glob.video_scheduler.video_schedule,
                                  log_dir=log_dir)
    else:
        env = gym_env.GymEnv(env,
                             video_schedule=None,
                             log_dir=log_dir,
                             record_video=False)
        env_test = gym_env.GymEnv(env_test, video_schedule=None,
                                  log_dir=log_dir,
                                  record_video=False)

    ############################################################
    ## POLICIES
    # Hide policy
    brown_log_dir = logger.get_snapshot_dir()
    if brown_log_dir[-1] != '/':
        brown_log_dir += '/'
    brown_log_dir += 'brown_agent_diagnostics/'

    if params['use_hide_alg'] == 1:
        goal_init = env.env.env.pose2goal(params['target'])
        hide_policy = brownianAgent(
            env=env,
            mode=params['mode'],
            r_min=params['r_min'],
            r_max=params['r_max'],
            action_variance=params['brown_act_variance'] * np.ones_like(env.action_space.low),
            start_pool_size=params['start_pool_size'],
            step_limit=params['hide_tmax'],
            starts_new_num=params['starts_new_num'],
            starts_old_num=params['starts_old_num'],
            starts_new_select_prob=params['starts_new_select_prob'],
            goal_states=[goal_init],
            starts_old_max_size=params['starts_old_max_size'],
            multigoal=multigoal_agent,
            use_classifier_sampler=params['brown_use_classif_sampler'],
            classif_label_alg=params['brown_classif_label_alg'],
            prob_weight_power=params['brown_prob_weight_power'],
            classif_weight_func=params['brown_classif_weight_func'],
            out_dir=brown_log_dir,
            sampling_temperature=params['brown_sampling_temperature'],
            sampling_adaptive_temperature=params['brown_sampling_adaptive_temperature'],
            sampling_t_min=params['brown_sampling_t_min'],
            sampling_t_max=params['brown_sampling_t_max'],
            sampling_func_pow=params['brown_sampling_func_pow'],
            sampling_prob_min=params['brown_sampling_prob_min'],
            sampling_prob_max=params['brown_sampling_prob_max'],
        )
    else:
        hide_policy = gaus_pol.GaussianMultiObsPolicy(
            obs_net_params=params['hide_obs_net_params'],
            fusion_net_params=params['hide_fuse_net_params'],
            obs_indx=(obs_indx,),
            action_dims=params['hide_fuse_net_params']['output_dimensions'],
            env=env,
            name='hide_policy',
            use_flat_obs=True
        )

    seek_policy = gaus_pol.GaussianMultiObsPolicy(
        obs_net_params=params['seek_obs_net_params'],
        fusion_net_params=params['seek_fuse_net_params'],
        obs_indx=(obs_indx,),
        action_dims=params['seek_fuse_net_params']['output_dimensions'],
        env=env,
        name='seek_policy',
        use_flat_obs=True
    )
    policies = {'hide': hide_policy, 'seek': seek_policy}


    ############################################################
    ## BASELINES
    if params['use_hide_alg'] == 1 or params['use_hide_alg'] == 2:
        hide_baseline = None
    else:
        hide_baseline = GaussianConvBaseline(
            env,
            obs_indx=obs_indx,
            regressor_args=params['hide_baseline_net_params'],
            name='hide_vf',
            error_file=log_dir_errors + 'hide_vf_errors.txt'
        )

    seek_baseline = GaussianConvBaseline(
        env,
        obs_indx=obs_indx,
        regressor_args=params['seek_baseline_net_params'],
        name='seek_vf',
        error_file=log_dir_errors + 'seek_vf_errors.txt'
    )

    baselines = {'hide': hide_baseline, 'seek': seek_baseline}

    if pickled_mode:
        from multigoal.rllab_utils.algos.trpo_comp import TRPO
    else:
        from multigoal.rllab_utils.algos.trpo_comp_nonserial import TRPO


    ############################################################
    ## TASK CLASSIFIER
    if params['task_classifier'] is None:
        task_classifier = None
    elif params['task_classifier'].lower() == 'gp':
        task_classifier = 'gp'
    elif params['task_classifier'].lower() == 'mlp':
        task_classifier = nn_classif.kerasBinaryClassifier(feat_size=env.observation_space.components[obs_indx].high.size)
    else:
        raise ValueError('ERROR: Unknown type of task classifier!')

    ############################################################
    ## OPTIMIZATION
    algo = TRPO(
        env=env,
        policies=policies,
        baselines=baselines,
        batch_size=params['batch_size'],
        batch_size_uniform=params['batch_size_uniform'],
        brown_uniform_anneal=params['brown_uniform_anneal'],
        test_episodes_num=params['test_episodes_num'],
        whole_paths=True,
        max_path_length=500,
        n_itr=int(params['iterations']),
        step_size=params['trpo_kl_step'],
        snn_n_samples=10,
        subsample_factor=1.0,
        use_replay_pool=True,
        use_kl_ratio=True,
        use_kl_ratio_q=True,
        n_itr_update=1,
        kl_batch_size=1,
        normalize_reward=params['norm_reward'],
        replay_pool_size=params['replay_pool_size'],
        n_updates_per_sample=5000,
        second_order_update=True,
        bnn_params=params['bnn_params'],
        unn_learning_rate=0.0001,
        rew_bnn_use=False,
        env_test=env_test,
        use_hide=params['use_hide'],
        use_hide_alg=params['use_hide_alg'],
        mode=params['mode'],
        rew_hide__search_time_coeff=params['rew_hide__search_time_coeff'],  # 1.
        rew_hide__actcontrol_middle=params['rew_hide__actcontrol_middle'],
        rew_hide__action_coeff=params['rew_hide__action_coeff'],  # -1.
        rew_seek__action_coeff=params['rew_seek__action_coeff'],  # -1.
        rew_hide__digit_correct_coeff=params['rew_hide__digit_correct_coeff'],
        # 1. #make <0 if we want to penalize correct predicitons by seek
        rew_hide__time_step=params['rew_hide__time_step'],  # -0.01 # Just penalty for taking time steps
        rew_hide__act_dist_coeff=params['rew_hide__act_dist_coeff'],
        rew_hide__search_force_coeff=params['rew_hide__search_force_coeff'],
        rew_seek__taskclassif_coeff=params['rew_seek__taskclassif_coeff'],
        rew_seek__final_digit_entropy_coeff=params['rew_seek__final_digit_entropy_coeff'],  # 1.
        rew_seek__final_digit_correct_coeff=params['rew_seek__final_digit_correct_coeff'],  # 1.
        rew_seek__digit_entropy_coeff=params['rew_seek__digit_entropy_coeff'],  # 1.
        rew_seek__digit_correct_coeff=params['rew_seek__digit_correct_coeff'],  # 1.
        rew_seek__time_step=params['rew_seek__time_step'],  # -0.01  # Just penalty for taking time steps
        rew_seek__act_dist_coeff=params['rew_seek__act_dist_coeff'],
        rew_seek__dist2target_coeff=params['rew_seek__dist2target_coeff'],
        rew_seek__center_reached_coeff=params['rew_seek__center_reached_coeff'],
        train_seek_every=params['train_seek_every'],
        show_rollout_chance=params['show_rollout_chance'],
        timelen_max=params['timelen_max'],
        timelen_avg=params['timelen_avg'],
        hide_tmax=params['hide_tmax'],
        rew_hide__search_time_power=params['rew_hide__search_time_power'],
        rew_hide__taskclassif_power=params['rew_hide__taskclassif_power'],
        rew_hide__taskclassif_coeff=params['rew_hide__taskclassif_coeff'],
        taskclassif_pool_size=params['taskclassif_pool_size'],
        taskclassif_balance_all_labels=params['taskclassif_balance_all_labels'],
        adaptive_timelen_avg=params['adaptive_timelen_avg'],
        adaptive_percentile=params['adaptive_percentile'],
        timelen_avg_hist_size=params['timelen_avg_hist_size'],
        rew_hide__taskclassif_middle=params['rew_hide__taskclassif_middle'],
        timelen_reward_fun=params['timelen_reward_fun'],
        taskclassif_use_allpoints=params['taskclassif_use_allpoints'],
        taskclassif_adaptive_middle=params['taskclassif_adaptive_middle'],
        taskclassif_rew_alg=params['taskclassif_rew_alg'],
        hide_stop_improve_after=params['hide_stop_improve_after'],
        obs_indx=obs_indx,
        starts_update_every_itr=params['starts_update_every_itr'],
        starts_adaptive_update_itr=params['starts_adaptive_update_itr'],
        brown_adaptive_variance=params['brown_adaptive_variance'],
        brown_variance_min=params['brown_variance_min'],
        brown_var_control_coeff=params['brown_var_control_coeff'],
        brown_tmax_adaptive=params['brown_tmax_adaptive'],
        brown_t_adaptive=params['brown_sampling_adaptive_temperature'],
        brown_prob_middle_adaptive=params['brown_sampling_prob_middle_adaptive'],
        brown_success_rate_pref=params['brown_success_rate_pref'],
        center_reached_ratio_max=params['brown_center_reached_ratio_max'],
        center_reached_ratio_min=params['brown_center_reached_ratio_min'],
        brown_seed_agent_period=params['brown_seed_agent_period'],
        brown_itr_min=params['brown_itr_min'],
        brown_itr_max=params['brown_itr_max'],
        task_classifier=task_classifier,
        eta=params['eta']
    )



    if pickled_mode:
        run_experiment_lite(
            algo.train_seek(),
            exp_prefix="trpo-expl",
            n_parallel=2,
            snapshot_mode="last",
            seed=seed,
            mode="local",
            script="rllab/run_experiment_lite.py",
        )
    else:
        from sandbox.vime.sampler import parallel_sampler_expl as parallel_sampler
        parallel_sampler.initialize(n_parallel=n_parallel)

        if seed is not None:
            set_seed(seed)
            parallel_sampler.set_seed(seed)

        if plot:
            from rllab.plotter import plotter
            plotter.init_worker()

        tabular_log_file_fullpath = osp.join(log_dir, tabular_log_file)
        text_log_file_fullpath = osp.join(log_dir, text_log_file)
        # params_log_file_fullpath = osp.join(log_dir, params_log_file)
        params_all_log_file_fullpath = osp.join(log_dir, params_all_log_file)

        # logger.log_parameters_lite(params_log_file, args)
        logger.add_text_output(text_log_file_fullpath)
        logger.add_tabular_output(tabular_log_file_fullpath)
        prev_snapshot_dir = logger.get_snapshot_dir()
        prev_mode = logger.get_snapshot_mode()
        logger.set_snapshot_dir(log_dir)
        logger.set_snapshot_mode(snapshot_mode)
        logger.set_log_tabular_only(log_tabular_only)
        logger.push_prefix("[%s] " % exp_name)

        ############################################################
        ## Dumping config
        with open(params_all_log_file_fullpath, 'w') as yaml_file:
            yaml_file.write(yaml.dump(params, default_flow_style=False))

        ############################################################
        ## RUNNING THE EXPERIMENT
        logger.log('Running the experiment ...')
        if params['use_hide']:
            if params['use_hide_alg'] == 1:
                if params['batch_size_uniform'] is not None and params['batch_size_uniform'] > 0:
                    logger.log('WARNING: Training with uniform sampling. Testing is done BEFORE the optimization !!!!')
                    algo.train_brownian_with_uniform()
                else:
                    algo.train_brownian()
            elif params['use_hide_alg'] == 2:
                if train_mode == 0:
                    algo.train_brownian_with_goals()
                elif train_mode == 1:
                    algo.train_brownian_reverse_repeat()
                elif train_mode == 2:
                    algo.train_brownian_multiseed()
                elif train_mode == 3:
                    algo.train_brownian_multiseed_swap_every_update_period()
                else:
                    raise NotImplementedError
            else: #my version of the alg
                algo.train_hide_seek()
        else:
            algo.train_seek()
        logger.log('Experiment finished ...')

        logger.set_snapshot_mode(prev_mode)
        logger.set_snapshot_dir(prev_snapshot_dir)
        logger.remove_tabular_output(tabular_log_file_fullpath)
        logger.remove_text_output(text_log_file_fullpath)
        logger.pop_prefix()

        print('Tabular log file:', tabular_log_file_fullpath)
