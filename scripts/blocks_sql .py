import argparse

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.misc.instrument import VariantGenerator

import multigoal.env_blocks.blocks_simple as bsmp
from multigoal.softqlearning.misc.instrument import run_sql_experiment
from multigoal.softqlearning.algorithms import SQL
from multigoal.softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from multigoal.softqlearning.misc.utils import timestamp
from multigoal.softqlearning.replay_buffers import SimpleReplayBuffer
from multigoal.softqlearning.value_functions import NNQFunction
from multigoal.softqlearning.policies import StochasticNNPolicy
from multigoal.softqlearning.environments import GymEnv
from multigoal.softqlearning.misc.sampler import SimpleSampler

import multigoal.env_utils.env_wrappers as env_wrap
import multigoal.rllab_utils.envs.gym_env as gym_env
import multigoal.rllab_utils.envs.globals as glob

SHARED_PARAMS = {
    'seed': [1, 2, 3],
    'policy_lr': 3E-4,
    'qf_lr': 3E-4,
    'discount': 0.99,
    'layer_size': 128,
    'batch_size': 128,
    'max_pool_size': 1E6,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'kernel_particles': 16,
    'kernel_update_ratio': 0.5,
    'value_n_particles': 16,
    'td_target_update_interval': 1000,
    'snapshot_mode': 'last',
    'snapshot_gap': 100,
}


ENV_PARAMS = {
    'swimmer': {  # 2 DoF
        'prefix': 'swimmer',
        'env_name': 'swimmer-rllab',
        'max_path_length': 1000,
        'n_epochs': 500,
        'reward_scale': 30,
    },
    'hopper': {  # 3 DoF
        'prefix': 'hopper',
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 2000,
        'reward_scale': 30,
    },
    'half-cheetah': {  # 6 DoF
        'prefix': 'half-cheetah',
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 30,
        'max_pool_size': 1E7,
    },
    'walker': {  # 6 DoF
        'prefix': 'walker',
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 5000,
        'reward_scale': 10,
    },
    'ant': {  # 8 DoF
        'prefix': 'ant',
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 300,
    },
    'ant-rllab': {  # 8 DoF
        'prefix': 'ant-rllab',
        'env_name': 'ant-rllab',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': [1, 3, 10, 30, 100, 300]
    },
    'humanoid': {  # 21 DoF
        'seed': [11, 12, 13, 14, 15],
        'prefix': 'humanoid',
        'env_name': 'humanoid-rllab',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'reward_scale': 100,
    },
    'BlocksSimpleXYQ-v0':{
        'seed': [1234],
        'prefix': 'blocks-xyq',
        'env_name': 'BlocksSimpleXYQ-v0',
        'max_path_length': 100,
        'n_epochs': 10,
        'reward_scale': 2, #####
        'blocks_multigoal': True,
        'timelen_max': 100,
        'blocks_simple_xml': 'blocks_simple_maze1.xml'
    },
}
DEFAULT_ENV = 'BlocksSimpleXYQ-v0'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', type=str, choices=AVAILABLE_ENVS, default=DEFAULT_ENV)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    return args


def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = SHARED_PARAMS
    params.update(env_params)

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg


def run_experiment(variant):
    if variant['env_name'] == 'humanoid-rllab':
        env = normalize(HumanoidEnv())
    elif variant['env_name'] == 'swimmer-rllab':
        env = normalize(SwimmerEnv())
    elif variant['env_name'] == 'ant-rllab':
        env = normalize(AntEnv())
    elif variant['env_name'] == 'BlocksSimpleXYQ-v0':
        target = [-1.0, 0.0]
        env = bsmp.BlocksSimpleXYQ(multi_goal=variant['blocks_multigoal'],
                                   time_limit=variant['max_path_length'],
                                   env_config=variant['blocks_simple_xml'],
                                   goal=target)
        env = env_wrap.obsTupleWrap(env, add_action_to_obs=False)
        env = gym_env.GymEnv(env,
                             video_schedule=glob.video_scheduler.video_schedule,
                             log_dir=".")
    else:
        env = normalize(GymEnv(variant['env_name']))

    pool = SimpleReplayBuffer(
        env=env, max_replay_buffer_size=variant['max_pool_size'])

    sampler = SimpleSampler(
        max_path_length=variant['max_path_length'],
        min_pool_size=variant['max_path_length'],
        batch_size=variant['batch_size'])

    base_kwargs = dict(
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        sampler=sampler)

    M = variant['layer_size']
    qf = NNQFunction(env=env, hidden_layer_sizes=(M, M))

    policy = StochasticNNPolicy(env=env, hidden_layer_sizes=(M, M))

    algorithm = SQL(
        base_kwargs=base_kwargs,
        env=env,
        pool=pool,
        qf=qf,
        policy=policy,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=variant['kernel_particles'],
        kernel_update_ratio=variant['kernel_update_ratio'],
        value_n_particles=variant['value_n_particles'],
        td_target_update_interval=variant['td_target_update_interval'],
        qf_lr=variant['qf_lr'],
        policy_lr=variant['policy_lr'],
        discount=variant['discount'],
        reward_scale=variant['reward_scale'],
        save_full_state=False)

    algorithm.train()


def launch_experiments(variant_generator, args):
    variants = variant_generator.variants()
    for i, variant in enumerate(variants):
        print('Launching {} experiments.'.format(len(variants)))
        full_experiment_name = variant['prefix']
        full_experiment_name += '-' + args.exp_name + '-' + str(i).zfill(2)

        run_sql_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=variant['prefix'] + '/' + args.exp_name,
            exp_name=full_experiment_name,
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=True)


def main():
    args = parse_args()
    variant_generator = get_variants(args)
    launch_experiments(variant_generator, args)


if __name__ == '__main__':
    main()