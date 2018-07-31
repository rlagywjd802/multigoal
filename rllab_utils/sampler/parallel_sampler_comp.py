# from rllab.sampler.utils import rollout
import numpy as np
from rllab.misc import ext
from rllab.misc import logger
from rllab.sampler.stateful_pool import singleton_pool
from rllab_utils.misc import tensor_utils
from multigoal.rllab_utils.misc.utils import rollout_hide_seek, rollout_seek, rollout_debug, rollout, rollout_brownian


def _worker_init(G, id):
    import os
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    G.worker_id = id


def initialize(n_parallel):
    singleton_pool.initialize(n_parallel)
    singleton_pool.run_each(
        _worker_init, [(id,) for id in range(singleton_pool.n_parallel)])


def _worker_populate_task(G, env, policies, dynamics):
    G.env = env
    G.policies = policies
    G.dynamics = dynamics


def populate_task(env, policies, dynamics):
    logger.log("Populating workers...")
    singleton_pool.run_each(
        _worker_populate_task,
        [(env, policies, dynamics)] * singleton_pool.n_parallel
    )
    logger.log("Populated")


def _worker_set_seed(_, seed):
    ext.set_seed(seed)


def set_seed(seed):
    singleton_pool.run_each(
        _worker_set_seed,
        [(seed + i,) for i in range(singleton_pool.n_parallel)]
    )


def _worker_set_policy_params(G, params):
    for key in params.keys():
        G.policies[key].set_param_values(params[key])


def _worker_set_dynamics_params(G, params):
    if params is not None:
        for key in params.keys():
            G.dynamics[key].set_param_values(params[key])

# Dynamics:
# - dynamics makes no modifications to original rewards. It merely computes KL divergence and adds it to paths
# - if you would like to compute KL obs_mean and std should be subtracted from corresponding observations in the tuple
def _worker_collect_one_path(G, max_path_length, itr, normalize_reward,
                             reward_mean, reward_std, kl_batch_size, n_itr_update, use_replay_pool,
                             obs_mean, obs_std, act_mean, act_std, second_order_update, use_hide=True, use_hide_alg=0, mode=None, show_rollout_chance=0, hide_tmax=None):
    # Show rollout
    # animated = (np.random.uniform() < show_rollout_chance)
    animated = True

    # Path rollout.
    if use_hide is None:
        path = rollout(G.env, G.policies, max_path_length, animated=animated, always_return_paths=True)
    elif use_hide:
        if use_hide_alg == 0:
            path = rollout_hide_seek(G.env, G.policies, max_path_length,
                                    mode=mode, animated=animated, always_return_paths=True, hide_tmax=hide_tmax)
        elif use_hide_alg == 1 or use_hide_alg == 2:
            path = rollout_brownian(G.env, G.policies, max_path_length,
                                    mode=mode, animated=animated,  always_return_paths=True, hide_tmax=hide_tmax)
        else:
            raise ValueError('SAMPLER: Wrong use_hide_alg: %s' % use_hide_alg)
    else:
        path = rollout_seek(G.env, G.policies, max_path_length, animated=animated, always_return_paths=True)

    # Computing intrinsic rewards.
    ## NEEDS fixing to use Tuple observations and 2 agents (hide/seek)
    if itr > 0 and (G.dynamics is not None):
        # Iterate over all paths and compute intrinsic reward by updating the
        # model on each observation, calculating the KL divergence of the new
        # params to the old ones, and undoing this operation.
        obs = (path['observations'] - obs_mean) / (obs_std + 1e-8)
        act = (path['actions'] - act_mean) / (act_std + 1e-8)
        rew = path['rewards']
        # inputs = (o,a), target = o'
        obs_nxt = np.vstack([obs[1:]])
        _inputs_obs = obs[:-1]
        _inputs_act = act[:-1]
        _targets = obs_nxt
        # KL vector assumes same shape as reward.
        kl = np.zeros(rew.shape)
        for j in range(int(np.ceil(obs.shape[0] / float(kl_batch_size)))):

            # Save old params for every update.
            G.dynamics.save_old_params()

            start = j * kl_batch_size
            end = np.minimum(
                (j + 1) * kl_batch_size, obs.shape[0] - 1)

            if second_order_update:
                # We do a line search over the best step sizes using
                # step_size * invH * grad
                #                 best_loss_value = np.inf
                for step_size in [0.01]:
                    G.dynamics.save_old_params()
                    loss_value = G.dynamics.train_update_fn(
                         _inputs_obs[start:end], _inputs_act[start:end], _targets[start:end], step_size)
                    kl_div = np.clip(loss_value, 0, 1000)
                    # If using replay pool, undo updates.
                    if use_replay_pool:
                        G.dynamics.reset_to_old_params()
            else:
                # Update model weights based on current minibatch.
                for _ in range(n_itr_update):
                    G.dynamics.train_update_fn(
                        _inputs_obs[start:end], _inputs_act[start:end], _targets[start:end])
                # Calculate current minibatch KL.
                kl_div = np.clip(
                    float(G.dynamics.f_kl_div_closed_form()), 0, 1000)

            for k in range(start, end):
                kl[k] = kl_div
            # If using replay pool, undo updates.
            if use_replay_pool:
                G.dynamics.reset_to_old_params()

        # Last element in KL vector needs to be replaced by second last one
        # because the actual last observation has no next observation.
        kl[-1] = kl[-2]

        # Stuff it in path
        path['KL'] = kl
        # ----------------------------

    if use_hide:
        if use_hide_alg == 0:
            samples_num = len(path['hide']["rewards"]) + len(path['seek']["rewards"])
        else:
            samples_num = len(path['seek']["rewards"])
    else:
        samples_num = len(path['seek']["rewards"])

    return path, samples_num


def sample_paths(
        policy_params,
        dynamics_params,
        max_samples,
        max_path_length=np.inf,
        itr=None,
        normalize_reward=None,
        reward_mean=None,
        reward_std=None,
        kl_batch_size=None,
        n_itr_update=None,
        use_replay_pool=None,
        obs_mean=None,
        obs_std=None,
        act_mean=None,
        act_std=None,
        second_order_update=None,
        use_hide=True,
        use_hide_alg='my',
        show_rollout_chance=0.01,
        hide_tmax=10,
        mode=None
):
    """
    :param policy_params: (dict) parameters for policies. This will be updated on each worker process
    :param max_samples: desired maximum number of samples to be collected. The actual number of collected samples
    might be greater since all trajectories will be rolled out either until termination or until max_path_length is
    reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :return: a list of collected paths
    """
    singleton_pool.run_each(
        _worker_set_policy_params,
        [(policy_params,)] * singleton_pool.n_parallel
    )

    # Set dynamics params.
    # --------------------
    singleton_pool.run_each(
        _worker_set_dynamics_params,
        [(dynamics_params,)] * singleton_pool.n_parallel
    )
    # --------------------
    # max_samples(params['batch_size']을 넘지 않으면서 random한 수의 path를 rollout
    return singleton_pool.run_collect(
        _worker_collect_one_path,
        threshold=max_samples,
        args=(max_path_length, itr, normalize_reward, reward_mean,
              reward_std, kl_batch_size, n_itr_update, use_replay_pool,
              obs_mean, obs_std, act_mean, act_std, second_order_update,
              use_hide, use_hide_alg, mode, show_rollout_chance, hide_tmax),
        show_prog_bar=False
    )


def truncate_paths(paths, max_samples):
    """
    Truncate the list of paths so that the total number of samples is exactly equal to max_samples. This is done by
    removing extra paths at the end of the list, and make the last path shorter if necessary
    :param paths: a list of paths
    :param max_samples: the absolute maximum number of samples
    :return: a list of paths, truncated so that the number of samples adds up to max-samples
    """
    # chop samples collected by extra paths
    # make a copy
    paths = list(paths)
    total_n_samples = sum(len(path["rewards"]) for path in paths)
    while len(paths) > 0 and total_n_samples - len(paths[-1]["rewards"]) >= max_samples:
        total_n_samples -= len(paths.pop(-1)["rewards"])
    if len(paths) > 0:
        last_path = paths.pop(-1)
        truncated_last_path = dict()
        truncated_len = len(
            last_path["rewards"]) - (total_n_samples - max_samples)
        for k, v in last_path.iteritems():
            if k in ["observations", "actions", "rewards", "rewards_orig"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_list(
                    v, truncated_len)
            elif k in ["env_infos", "agent_infos"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_dict(
                    v, truncated_len)
            else:
                raise NotImplementedError
        paths.append(truncated_last_path)
    return paths
