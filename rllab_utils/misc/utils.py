import time
import copy

import numpy as np
from rllab.misc import tensor_utils
from rllab_utils.misc import tensor_utils as e2e_tensor_utils
from rllab_utils.misc.glob_config import glob_config

def sample_multidim(array, samp_num):
    np.random.seed()
    samp_num = int(samp_num)
    indices_all = np.arange(0,array.shape[0])
    # print('sample_multidim: Indices all = ', indices_all, 'Sample num = ', samp_num)
    indices = np.random.choice(indices_all, size=samp_num, replace=False)
    array_sampled = array[indices, :]
    return array_sampled, indices

def img2grey(img):
    """
    Given numpy array convert to grey scale
    :param img: (np.array) numpy array of image
    :return: (np.array) grey scaled image
    """
    return np.expand_dims((0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8),
                          axis=2)


def rollout_hide_seek(env, agents,
                      max_path_length=np.inf,
                      animated=False, speedup=1, always_return_paths=False, mode=None, hide_tmax=None):
    # animated = True
    ## HIDE AGENT
    #Reset the model configuration
    env.reset()
    obs = env.env.env.reload_model()
    last_goal = env.env.unwrapped.get_all_pose()

    # print('-----------------------------------------------------')
    # print('goal hide: ', env.env.env.goal, 'obs:', obs)

    # if animated:
    #     print('rollout: HIDE')
        # print('Frame skip = ', env.env.unwrapped.frame_skip)
        # frame_skip_prev = env.env.unwrapped.frame_skip
        # env.env.unwrapped.frame_skip = 20

    hide_observations = []
    hide_actions = []
    hide_rewards = []
    hide_agent_infos = []
    hide_env_infos = []

    # Hide is capable of stopping so let's set stop if available
    # WARNING: It is important to do all this stuff after reset, since
    # blocks dependent stuff could be reset from config file as well
    if mode is not None:
        if mode == 'seek_force_only':
            env.env.env.use_stop = True
            env.env.env.add_mnist_reward(False)
            env.env.env.use_mnist_stop_criteria(False)
        elif mode == 'reach_center_and_stop':
            env.env.env.use_stop = True
            env.env.env.use_distance2center_stop_criteria = False
            prev_set_limit = env.env.unwrapped.step_limit
            if hide_tmax is not None:
                env.env.unwrapped.step_limit = hide_tmax
            # print('rollout: hide step_limit = ', env.env.unwrapped.step_limit)

    agents['hide'].reset()
    hide_path_length = 0

    if animated:
        env.render()
    # print('rollout: HIDE')
    while hide_path_length < max_path_length:
        a, agent_info = agents['hide'].get_action(obs)
        # print('hide action: ', a)
        if animated:
            env.render()
        obs_next, r, d, env_info = env.step(a)
        hide_observations.append(obs)
        hide_rewards.append(r)
        hide_actions.append(env.action_space.flatten(a))
        hide_agent_infos.append(agent_info)
        hide_env_infos.append(env_info)
        hide_path_length += 1
        last_pose = env.env.unwrapped.get_all_pose()
        # last_goal = copy.deepcopy(env.env.env.goal)
        obs = obs_next
        # print('hide obs: ', obs_next)
        # time.sleep(0.5)
        # if r > 0:
        #     print('!!!!!!!!!!!!!! r:', r, 'stop crit: ', env.env.unwrapped.use_distance2center_stop_criteria)
        if d:
            break
            # print('step hide')
    # print('-------------------------')
    # print('goal hide last: ', env.env.env.goal, 'obs:', obs[1])


    hide_paths = dict(
        observations=e2e_tensor_utils.stack_tensor_list(hide_observations),
        actions=tensor_utils.stack_tensor_list(hide_actions),
        rewards=tensor_utils.stack_tensor_list(hide_rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(hide_agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(hide_env_infos),
    )


    if animated:
        time.sleep(1)

    ##############################################
    # SEEK AGENT
    # print('last obs: ', obs[1])
    if env.spec.id[:6] != 'Blocks' or env.spec.id[:12] == 'BlocksSimple':
        # Avoiding randomization for blocks env
        env.reset() #must do reset for reacher otherwise it feaks out
    obs = env.env.env.reload_model(pose=last_pose, goal=last_goal)
    # print('goal seek: ', env.env.env.goal, 'obs:', obs)

    # print('Timelen max = ', env.env.unwrapped.step_limit)
    # print('Prev limit = ', prev_set_limit)
    if animated:
        print('rollout: SEEK')
        # env.env.unwrapped.frame_skip = 10

    # print('rollout: SEEK')
    if mode is not None:
        if mode == 'seek_force_only':
            env.env.env.use_stop = False
            env.env.env.add_mnist_reward(True)
            env.env.env.use_mnist_stop_criteria(True)
        elif mode == 'reach_center_and_stop':
            env.env.env.use_stop = False
            env.env.env.use_distance2center_stop_criteria = True
            if hide_tmax is not None:
                env.env.unwrapped.step_limit = prev_set_limit

    seek_observations = []
    seek_actions = []
    seek_rewards = []
    seek_agent_infos = []
    seek_env_infos = []

    # obs = env.reset()
    agents['seek'].reset()
    seek_path_length = 0

    if animated:
        env.render()
    while seek_path_length < max_path_length:
        # if seek_path_length < 2: print('seek obs: ', obs)
        a, agent_info = agents['seek'].get_action(obs)
        if animated:
            # print('Seek obs: ', obs, 'action:', a)
            # print('action:', a)
            env.render()
        obs_next, r, d, env_info = env.step(a)
        seek_observations.append(obs)
        seek_rewards.append(r)
        seek_actions.append(env.action_space.flatten(a))
        seek_agent_infos.append(agent_info)
        seek_env_infos.append(env_info)
        seek_path_length += 1

        if d:
            # print('break ...')
            break
        obs = obs_next
        # print('step seek')

    # if animated:
    #     env.env.unwrapped.frame_skip = frame_skip_prev

    if animated and not always_return_paths:
        return

    seek_paths = dict(
        observations=e2e_tensor_utils.stack_tensor_list(seek_observations),
        actions=tensor_utils.stack_tensor_list(seek_actions),
        rewards=tensor_utils.stack_tensor_list(seek_rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(seek_agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(seek_env_infos),
    )

    hide_paths['actions'] = hide_paths['actions'].astype(glob_config.dtype)
    seek_paths['actions'] = seek_paths['actions'].astype(glob_config.dtype)
    hide_paths['rewards'] = hide_paths['rewards'].astype(glob_config.dtype)
    seek_paths['rewards'] = seek_paths['rewards'].astype(glob_config.dtype)

    return {'hide':hide_paths, 'seek':seek_paths}


def rollout_hide(env, agents,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False,
            mode=None, hide_tmax=None, init_state=None, init_goal=None,
            return_states_as_list=False):

    ## HIDE AGENT
    # animated = True
    # Reset the model configuration
    # print('Init goal: ', init_goal)
    if env.spec.id[:6] == 'Blocks':
        env.reset()
        obs = env.env.env.reload_model(pose=init_state, goal=init_goal)
    else:
        env.reset()
        obs = env.env.env.reload_model(pose=init_state, goal=init_goal)

    # time.sleep(1)
    # if animated:
    #     print('rollout: HIDE')
    #     frame_skip_prev = env.env.unwrapped.frame_skip
    #     env.env.unwrapped.frame_skip = 20

    hide_observations = []
    hide_states = []
    hide_actions = []
    hide_rewards = []
    hide_agent_infos = []
    hide_env_infos = []

    # Hide is capable of stopping so let's set stop if available
    # WARNING: It is important to do all this stuff after reset, since
    # blocks dependent stuff could be reset from config file as well
    if mode is not None:
        if mode == 'seek_force_only':
            env.env.env.use_stop = True
            env.env.env.add_mnist_reward(False)
            env.env.env.use_mnist_stop_criteria(False)
        elif mode == 'reach_center_and_stop':
            env.env.env.use_stop = True
            env.env.env.use_distance2center_stop_criteria = False
            prev_set_limit = env.env.unwrapped.step_limit
            if hide_tmax is not None:
                env.env.unwrapped.step_limit = hide_tmax

    agents['hide'].reset()
    hide_path_length = 0

    if animated:
        env.render()

    while hide_path_length < max_path_length:
        a, agent_info = agents['hide'].get_action(obs)
        if animated:
            env.render()

        # need to do it before the step, to match states to observations in the vector
        hide_states.append(env.env.unwrapped.get_all_pose())

        obs_next, r, d, env_info = env.step(a)
        # print('action:', a)

        hide_observations.append(obs)
        hide_rewards.append(r)
        hide_actions.append(env.action_space.flatten(a))
        hide_agent_infos.append(agent_info)
        hide_env_infos.append(env_info)
        hide_path_length += 1
        obs = obs_next
        if d:
            print('Hide | path_length:', hide_path_length)
            break

    if mode is not None:
        if mode == 'seek_force_only':
            env.env.env.use_stop = False
            env.env.env.add_mnist_reward(True)
            env.env.env.use_mnist_stop_criteria(True)
        elif mode == 'reach_center_and_stop':
            env.env.env.use_stop = False
            env.env.env.use_distance2center_stop_criteria = True
            if hide_tmax is not None:
                env.env.unwrapped.step_limit = prev_set_limit

    if not return_states_as_list:
        hide_states = tensor_utils.stack_tensor_list(hide_states)

    hide_paths = dict(
        observations=e2e_tensor_utils.stack_tensor_list(hide_observations),
        actions=tensor_utils.stack_tensor_list(hide_actions),
        rewards=tensor_utils.stack_tensor_list(hide_rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(hide_agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(hide_env_infos),
        states=hide_states,
    )
    # print('Episode done:', hide_path_length)
    return hide_paths


def rollout_seek(env, agents,
                      max_path_length=np.inf,
                      animated=False, speedup=1, always_return_paths=False, mode=None):
    ##############################################
    # SEEK AGENT
    # env.env.unwrapped.reload_model(pose=last_pose)

    seek_observations = []
    seek_actions = []
    seek_rewards = []
    seek_agent_infos = []
    seek_env_infos = []

    if mode == 'mnist_stop':
        env.env.env.use_stop = False
        env.env.env.use_mnist_reward(True)
        env.env.env.use_mnist_stop_criteria(True)
    else:
        env.env.env.use_stop = False

    obs = env.reset()
    agents['seek'].reset()
    seek_path_length = 0
    # print('obs: ', obs[1])

    if animated:
        env.render()
    while seek_path_length < max_path_length:
        a, agent_info = agents['seek'].get_action(obs)
        if animated:
            env.render()
        obs_next, r, d, env_info = env.step(a)
        seek_observations.append(obs)
        seek_rewards.append(r)
        seek_actions.append(env.action_space.flatten(a))
        seek_agent_infos.append(agent_info)
        seek_env_infos.append(env_info)
        seek_path_length += 1
        obs = obs_next
        if d:
            break
            print('SEEK Test | path_length:', seek_path_length)
    if animated and not always_return_paths:
        return

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


def rollout_debug(env, agents,
                      max_path_length=np.inf,
                      animated=False, speedup=1, always_return_paths=False):
    ##############################################
    # SEEK AGENT
    # env.env.unwrapped.reload_model(pose=last_pose)

    animated = True
    always_return_paths = True

    seek_observations = []
    seek_actions = []
    seek_rewards = []
    seek_agent_infos = []
    seek_env_infos = []

    env.env.env.use_stop = False
    env.env.env.use_mnist_reward(True)
    env.env.env.use_mnist_stop_criteria(True)

    obs = env.reset()
    agents['seek'].reset()
    seek_path_length = 0

    if animated:
        env.render()
    while seek_path_length < max_path_length:
        a, agent_info = agents['seek'].get_action(obs)
        if animated:
            env.render()
        obs_next, r, d, env_info = env.step(a)
        seek_observations.append(obs)
        seek_rewards.append(r)
        seek_actions.append(env.action_space.flatten(a))
        seek_agent_infos.append(agent_info)
        seek_env_infos.append(env_info)
        seek_path_length += 1
        obs = obs_next

        print('Distance = ', env_info['act_min_dist'], ' Max_dist = ', env_info['act_dist_max'])
        time.sleep(0.5)
        if d:
            break
            # print('step seek')
    if animated and not always_return_paths:
        return

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


def rollout(env, agents,
            max_path_length=np.inf,
            animated=False, speedup=1, always_return_paths=False):
    ##############################################
    # SEEK AGENT
    seek_observations = []
    seek_actions = []
    seek_rewards = []
    seek_agent_infos = []
    seek_env_infos = []

    obs = env.reset()
    agents['seek'].reset()
    seek_path_length = 0

    if animated:
        env.render()
    while seek_path_length < max_path_length:
        # print('rollout: obs shape = ', obs[0].shape)
        a, agent_info = agents['seek'].get_action(obs)
        if animated:
            env.render()
        obs_next, r, d, env_info = env.step(a)
        seek_observations.append(obs)
        seek_rewards.append(r)
        seek_actions.append(env.action_space.flatten(a))
        seek_agent_infos.append(agent_info)
        seek_env_infos.append(env_info)
        seek_path_length += 1
        obs = obs_next

        if d:
            break

    if animated and not always_return_paths:
        return

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



def rollout_brownian(env, agents,
                      max_path_length=np.inf,
                      animated=False, speedup=1, always_return_paths=False, mode=None, hide_tmax=None):

    ##############################################
    ## HIDE AGENT
    env.reset()
    # 현재 agent['hide'].starts에서 random하게 하나를 뽑는다.
    start_pose, start_pose_id = agents['hide'].sample_one_start()
    start_pose = np.array(start_pose)
    # 현재 agent['hide'].starts에서 p의 확률로 agent['hide'].starts_old에서 1-p의 확률로 random하게 하나를 뽑는다.
    goal, goal_id = agents['hide'].sample_one_goal()
    obs = env.env.env.reload_model(pose=start_pose, goal=goal)
    print("++++++++++++++++++++++++++++++++++++")
    # print('start_pose: ', start_pose)
    # print('goal: ', goal)
    # print('obs: ', obs)
    print('start_pose:', start_pose[0][0:2], '     goal:', goal[0][0:2])
    print('start_pose:', np.array(obs[0][0:2]) * 2.4, '     goal:',np.array(obs[0][-2:]) * 2.4)

    ##############################################
    ## SEEK AGENT
    # print('rollout: Student')
    if animated:
        env.render()
        # env.env.unwrapped.frame_skip = 10

    if mode is not None:
        if mode == 'seek_force_only':
            env.env.env.use_stop = False
            env.env.env.add_mnist_reward(True)
            env.env.env.use_mnist_stop_criteria(True)
        elif mode == 'reach_center_and_stop':
            env.env.env.use_stop = False
            env.env.env.use_distance2center_stop_criteria = True

    seek_observations = []
    seek_actions = []
    seek_rewards = []
    seek_agent_infos = []
    seek_env_infos = []

    agents['seek'].reset()
    seek_path_length = 0

    if animated:
        env.render()
    while seek_path_length < max_path_length:
        # if seek_path_length < 2: print('seek obs: ', obs)
        a, agent_info = agents['seek'].get_action(obs)
        print('action:',a)
        if animated:
            env.render()
        obs_next, r, d, env_info = env.step(a)
        seek_observations.append(obs)
        seek_rewards.append(r)
        seek_actions.append(env.action_space.flatten(a))
        seek_agent_infos.append(agent_info)
        seek_env_infos.append(env_info)
        seek_path_length += 1

        if d:
            print('SEEK| path_length:', len(seek_rewards))
            break
        obs = obs_next
        # print('step seek')

    ## Here we assigning if the goal was reached for a particular goal
    step_limit = env.env.unwrapped.step_limit
    goal_reached = float(seek_path_length < step_limit)

    # starts값에 대한 reward값 저장
    if agents['hide'].reverse_mode:
        agents['hide'].rewards[start_pose_id].append(goal_reached)
    else:
        agents['hide'].rewards[goal_id].append(goal_reached)

    if animated and not always_return_paths:
        return

    seek_paths = dict(
        observations=e2e_tensor_utils.stack_tensor_list(seek_observations),
        actions=tensor_utils.stack_tensor_list(seek_actions),
        rewards=tensor_utils.stack_tensor_list(seek_rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(seek_agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(seek_env_infos),
    )

    seek_paths['actions'] = seek_paths['actions'].astype(glob_config.dtype)
    seek_paths['rewards'] = seek_paths['rewards'].astype(glob_config.dtype)

    return {'seek': seek_paths}
