#!/usr/bin/env python
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from ctypes import byref
import anymarkup

import gym
from gym import error, spaces
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Tuple as gym_tuple

import mujoco_py
# from mujoco_py import glfw
# from mujoco_py.mjlib import mjlib
#
# from e2eap_training.rllab_utils.misc import utils as rollout_utils
# import e2eap_training.env_blocks.blocks_action_wrap as baw
# import rllab_utils.envs.gym_env as gym_env
# import e2eap_training.env_utils.env_wrappers as env_wrap
# from multigoal.rllab_utils.algos import brownian_agent_with_goals

from transforms3d.euler import mat2euler
from transforms3d.quaternions import quat2mat

import gym.envs.registration as gym_reg

## NOTES
# Options to provide a goal:
# - goal location is reset every time
# - goal is manually provided during training through reload function
# - goal is static through all episodes

## TODO
# test brown rollout
# test speed

class BlocksSimple(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, time_limit=100,
                 multi_goal=False, goal_tolerance = None, zero_speed_init=True,
                 tuple_obspace=False, dense_rewards=False,
                 env_config='blocks_simple.xml', goal=[0.0, 0.0], force_duration=10, frame_skip=20, Fmax=450., orient_euler=False,
                 normalize_obs=True):

        """
        Simplified version of the blocks environment - suppose to run faster than the orginal one
        Under assumption of using low dim features only and applying Fxy actions at the COM of a single block on the scene
        Observations = [x,y,z, qw, q0, q1, q2], where q == quaternion
        Actions = [Fx,Fy]
        :param time_limit:
        :param multi_goal:
        :param goal_tolerance: (int or array-like) if None then take tolerance from xml file
        :param zero_speed_init:
        :param tuple_obspace:
        :param dense_rewards:
        :param env_config:
        :param goal:
        :param force_duration:
        :param Fmax:
        """

        self.render_current_step = False

        # Viewer parameters
        self.scene_width = 126
        self.scene_height = 126

        self.Fmax = Fmax

        self._goal = None
        self._goal_static = None

        self.goal_static = goal


        self.time_step = 0
        self.max_episode_steps = time_limit
        self.pos_prev = np.array([0, 0, 0])
        self.max_dist2goal = 0.4
        self.use_static_goal = not multi_goal
        self.goal_feature_size = 2 #this property should go before goal_tolerance assignment
        self.goal_tolerance = goal_tolerance

        self.orient_euler = orient_euler
        if not orient_euler:
            self.orient_default = np.array([1.,0.,0.,0.])
            self.angle_coord_size = 4  # 4 == quaternions are used, 3 == euler is used

            self.get_obs_indx_names()
        else:
            raise NotImplemented



        self.pose_size = 3 + self.angle_coord_size #7d
        self.tuple_obspace = tuple_obspace
        self.force_duration = force_duration


        ## Tolerances on velocities to settle as a part of
        # the logic that waits till the cube stops
        self.Vxy_atol = 1e-2
        self.Vxy_rtol = 1e-2
        self.Vang_atol = 1e-2
        self.Vang_rtol = 1e-2

        self.Vxy_atol = np.array([self.Vxy_atol] * 2)
        self.Vxy_rtol = np.array([self.Vxy_rtol] * 2)
        self.Vang_atol = np.array([self.Vang_atol] * self.angle_coord_size)
        self.Vang_rtol = np.array([self.Vang_rtol] * self.angle_coord_size)

        self.Vxy_null = np.zeros_like(self.Vxy_atol)
        self.Vang_null = np.zeros_like(self.Vang_atol)

        #These introduced for compatibility
        self.use_distance2center_stop_criteria = True
        self.use_mnist_stop_criteria = False
        self.use_mnist_reward = False
        self.max_action = np.array([1., 1.])
        self.zero_speed_init = zero_speed_init
        self.dense_rewards = dense_rewards

        # Reading table parameters directly from the xml
        env_config = os.path.join(os.path.split(os.path.abspath(__file__))[0], env_config)
        params_xml = self.read_xml_parameters(xml_filename=env_config)

        self.space_size = params_xml['space_size']
        self.table_wall_width = params_xml['table_wall_width']
        self.goal_radius = params_xml['goal_radius']
        # self.space_size = np.array([1.0,1.0,1.0])
        self.camera_distance = 2.5 * params_xml['space_size'][0]

        self.normalize_obs = False #Initialliy we need to set it to false to prevent using normalization during initialization
        print('Pickle init ...')
        utils.EzPickle.__init__(self)

        print('Mujoco model init ...')
        mujoco_env.MujocoEnv.__init__(self, model_path=env_config, frame_skip=frame_skip)


        self.normalize_obs = normalize_obs
        # self.viewer = self._get_viewer()
        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = np.zeros_like(self.model.data.qvel.ravel())
        self.init_qfrc = self.model.data.qfrc_applied.ravel().copy()

        action_low  = np.array([-1., -1.])
        action_high = np.array([ 1.,  1.])
        self.action_space = spaces.Box(action_low, action_high)

        if goal_tolerance is None:
            self.goal_tolerance = self.goal_radius

        self.get_obs_space()

        print('Pose: ', self.model.data.qpos.flatten())
        print('Vel: ', self.model.data.qvel.flatten())
        print('Com block: ', self.get_body_com("block"))
        print('Com goal: ', self.get_body_com("goal"))

        # self.goal_static = self.pose2goal(goal) #Must be filled in first since it will contain the full state
        # print('goal static: ', self.goal_static)
        self.goal = copy.deepcopy(self.goal_static)

        if self.tuple_obspace:
            self.observation_space = gym_tuple((self.observation_space,))

        if self.spec is None:
            self.spec = gym_reg.EnvSpec(id='BlocksSimple-v0', max_episode_steps=self.max_episode_steps)

        self.max_action = np.linalg.norm(np.abs(self.action_space.high))

        print('Max action: ', self.max_action)


    def get_obs_indx_names(self):
        self.obs_indx_name = {
            0: 'x',
            1: 'y',
            2: 'z',
            3: 'q0',
            4: 'q1',
            5: 'q2',
            6: 'q3',
            7: 'Vx',
            8: 'Vy',
            9: 'goal_x',
            10: 'goal y'
        }

    def get_obs_space(self):
        ## Observations
        if self.normalize_obs:
            if not self.orient_euler:
                obs_low = np.array([-1., -1., -1.,  -1., -1., -1., -1.,  1.,1.,  -1, -1])
                # Quaternion is supposed to be normalized, but every inidividual dimension can take 1. as max value
                obs_high = np.array([1., 1., 1.,  1., 1., 1., 1.,  1.,1.,  1., 1.])

                self.obs_bias  = np.array([0., 0., 0.,  0., 0., 0., 0.,  1.,1.,  0., 0.])
                self.obs_scale = np.array([1./self.space_size[0], 1./self.space_size[1], 1./self.space_size[2],
                                           1., 1., 1., 1.,
                                           1.,1.,
                                           1./self.space_size[0], 1./self.space_size[1]])

            else:
                raise NotImplemented
        else:
            if not self.orient_euler:
                obs_low = np.array([-self.space_size[0], -self.space_size[1], -self.space_size[2],
                                    -1., -1., -1., -1.,
                                    1., 1.,
                                    -self.space_size[0], -self.space_size[1]])
                # Quaternion is supposed to be normalized, but every inidividual dimension can take 1. as max value
                obs_high = np.array([self.space_size[0], self.space_size[1], self.space_size[2],
                                     1., 1., 1., 1.,
                                     1., 1.,
                                     self.space_size[0], self.space_size[1]])

                self.obs_bias  = np.array([0., 0., 0.,  0., 0., 0., 0.,  0.,0.,  0., 0.])
                self.obs_scale = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
            else:
                raise NotImplemented

        self.observation_space = spaces.Box(obs_low, obs_high)
        return self.observation_space

    ## Property for checking goal tolerance
    @property
    def goal_tolerance(self):
        return self._goal_tolerance

    @goal_tolerance.setter
    def goal_tolerance(self, tolerance):
        if tolerance is None:
            tolerance = 0.0
        self._goal_tolerance = np.array(tolerance)
        if self._goal_tolerance.size == 1:
            self._goal_tolerance = np.repeat(self._goal_tolerance, self.goal_feature_size)
        elif self._goal_tolerance.size != self.goal_feature_size:
            raise ValueError(self.__class__.__name__ + ': wrong goal tolerance size')

    ## Property introduced for compatibility
    @property
    def step_limit(self):
        return self.max_episode_steps

    @step_limit.setter
    def step_limit(self, tmax):
        self.max_episode_steps = tmax

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, goal_in):
        if isinstance(goal_in, tuple):
            goal_in = goal_in[0]
        self._goal = np.array(goal_in)
        self._goal = self._goal[0:2]

    def read_xml_parameters(self, xml_filename):
        xmldict = anymarkup.parse_file(xml_filename)
        geoms = xmldict['mujoco']['worldbody']['geom']
        bodies = xmldict['mujoco']['worldbody']['body']
        table_top_name = 'tableTop'
        table_roof_name = 'tableRoof'
        table_wall_name = 'table_wall_1'
        table_top_geom = None
        table_roof_geom = None
        table_wall_geom = None
        for geom in geoms:
            if geom['@name'] == table_top_name:
                table_top_geom = copy.deepcopy(geom)
            if geom['@name'] == table_roof_name:
                table_roof_geom = copy.deepcopy(geom)
            if geom['@name'] == table_wall_name:
                table_wall_geom = copy.deepcopy(geom)
        assert table_top_geom is not None
        assert table_roof_geom is not None
        table_top_size = [float(x) for x in table_top_geom['@size'].split(' ')]
        table_roof_pos = [float(x) for x in table_roof_geom['@pos'].split(' ')]
        table_wall_size = [float(x) for x in table_wall_geom['@size'].split(' ')]
        space_size = copy.deepcopy(table_top_size)
        space_size[2] = table_roof_pos[2] #Pos of the table roof defines max verical size of the space size
        table_wall_width = table_wall_size[2]

        block_body_name = 'block'
        goal_body_name = 'goal'
        body_goal = None
        body_block = None
        for body in bodies:
            if body['@name'] == block_body_name:
                body_block = copy.deepcopy(body)
            if body['@name'] == goal_body_name:
                body_goal = copy.deepcopy(body)

        assert body_block is not None
        assert body_goal is not None

        goal_radius = float(body_goal['geom']['@size'])

        params = {'space_size':space_size,
                  'table_wall_width': table_wall_width,
                  'goal_radius': goal_radius}
        print(params)
        return params

    @property
    def goal_static(self):
        return self._goal_static

    @goal_static.setter
    def goal_static(self, goal_in):
        self._goal_static = np.array(goal_in)[:2]  # assigning fingertip as a goal, i.e. init of goal from the state

    def is_multigoal(self):
        return not self.use_static_goal

    def apply_action(self, action_, n_frames):
        # print('!!!!!!!!!!! action size: ', action_.shape)
        action_ = np.array(action_)
        action_ = np.clip(action_, -1, 1)
        if action_.size == 0:
            print('WARNING: Empty action has been applied. It is ok during init of env, but not ok after.')
            for _ in range(n_frames):
                self.model.step()
        else:
            action = np.zeros([9])
            action_xyz = copy.deepcopy(self.get_body_com("block"))
            action_xyz[2] = 0.07
            action[:3] =  action_xyz #Point of force application - applying exactly at the center of mass (COM)
            action[3:5] = action_ * self.Fmax       #Force: Fxy
            self.apply_action_general(action=action, n_frames=n_frames)

    def apply_action_general(self, action, n_frames):
        # print('apply action general: ', action)
        # self.model.data.ctrl = ctrl
        self.model.data.qfrc_applied = self.model.applyFT(action[:3], action[3:6], action[6:], 'block')
        for _ in range(n_frames):
            self.model.step()
        self.model.data.qfrc_applied = self.init_qfrc.ravel().copy()

    def do_simulation(self, n_frames):
        for _ in range(n_frames):
            self.model.step()

    # def _get_viewer(self, reset=False):
    #     if self.viewer is None or reset:
    #         if self.viewer is not None and reset: self.viewer.finish()
    #         self.viewer = mujoco_py.MjViewer(init_width=self.scene_width,
    #                                          init_height=self.scene_height)
    #         self.viewer.start()
    #         self.viewer = self.viewer_setup()
    #         self._set_viewer_data(self.viewer)
    #     return self.viewer
    #
    # def _set_viewer_data(self, viewer):
    #     glfw.make_context_current(viewer.window)
    #     self.viewer_setup(viewer)
    #     viewer.model = self.model
    #     viewer.data = self.model.data
    #     if viewer.running:
    #         if viewer.model:
    #             if viewer.model:
    #                 mjlib.mjr_makeContext(viewer.model.ptr, byref(viewer.con), 150)
    #             else:
    #                 mjlib.mjr_makeContext(None, byref(viewer.con), 150)
    #             viewer.render()
    #     return viewer

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(init_width=self.scene_width, init_height=self.scene_height)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def viewer_setup(self):
        '''
        Sets the camera to the top-view of the tabletop.
        :return:
        '''
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -90.00
        self.viewer.cam.azimuth = 90.33
        self.viewer.cam.distance = self.camera_distance

    def is_steady(self):
        """
        This method checks if the simulation is in a steady state by comparing the COMs
        of the blocks between two successive steps in simulation. Calling this will change the
        preset state if not already steady.
        """
        Vxy = self.model.data.qvel.flat[:2] #2 instead of 3 since we are skipping z dimension as irrelevant for now
        Vang = self.model.data.qvel.flat[3:(3 + self.angle_coord_size)]
        # print(Vxy, Vang, self.Vxy_null, self.Vang_null)
        return all(np.isclose(Vxy, self.Vxy_null, atol=self.Vxy_atol, rtol=self.Vxy_rtol)) \
               and all(np.isclose(Vang, self.Vang_null, atol=self.Vang_atol, rtol=self.Vang_rtol))

    def _render(self, mode='human', close=False):
        self.render_current_step = True
        super(BlocksSimple, self)._render(mode=mode, close=close)

    def _step(self, a):
        # Don't move if you are in the deadzone
        self.time_step += 1
        self.frames_skipped = 0
        goal_direction_vec = self.get_body_com("block")-self.get_body_com("goal")
        xyz_prev = self.get_body_com("block")

        ## Running simulation
        # --- Apply action
        self.frames_skipped += self.force_duration
        # print(self.__class__.__name__, ': applying action ...')

        # print(self.__class__.__name__,': action: ', a)
        self.apply_action(action_=a, n_frames=self.force_duration)

        # --- Wait till it settles
        # print(self.__class__.__name__, ': waiting simulation to finish ...')
        while not self.is_steady():
            self.frames_skipped += self.frame_skip
            # if self.render_current_step:
            #     self.render()
            self.do_simulation(self.frame_skip)

        # self.block_vel = ((self.get_body_com("block") - self.pos_prev) / self.dt)[0:2]
        self.block_vel = self.get_pose_tuple()[1][:2]

        ## Here is with the velocity
        # self.goal_coord_absdiff = np.abs(np.concatenate(
        #     [(self.get_body_com("block") - self.get_body_com("goal"))[0:2],
        #      self.block_vel
        #      ]))
        self.goal_coord_absdiff = np.abs(self.get_body_com("block")[0:2] - self.get_body_com("goal")[0:2])

        self.pos_prev = self.get_body_com("block")

        center_reached = np.all(self.goal_coord_absdiff < self.goal_tolerance)


        ob = self._get_obs()
        done = (self.time_step >= self.max_episode_steps) or \
               (center_reached and self.use_distance2center_stop_criteria)
        # print(self.__class__.__name__, ': center_reached:', center_reached, 'goal_coord_absdiff:', self.goal_coord_absdiff, 'goal:', self.goal, 'goal_tol:', self.goal_tolerance)
        # print('done:', done, 'time_step:', self.time_step, 'self.max_step:', self.max_episode_steps, 'center_reached:', center_reached, 'use_dist_stop:', self.use_distance2center_stop_criteria)

        if self.tuple_obspace:
            ob = (ob,)

        act_norm = np.linalg.norm(a)

        if self.dense_rewards:
            ## Old (dense)
            goal_vec = self.get_body_com("block") - self.get_body_com("goal")
            reward_dist = - np.linalg.norm(goal_vec)
            reward_ctrl = - np.square(a).sum()
            reward = reward_dist + reward_ctrl
        else:
            reward = int(center_reached)


        info = {
            'center_reached': center_reached,
            'distance2center_norm': np.linalg.norm(goal_direction_vec) / self.max_dist2goal,
            'act_force_norm': act_norm / self.max_action,
            'act_force': act_norm,
            'act_min_dist': 0,  # just for compatibility
            'act_min_dist_norm': 0, #just for compatibility
            'digit_revealed': False, #just for compatibility
            'rew_mnist': 0., #just for compatibility
            'rew_mnistANDtargetloc': 0., #just for compatibility
            'xyz': self.get_body_com("block"),
            'xyz_prev': xyz_prev,
            'xyz_prev_normalized': xyz_prev / self.space_size,
            'xyz_goal_relative': self.get_body_com("block") - self.get_body_com("goal"),
            'xyz_goal_relative_prev': xyz_prev - self.get_body_com("goal"),
            'goal': self.get_body_com("goal")[:2],
            }

        self.render_current_step = False
        # print('Obs: ', ob, ' State: ', self.get_all_pose(), 'xyz_prev:', xyz_prev , 'goal', self.goal)
        # print('Frames skipped:', self.frames_skipped, 'Steps: ', self.time_step)

        return ob, reward, done, info

    def _reset(self):
        ob = super(BlocksSimple, self)._reset()
        self.time_step = 0
        self.pos_prev = self.get_body_com("block")

        if self.tuple_obspace:
            ob = (ob,)
        return ob

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -90.00
        self.viewer.cam.azimuth = 90.33
        self.viewer.cam.distance = self.camera_distance

    def reset_model(self):
        """
        This function is used in the _reset() function of the parent class
        :return: (np.array) observation
        """
        block_xyz = self.np_random.uniform(low =np.array([-self.space_size[0] + self.table_wall_width, -self.space_size[1] + self.table_wall_width, 0.08]),
                                           high=np.array([ self.space_size[0] - self.table_wall_width,  self.space_size[1] - self.table_wall_width, 0.08]))

        block_orient = copy.deepcopy(self.orient_default)

        if self.use_static_goal:
            # print('!!!!!!!!!!!! Static goal init is used')
            self.goal = copy.deepcopy(self.goal_static)
        else:
            self.goal = self.np_random.uniform(low=np.array([-self.space_size[0], -self.space_size[1]]),
                                           high=np.array([ self.space_size[0],  self.space_size[1]]))

        goal_xyz = np.array([self.goal[0], self.goal[1]])
        qpos = np.concatenate([block_xyz, block_orient, goal_xyz])
        qvel = copy.deepcopy(self.init_qvel)
        if self.zero_speed_init:
            qvel = np.zeros_like(qvel)

        # print('Init qpos: ', qpos, ' Init qvel: ', qvel, ' goal: ', self.goal, 'Cur pose: ', self.get_all_pose())
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_all_pose(self):
        return (self.model.data.qpos.flat[:], self.model.data.qvel.flat[:])

    def _get_obs(self):
        qpos = self.model.data.qpos.flat[:]
        qvel = self.model.data.qvel.flat[:]

        blocks_pose = qpos[:self.pose_size]
        blocks_vel  = qvel[:2]
        goal_xy = qpos[self.pose_size:(self.pose_size + 2)]

        ob = np.concatenate([
            blocks_pose,
            blocks_vel,
            goal_xy
        ])

        if self.normalize_obs:
            ob = (ob - self.obs_bias) * self.obs_scale
        return ob

    def state2obs(self, state):
        """
        Convert state to obs. Need it for a sampler classifier
        :param state:
        :return:
        """
        qpos = state[0]
        qvel = state[1]

        blocks_pose = qpos[:self.pose_size]
        blocks_vel  = qvel[:2]
        goal_xy = qpos[self.pose_size:(self.pose_size + 2)]

        ob = np.concatenate([
            blocks_pose,
            blocks_vel,
            goal_xy
        ])

        if self.normalize_obs:
            ob = (ob - self.obs_bias) * self.obs_scale
        return ob

    def get_pose_tuple(self):
        return (
            self.model.data.qpos.flat[:self.pose_size], #blocks 7d pose
            self.model.data.qvel.flat[:self.pose_size], #blocks 7d vel
            self.model.data.qpos.flat[self.pose_size:(self.pose_size + 2)] #goal x,y
        )

    def pose2goal(self, pose):
        """
        Transforms pose state to the goal state
        Or in case pose is provided as xy - transforms to full state
        :param pose:
        :return:
        """
        if not isinstance(pose, tuple):
            pose_out = (copy.deepcopy(self.init_qpos), copy.deepcopy(self.init_qvel))
            pose_out[0][:2] = pose[:2]

            pose_out[0][-2:] = pose_out[0][:2] #current pose to goal
        return pose_out

    def reload_model(self, pose=None, goal=None):
        """
        Reloads the model at prespecified pose / goal. It is essential for hide/seek (teacher/student) rollouts
        Essentially it:
        - randomizes and matches location of the hand to the goal if both not specified: needed for the hide rollout, since they should be random and match at the beginning
        - Keeps goal the same and sets pose if only pose is specified: needed for the student rollout coming right after teacher's rollout to keep goal the same
        - Matches hand to the goal if only goal is specified (needed for hide rollouts and static goals since they are always pre-filled)
        (i.e. both provided or pose is provided and a static goal is used)
        :param pose: Reload with a pose. If pose is None - it will be initialized with a location of a goal (specifically for hide rollouts)
            (Example: [[1.046, -2.093, 0., 0.],[0.,0.,0.,0.],[0.,0.], [0.,0.]])
        :param goal: Specify goal that you want. If None then:
                - if pose is not specified - both will be matched and both will be random (specifically for hide/seek rollouts)
                - if pose specified then goal will be the same, but pose will be changed to specified
            prefilled with a static goal if you specifier using static goal in the constructor
            (Example: [[1.046, -2.093, 0., 0.],[0.,0.,0.,0.],[0.,0.],[0.,0.]])
        :return:
        """
        if pose is not None:
            pose = copy.deepcopy(pose)

        if goal is not None:
            goal = copy.deepcopy(goal)
        elif self.use_static_goal:
            goal = copy.deepcopy(self.goal_static)

        if pose is None:
            ## Here we need to randomize the pose to the goal location, since they were not specified
            if goal is None:
                qpos = copy.deepcopy(self.init_qpos)
                qvel = copy.deepcopy(self.init_qvel)
                if self.is_multigoal():
                    block_xyz = self.np_random.uniform(low=np.array(
                        [-self.space_size[0] + self.table_wall_width, -self.space_size[1] + self.table_wall_width, 0.08]),
                                                       high=np.array([self.space_size[0] - self.table_wall_width,
                                                                      self.space_size[1] - self.table_wall_width, 0.08]))
                    self.goal = block_xyz[:2]
                else:
                    self.goal = copy.deepcopy(self.goal_static)
                    block_xyz = np.array(self.goal[0], self.goal[1], 0.08)

                ## Matching block and goal for hide-seek rollout
                qpos[:2] = block_xyz[:2]
                # Setting goal == block
                qpos[self.pose_size:(self.pose_size + 2)] = block_xyz[:2]
                # print('reload_model: qpos:', qpos, 'goal:', self.goal , 'block_xyz:', block_xyz)
                self.set_state(qpos, qvel)
            else:
                ## If the goal is specified then pose and goal are matched
                # Needed this way because often the goal is pre-filled and we need to replicate behavior from both None
                self.goal = goal
                qpos = copy.deepcopy(self.init_qpos)
                qvel = copy.deepcopy(self.init_qvel)
                qpos[self.pose_size:] = self.goal
                qpos[:2] = self.goal[:2]
                if not self.orient_euler:
                    qpos[3] = 1.0 #Quaternion
                # print('reload_model: qpos:', qpos, 'goal:', self.goal)
                self.set_state(qpos, qvel)
                # print('state after reload: ', self.get_all_pose())

        else:
            ## If pose was specified then:
            # - if goal is None then it is kept the same
            # - if goal is not None then both will be set to prespecified locations
            self.time_step = 0
            if self.zero_speed_init:
                init_vel = copy.deepcopy(self.init_qvel)
            else:
                init_vel = pose

            if goal is not None:
                self.goal = goal
            # print('reload_model: pose: ', pose, 'goal:', self.goal)
            pose[0][self.pose_size:] = self.goal #keeps goal the same

            ## If we have single goal, then we enforce this goal, just in case
            if not self.is_multigoal():
                pose[0][self.pose_size:] = self.goal_static
                self.goal = copy.deepcopy(self.goal_static)
                # print('goal_static: ', self._goal_static)

            self.set_state(qpos=np.array(pose[0]), qvel=init_vel)
            # print('reload_model: qpos:', np.array(pose[0]), 'goal:', self.goal)
            # print('pose[0]', self.goal, goal)

        ob = self._get_obs()
        if self.tuple_obspace:
            ob = (ob,)
        return np.array(ob)

    def get_task_features(self, obs):
        return obs


    def get_all_rotation_matrix(self):
        """
        Returns a list of flattened rotation matrix of each of the blocks in the scene.
        """
        qpos = self.model.data.qpos.ravel().copy()
        rotmats = []
        for i in range(len(self.model.data.xpos[1:])):
            orientation = qpos[i * self.pose_size + 3 : i * self.pose_size + self.pose_size]
            rotmats.append(list(quat2mat(orientation).flatten()))
        return rotmats

    # def get_all_pose(self, qpos=None):
    #     """
    #     Returns an array of current poses(position + orientation) of blocks in the scene.
    #     Orientation in euler angles.
    #     """
    #     pose = []
    #     qpos = self.model.data.qpos.ravel().copy() if qpos is None else qpos
    #     for i in range(len(self.model.data.xpos[1:])):
    #         orientation = qpos[i * 7 + 3 : i * 7 + 7]
    #         euler = np.array([mat2euler(quat2mat(orientation))]).flatten()
    #         position = qpos[i * 7 : i * 7 + 3]
    #         pose.append(np.hstack([position, euler]))
    #     return np.array(pose)



class BlocksSimpleXY(BlocksSimple):
    """
    BlocksSimple version with
    [x,y,goal_x, goal_y]
    as observations
    """
    def _get_obs(self):
        qpos = self.model.data.qpos.flat[:]
        blocks_pose = qpos[:2]
        goal_xy = qpos[self.pose_size:(self.pose_size + 2)]

        ob = np.concatenate([
            blocks_pose,
            goal_xy
        ])

        if self.normalize_obs:
            ob = (ob - self.obs_bias) * self.obs_scale
        return ob


    def get_obs_space(self):
        ## Observations
        if self.normalize_obs:
            obs_low = np.array([-1., -1., -1, -1])
            # Quaternion is supposed to be normalized, but every inidividual dimension can take 1. as max value
            obs_high = np.array([1., 1., 1., 1.])

            self.obs_bias = np.array([0., 0., 0., 0.])
            self.obs_scale = np.array([1. / self.space_size[0], 1. / self.space_size[1],
                                       1. / self.space_size[0], 1. / self.space_size[1]])


        else:
            if not self.orient_euler:
                obs_low = np.array([-self.space_size[0], -self.space_size[1], -self.space_size[2],
                                    -self.space_size[0], -self.space_size[1]])
                # Quaternion is supposed to be normalized, but every inidividual dimension can take 1. as max value
                obs_high = np.array([self.space_size[0], self.space_size[1], self.space_size[2],
                                     self.space_size[0], self.space_size[1]])

                self.obs_bias  = np.array([0., 0., 0., 0.])
                self.obs_scale = np.array([1., 1., 1., 1.])
            else:
                raise NotImplemented

        self.observation_space = spaces.Box(obs_low, obs_high)
        return self.observation_space

    def state2obs(self, state):
        """
        Convert state to obs. Need it for a sampler classifier
        :param state:
        :return:
        """
        qpos = state[0]
        blocks_pose = qpos[:2]
        goal_xy = qpos[self.pose_size:(self.pose_size + 2)]

        ob = np.concatenate([
            blocks_pose,
            goal_xy
        ])

        if self.normalize_obs:
            ob = (ob - self.obs_bias) * self.obs_scale
        return ob

    def get_obs_indx_names(self):
        self.obs_indx_name = {
            0: 'x',
            1: 'y',
            2: 'goal_x',
            3: 'goal y'
        }

class BlocksSimpleXYQ(BlocksSimple):
    """
    BlocksSimple version with
    [x,y, quat, goal_x, goal_y]
    as observations
    """

    def _get_obs(self):
        qpos = self.model.data.qpos.flat[:]

        blocks_pose = qpos[:self.pose_size]
        goal_xy = qpos[self.pose_size:(self.pose_size + 2)]

        ob = np.concatenate([
            blocks_pose,
            goal_xy
        ])

        if self.normalize_obs:
            ob = (ob - self.obs_bias) * self.obs_scale
        return ob

    def get_obs_space(self):
        ## Observations
        if self.normalize_obs:
            obs_low = np.array([-1., -1., -1.,  -1., -1., -1., -1.,  -1., -1.])
            # Quaternion is supposed to be normalized, but every inidividual dimension can take 1. as max value
            obs_high = np.array([1., 1., 1.,  1., 1., 1., 1.,  1., 1.])

            self.obs_bias = np.array([0., 0., 0.,  0., 0., 0., 0.,  0., 0.])
            self.obs_scale = np.array([1. / self.space_size[0], 1. / self.space_size[1], 1. / self.space_size[2],
                                       1.,1.,1.,1.,
                                       1. / self.space_size[0], 1. / self.space_size[1]])


        else:
            if not self.orient_euler:
                obs_low = np.array([-self.space_size[0], -self.space_size[1], -self.space_size[2],
                                    -1., -1., -1., -1.,
                                    -self.space_size[0], -self.space_size[1]])
                # Quaternion is supposed to be normalized, but every inidividual dimension can take 1. as max value
                obs_high = np.array([self.space_size[0], self.space_size[1], self.space_size[2],
                                     1., 1., 1., 1.,
                                     self.space_size[0], self.space_size[1]])

                self.obs_bias = np.array([0., 0., 0.,  0., 0., 0., 0.,  0., 0.])
                self.obs_scale = np.array([1., 1., 1.,  1.,1.,1.,1.,  1., 1.])
            else:
                raise NotImplemented

        self.observation_space = spaces.Box(obs_low, obs_high)
        return self.observation_space

        self.observation_space = spaces.Box(obs_low, obs_high)
        return self.observation_space

    def state2obs(self, state):
        """
        Convert state to obs. Need it for a sampler classifier
        :param state:
        :return:
        """
        qpos = state[0]

        blocks_pose = qpos[:self.pose_size]
        goal_xy = qpos[self.pose_size:(self.pose_size + 2)]

        ob = np.concatenate([
            blocks_pose,
            goal_xy
        ])

        if self.normalize_obs:
            ob = (ob - self.obs_bias) * self.obs_scale
        return ob

    def get_obs_indx_names(self):
        self.obs_indx_name = {
            0: 'x',
            1: 'y',
            2: 'z',
            3: 'q0',
            4: 'q1',
            5: 'q2',
            6: 'q3',
            7: 'goal_x',
            8: 'goal_y'
        }

