from operator import itemgetter

import lasagne
import lasagne.init as LI
import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.nonlinearities as NL
import numpy as np
import theano
import theano.tensor as TT
import theano.tensor as TT
from e2eap_training.rllab_utils.core.network import ConvNetwork
from gym.spaces import Tuple as gym_tuple
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.policies.base import StochasticPolicy
from rllab.spaces import Box
from rllab.spaces.product import Product as rllab_tuple
from rllab_utils.core.lasagne_layers import ParamMultiInLayer
from e2eap_training.utils import print_format as pf
import copy


class ConvMultiObsNetwork(object):
    def __init__(self,
                 input_shapes,
                 output_dim,
                 obs_network_params=[],
                 input_layers=None,
                 fusion_net_params=None,
                 hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 use_flat_obs=True,
                 name=None):
        """
        :param output_dim: (list of int or just int) number of output dimensions
        :param output_nonlinearities: (nonlinearity of a list of nonlinearities if output_dim is a list as well)
        :param input_layers: (list of input layers)
        :param fusion_hidden_sizes: (tuple of int) fusion network hidden sizes
        :param obs_network_params: (list of dict) parameters of observation networks:
            conv_filters: (dict of int) number of conv filters per layer
            conv_filter_sizes: (dict of int) spatial sizes of square conv filters (only supports square filters at this point)
            conv_strides: (dict of int) strides of convolutional filters
            conv_pads: (dict of str) padding alg to use by layer: Options: 'valid', 'full', 'same'.
                        See lasagne documentation: http://lasagne.readthedocs.io/en/latest/modules/layers/conv.html
            hidden_sizes: (dict of int) number of hidden units per fully connected layer
            hidden_nonlinearity: (str) nonlinearity for hidden units: See lasagne for options. Examples: 'rectify', 'tanh'
            output_nonlinearity: (str) nonlinearity of outputs: see lasagne docs.
        :param name: (str) name
        """
        self.use_flat_obs = use_flat_obs

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        ## Setting default values of parameters
        if fusion_net_params is None:
            fusion_net_params = {}

        if 'hidden_sizes' not in fusion_net_params:
            fusion_hidden_sizes = []
        else:
            fusion_hidden_sizes = fusion_net_params['hidden_sizes']

        if 'hidden_nonlinearity' not in fusion_net_params:
            fusion_net_params['output_nonlinearities'] = 'rectify'

        if 'output_nonlinearities' not in fusion_net_params:
            output_nonlinearities = None
            print('!!! WARNING: Output nonlinearities were not assigned ')
        else:
            output_nonlinearities = fusion_net_params['output_nonlinearities']

        obs_networks = []
        obs_network_outputs = []
        obs_network_inputs = []
        obs_network_input_vars = []

        if not isinstance(input_layers, list):
            input_layers = [input_layers] * len(obs_network_params)

        # for net_i, obs_param in enumerate(obs_network_params):
        print(self.__class__.__name__ + ': OBS NUM = ', len(input_shapes), ' OBS_SHAPES: ', input_shapes)
        for net_i in range(len(input_shapes)):
            obs_param = obs_network_params[net_i]
            ## Checking parameters
            if 'conv_filters' not in obs_param or 'conv_filters' is None:
                obs_param['conv_filters'] = []
            if 'conv_filter_sizes' not in obs_param or 'conv_filter_sizes' is None:
                obs_param['conv_filter_sizes'] = []
            if 'conv_strides' not in obs_param or 'conv_strides' is None:
                obs_param['conv_strides'] = [1] * len(obs_param['conv_filters'])
            if 'conv_pads' not in obs_param or 'conv_pads' is None:
                obs_param['conv_pads'] = ['valid'] * len(obs_param['conv_filters'])

            if 'hidden_sizes' not in obs_param or 'hidden_sizes' is None:
                obs_param['hidden_sizes'] = []
            if 'hidden_nonlinearity' not in obs_param:
                obs_param['hidden_nonlinearity'] = LN.rectify
            if 'output_nonlinearity' not in obs_param:
                obs_param['output_nonlinearity'] = LN.rectify

            # If you flatten images before you feed them the use_flat_obs = True
            # WARNING: At this moment it actually breaks if I don't use flat obs
            var_name = 'obs_%d' % (net_i)
            if self.use_flat_obs:
                obs_var = theano.tensor.matrix(name=var_name)
            else:
                if len(input_shapes[net_i]) == 3:
                    obs_var = theano.tensor.tensor4(name=var_name)
                elif len(input_shapes[net_i]) == 2:
                    obs_var = theano.tensor.tensor3(name=var_name)
                else:
                    obs_var = theano.tensor.matrix(name=var_name)

            obs_net = ConvNetwork(
                input_shape=input_shapes[net_i],
                input_layer=input_layers[net_i],
                conv_filters=obs_param['conv_filters'],
                conv_filter_sizes=obs_param['conv_filter_sizes'],
                conv_strides=obs_param['conv_strides'],
                conv_pads=obs_param['conv_pads'],
                hidden_sizes=obs_param['hidden_sizes'],
                hidden_nonlinearity=getattr(NL, obs_param['hidden_nonlinearity']),
                output_nonlinearity=getattr(NL, obs_param['output_nonlinearity']),
                name=name + '_obs%d' % net_i,
                input_var=obs_var,
                use_flat_obs=use_flat_obs
            )
            obs_networks.append(obs_net)
            obs_network_inputs.append(obs_net.input_layer)
            obs_network_input_vars.append(obs_net.input_var)
            embed_shape = obs_net.output_layer.output_shape
            embed_shape_flat = ([0], int(np.prod(embed_shape[1:])))
            obs_network_outputs.append(L.ReshapeLayer(obs_net.output_layer, shape=embed_shape_flat))
            print('Obs_%d Flattened output shape:' % net_i, embed_shape_flat)

        print('--- FUSION NET ----------------------------------------------------------')
        # Concatenating observation layers
        l_hid = L.ConcatLayer(obs_network_outputs)
        print('Merged obs embeding shape:', l_hid.output_shape)

        # Fusion MLP layers
        for idx, hidden_size in enumerate(fusion_hidden_sizes):
            l_hid = L.DenseLayer(
                l_hid,
                num_units=hidden_size,
                nonlinearity=getattr(NL, fusion_net_params['hidden_nonlinearity']),
                name="%shidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )
            print('Dense layer out shape = ', l_hid.output_shape, ' Nonlinearity = ', fusion_net_params['hidden_nonlinearity'])

        # Outputs
        if not isinstance(output_dim, list):
            output_dim = [output_dim]

        if not isinstance(output_nonlinearities, (list,tuple)):
            output_nonlinearities = [output_nonlinearities] * len(output_dim)
        else:
            assert len(output_nonlinearities) == len(output_dim), ' ERROR: Number of outputs does not match number of nonlinearities'

        outputs = []
        print('--- Fusion net outputs: ')
        for dim_i in range(len(output_dim)):
            outputs.append(L.DenseLayer(
                l_hid,
                num_units=output_dim[dim_i],
                nonlinearity=getattr(NL, output_nonlinearities[dim_i]),
                name="%s_output_%d" % (prefix, dim_i),
                W=output_W_init,
                b=output_b_init,
            ))
            print('Output %d shape = ' % dim_i, outputs[-1].output_shape, ' Nonlinearity = ',
                  output_nonlinearities[dim_i])

        # Concatenate outputs
        if len(outputs) != 0:
            l_out = L.ConcatLayer(outputs)
            print('Merged outputs shape: ', l_out.output_shape)
        else:
            print('!!! WARING: No outputs were specified, thus the last hidden layer of fusion net will be used')
            l_out = l_hid

        self._l_in = obs_network_inputs
        self._l_out = l_out
        self._input_vars = obs_network_input_vars

    @property
    def input_layers(self):
        """
        :return: (list of layers)
        """
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_vars(self):
        """
        :return: (list of theano vars)
        """
        return self._input_vars

class GaussianMultiObsPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env,
            obs_net_params,
            name,
            fusion_net_params=None,
            obs_indx=None,
            obs_shapes=None,
            action_dims=None,
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_net_parameters=None,
            std_fusion_net_params=None,
            std_share_network=False,
            min_std=1e-6,
            mean_network=None,
            std_network=None,
            use_flat_obs=True,
            dist_cls=DiagonalGaussian
    ):
        """
        :param env:
        :param obs_net_params: (list of dict) parameters for the observation networks
        :param name: (str) name is essential and should be consistent everywhere. Policy uses it to access relevant data
        :param fusion_net_params: (dict) parameters of the fusion network. If single observation type is used you could set it None
        :param obs_indx: (list of int) if obs is provided as tuple, which indices in tuple to use. If None is given tries to use all indices
        :param obs_shapes: (list of tuples of int) observation shapes for manual assignment (in case some other observations might be used)
        :param action_dims: (list of int  or just int) number of actions, If None is provided - env actions are used. Providing a list of action dims allows also have different nonlinearities for every subset of actions
        :param std_net_parameters: (dict) parameters for the std network
        :param learn_std: Is std trainable (does not need std network parameters)
        :param init_std: Initial std
        :param adaptive_std: (bool) should std to be learnable. If True specify std_network_parameters
        :param std_share_network:
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param mean_network: custom network for the output mean
        :param use_flat_obs: if set then one must provide observations in flat form even if they are images (insde they are reshaped). For some reason it breaks if they are not flat, so use True for now
        :param std_network: custom network for the output log std
        :return:
        """
        self.env_observation_space = copy.deepcopy(env.observation_space)
        self.name = name
        self.obs_indx = obs_indx
        # Update observation indices and shapes
        self.obs_shapes = self.checkListOfTuples(obs_shapes)
        self.use_flat_obs = use_flat_obs

        Serializable.quick_init(self, locals())
        assert isinstance(env.action_space, Box)

        # If not specified - use action space of the environment
        pf.print_sec0('MULTIOBSERVATION POLICY:')
        if action_dims is None:
            assert env is not None, "ERROR: getNetwork(): Either provide env or output_dim"
            action_dims = np.prod(env.action_space.shape)

        # create network
        if mean_network is None:
            mean_network = self.getNetwork(obs_net_params=obs_net_params,
                                           fusion_net_params=fusion_net_params,
                                           obs_shapes=self.obs_shapes,
                                           output_dim=action_dims,
                                           name=name + '_mean_net',
                                           use_flat_obs=self.use_flat_obs,
                                           env=env)
            ## Typical parameters
            # hidden_sizes=(32, 32),
            # hidden_nonlinearity=NL.tanh,
            # output_nonlinearity=None,

        self._mean_network = mean_network
        l_mean = mean_network.output_layer

        self.input_vars = []
        for layer in mean_network.input_layers:
            self.input_vars.append(layer.input_var)

        if std_network is not None:
            l_log_std = std_network.output_layer
        else:
            if adaptive_std:
                std_network = self.getNetwork(obs_net_params=std_net_parameters,
                                              fusion_net_params=std_fusion_net_params,
                                              obs_layers=mean_network.input_layers,
                                              output_dim=action_dims,
                                              env=env,
                                              use_flat_obs=self.use_flat_obs,
                                              name=name + '_std_net')
                ## Typical parameters:
                # input_shape=(obs_dim,)
                # std_hidden_sizes = (32,32)
                # std_hidden_nonlinearity = NL.tanh
                # output_nonlinearity=None
                l_log_std = std_network.output_layer
            else:
                action_nums = np.sum(action_dims)
                l_log_std = ParamMultiInLayer(
                    mean_network.input_layers,
                    num_units=action_nums,
                    param=lasagne.init.Constant(np.log(init_std)),
                    name="output_log_std",
                    trainable=learn_std,
                )

        self._std_network = std_network
        self.min_std = min_std

        mean_var, log_std_var = L.get_output([l_mean, l_log_std])

        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(min_std))

        self._mean_var, self._log_std_var = mean_var, log_std_var

        self._l_mean = l_mean
        self._l_log_std = l_log_std

        self._dist = dist_cls(action_dims)
        LasagnePowered.__init__(self, [l_mean, l_log_std])
        super(GaussianMultiObsPolicy, self).__init__(env)

        self._f_dist = ext.compile_function(
            inputs=self.input_vars,
            outputs=[mean_var, log_std_var],
        )

    @property
    def obs_shapes(self):
        return self._obs_shapes

    @obs_shapes.setter
    def obs_shapes(self, shapes):
        if shapes is None:
            if self.obs_indx is None:
                if isinstance(self.env_observation_space, gym_tuple):
                    self.obs_indx = tuple([i for i in range(len(self.env_observation_space.spaces))])
                    shapes = []
                    for space in self.env_observation_space.spaces:
                        shapes.append(space.shape)
                elif isinstance(self.env_observation_space, rllab_tuple):
                    self.obs_indx = tuple([i for i in range(len(self.env_observation_space.components))])
                    shapes = []
                    for space in self.env_observation_space.components:
                        shapes.append(space.shape)
                else:
                    self.obs_indx = (0,)
                    shapes.append(self.env_observation_space.shape)
            else:
                if not isinstance(self.obs_indx, (list,tuple)):
                    self.obs_indx = [self.obs_indx]

                print('Observation space type = ', type(self.env_observation_space))
                if isinstance(self.env_observation_space, gym_tuple):
                    shapes = []
                    for idx in self.obs_indx:
                        shapes.append(self.env_observation_space.spaces[idx].shape)
                if isinstance(self.env_observation_space, rllab_tuple):
                    shapes = []
                    for idx in self.obs_indx:
                        shapes.append(self.env_observation_space.components[idx].shape)
                else:
                    raise ValueError('ERROR: Dont provide observation indices for non tuple observation space')
        self._obs_shapes = shapes
        print(self.__class__.__name__ + ': Observation shapes = ', self._obs_shapes)
        return shapes


    @staticmethod
    def getNetwork(obs_net_params, obs_shapes=None,
                   fusion_net_params=[], obs_layers=None,
                   output_dim=None, use_flat_obs=True,
                   env=None, name='net'):
        """
        Abstracts network initialization
        :param obs_net_params: (list of dict) network parameters for observation networks
        :param obs_shapes: (list of tuples of ints) list of observation shapes
        :param fusion_net_params: (dict) parameters of fusion network (MLP)
        :param output_dim: (list of int or just int) output dimensions. If not provided env action space size is used
        :param obs_layers: (list of layer vars) provide observation layers if you want to re-use them from somewhere
        :param input_layer: (layer) input layer (typically used for std network)
        :param env: (env_spec) environment spec (only necessary if output_dim is not provided)
        :param name: (str) name of the network
        :return: (network object)
        """
        # print("getNetwork(): ouput dim = ", output_dim)
        if output_dim is None:
            print('MultiObsPolicy: ', type(env))
            assert env is not None, "ERROR: getNetwork(): Either provide env or outpuPOLICIESt_dim"
            output_dim = np.prod(env.action_space.shape)

        network = ConvMultiObsNetwork(
            input_shapes=obs_shapes,
            input_layers=obs_layers,
            output_dim=output_dim,
            obs_network_params=obs_net_params,
            fusion_net_params=fusion_net_params,
            use_flat_obs=use_flat_obs,
            name=name
        )
        return network

    def checkListOfTuples(self, lst):
        if lst is not None:
            if len(lst) == 0:
                lst = [lst]
            elif isinstance(lst[0], list):
                for l in range(len(lst)):
                    lst[l] = tuple(l)
            elif isinstance(lst[0], tuple):
                pass
            else:
                lst = [lst]
        return lst

    def dist_info_sym(self, obs_vars, state_info_vars=None):
        if not isinstance(obs_vars, list):
            obs_vars = [obs_vars]

        # input_dict = {}
        # for idx, in_lyr in enumerate(self._std_network.input_layers):
        #     input_dict[in_lyr] = obs_vars[idx]
        # mean_var, log_std_var = L.get_output([self._l_mean, self._l_log_std], inputs=input_dict)

        mean_var, log_std_var = L.get_output([self._l_mean, self._l_log_std])
        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(self.min_std))
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        if not isinstance(observation, (tuple, list)):
            obs_list = [observation]
        else:
            if len(self.obs_indx) == 1:
                obs_list = [observation[self.obs_indx[0]]]
            else:
                obs_list = list(itemgetter(*self.obs_indx)(observation))
        # print(self.__class__.__name__, ':Obs shapes = ',[obs.shape for obs in obs_list])

        # Adding batch dimension to observations
        if self.use_flat_obs:
            obs_list = [np.expand_dims(obs.flatten(), axis=0) for obs in obs_list]
        else:
            obs_list = [np.expand_dims(obs, axis=0) for obs in obs_list]

        mean, log_std = [x[0] for x in self._f_dist(*obs_list)]
        # print('Action: mean = ', mean, ' std = ', log_std)
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        # flat_obs = self.observation_space.flatten_n(observations)
        if not isinstance(observations, (tuple, list)):
            obs_list = [observations]
        else:
            if len(self.obs_indx) == 1:
                obs_list = [observations[self.obs_indx[0]]]
            else:
                obs_list = list(itemgetter(*self.obs_indx)(observations))
        # print(self.__class__.__name__, ':Obs shapes = ',[obs.shape for obs in obs_list])

        # Adding batch dimension to observations
        if self.use_flat_obs:
            obs_list = [obs.reshape((obs.shape[0], -1), axis=0) for obs in obs_list]

        means, log_stds = self._f_dist(*obs_list)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self, obs_vars, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_vars: (list of Theano tensors or a theano tensor) list of theano tensors corresponding to observations
        :param old_dist_info_vars:
        :return:
        """

        new_dist_info_vars = self.dist_info_sym(obs_vars, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (TT.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * TT.exp(new_log_std_var)
        return new_action_var

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    def get_data(self, samples_data):
        """
        The function is meant to extract data relevant for this particular policy
        From samples_data dictionary
        :param samples_data: (dict) dictionary with data
        :return:
        """
        all_input_values, key_positions = self.extract(
            samples_data,
            ["observations", "actions", "advantages"],
            [self.obs_indx, None, None]
        )
        if self.use_flat_obs:
            print('get_data: obs_positions: ', key_positions['observations'])
            for i in key_positions['observations']:
                obs = all_input_values[i]
                all_input_values[i] = obs.reshape((obs.shape[0],-1))
        return tuple(all_input_values)

    @staticmethod
    def extract(x, keys, indx=None):
        """
        The function extracts values of keys from x into a list
        If x[key] is a tuple (for example in case of tuple observations) it will get individual values in the list
        indx allows one to specify if one does not need all elements in a tuple, but a subset of them
        :param keys: (list of str) a list of keys to extract.
        :param indx: (a list of lists or tuples of int) indices in a x[key] variable that we should extract.
        Use None if a certain key is not a tuple. If it is None will extract all values.
        IF the whole variable is set to None then all variables of all keys will be extractd
        Ex: indx=None or indx=[None,None,None] (in case 3 keys provided) to extract everything from all keys
        Ex: [(0,),None,None] will only extract 0 element from the variable with the first key and ex
        :return: (list) of extracted values, (dict of int) dictionary of indices occupied by every key in the value list
        """
        if indx is None:
            indx = [None] * len(keys)
        if isinstance(x, (dict,ext.lazydict)):
            values = []
            key_pos = {}
            pos = 0
            for k_i, k in enumerate(keys):
                key_pos[k] = []
                if isinstance(x[k], tuple):
                    # Simply add all values to the vector
                    if indx[k_i] is None:
                        key_pos[k].extend(np.arange(pos, pos + len(x[k])))
                        pos += len(x[k])
                        values.extend(x[k])
                    elif len(indx[k_i]) == 0:
                        raise ValueError('ERROR: extract(): empty list of indices provided')
                    # Select values according to indices provided
                    else:
                        key_pos[k].extend(np.arange(pos, pos + len(indx[k_i])))
                        pos += len(indx[k_i])
                        values_cur = itemgetter(*indx[k_i])(x[k])
                        if isinstance(values_cur, (tuple, list)):
                            values.extend(values_cur)
                        else:
                            values.append(values_cur)
                else:
                    key_pos[k].append(pos)
                    pos += 1
                    values.append(x[k])

            return values, key_pos
        # elif isinstance(x, list):
        #     return tuple([xi[k] for xi in x] for k in keys)
        else:
            raise NotImplementedError

    @property
    def distribution(self):
        return self._dist
