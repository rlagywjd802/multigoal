import numpy as np

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
#from rllab.regressors.gaussian_conv_regressor import GaussianConvRegressor
# substituted to fix a bug of Nans appearing
from multigoal.rllab_utils.algos.gaussian_conv_regressor import GaussianConvRegressor

from gym.spaces import Tuple as gym_tuple
from rllab.spaces.product import Product as rllab_tuple

# from e2eap_training.rllab_utils.gaussian_conv_regressor import GaussianConvRegressor

class GaussianConvBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env,
            subsample_factor=1.,
            obs_indx=0,
            regressor_args=None,
            name='vf',
            error_file='errors__gaus_conv_baseline.txt'
    ):
        """
        :param env:
        :param subsample_factor:
        :param obs_indx: (int) in case observations are tuple (multiple modalities case), which observation index to use
        :param regressor_args: (dict) regressor parameters
        """
        Serializable.quick_init(self, locals())
        super(GaussianConvBaseline, self).__init__(env)
        if regressor_args is None:
            regressor_args = dict()

        self.obs_indx = obs_indx
        if isinstance(env.observation_space, gym_tuple):
            obs_shape = env.observation_space.spaces[self.obs_indx].shape
        elif isinstance(env.observation_space, rllab_tuple):
            obs_shape = env.observation_space.components[self.obs_indx].shape
        else:
            obs_shape = env.observation_space.shape


        self._regressor = GaussianConvRegressor(
            input_shape=obs_shape,
            output_dim=1,
            name=name,
            error_file=error_file,
            **regressor_args
        )

    @overrides
    def fit(self, paths):
        # Here I am trying to concatenate all observations from all paths.
        # By this moment at every path we have observations which is tuple of np.arrays
        # Every np.array in the tuple already contains all observations in a single path
        if isinstance(paths[0]["observations"], tuple):
            observations = np.concatenate([p["observations"][self.obs_indx].reshape((p["observations"][self.obs_indx].shape[0],-1)) for p in paths])
        else:
            observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        # print('baselines: predict: observations type = ', type(path["observations"]), ' shape=', path["observations"].shape, ' type[0]', type(path["observations"][0]), '[0].shape = ', path["observations"][0].shape)
        if isinstance(path["observations"], tuple):
            return self._regressor.predict(path["observations"][self.obs_indx].reshape((path["observations"][self.obs_indx].shape[0],-1))).flatten()
        else:
            return self._regressor.predict(path["observations"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)



class LinearFeatureBaseline(Baseline):
    def __init__(self, env, obs_indx=0, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff
        self.obs_indx = obs_indx

    @overrides
    def get_param_values(self, **tags):
        return self._coeffs

    @overrides
    def set_param_values(self, val, **tags):
        self._coeffs = val

    def _features(self, path):
        if isinstance(path["observations"], tuple):
            o = np.clip(path["observations"][self.obs_indx], -10, 10)
        else:
            o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0

        # So here we construct features like [obs, obs^2, time_step, time_step^2, time_step^3, 1]
        feat_mat = np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)
        # print('features = ', feat_mat.shape, ' l = ', l, ' o = ', o.shape, 'al = ', al.shape)
        return feat_mat

    @overrides
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        reg_coeff = self._reg_coeff
        for _ in range(5):
            # print('rewards = ', returns)
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    @overrides
    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self._coeffs)