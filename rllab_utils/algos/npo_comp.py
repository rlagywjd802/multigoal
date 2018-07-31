import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.misc import ext
from rllab.misc.overrides import overrides
from multigoal.rllab_utils.algos.batch_polopt_hide_seek import BatchPolopt


class NPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizers=None,
            optimizer_args=None,
            step_size=0.01,
            **kwargs):
        if optimizers is None:
            raise ValueError('ERROR: NPO: Optimizers are not provided !!!!')
        self.optimizers = optimizers
        self.step_size = step_size
        super(NPO, self).__init__(**kwargs)
        print('NPO with exploration initialized')

    @overrides
    def init_opt(self, policy_name):
        is_recurrent = int(self.policies[policy_name].recurrent)

        ## By extra_dims they actually mean shape of the tensor
        # Thus, for recurrent they need an extra dimensions in the tensor to store sequences
        # We have 2 options:
        # - either re-use observation vars from policy
        # - create observation vars again (it gives an error at this point: probably requires to dublicate variables)
        reuse_obs_vars = True
        if reuse_obs_vars:
            obs_vars = self.policies[policy_name].input_vars
        else:
            obs_vars = []
            for idx, obs_shape in enumerate(self.policies[policy_name].obs_shapes):
                # name = 'obs_%d' % (idx)
                name = 'obs'
                obs_var_cur = self.env.observation_space.new_tensor_variable(
                    name,
                    extra_dims=1 + is_recurrent,
                )
                obs_vars.append(obs_var_cur)
            print('NPO: Observation vars are created for policy %s' % policy_name, obs_vars)

        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policies[policy_name].distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
        }
        old_dist_info_vars_list = [old_dist_info_vars[k]
                                   for k in dist.dist_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        # Here we need to get output variables based on input variables
        # dist_info_sym takes input features and spits out outputs of the policy graph
        # typically input variables are observations (sometimes actions as well)
        dist_info_vars = self.policies[policy_name].dist_info_sym(obs_vars, action_var)

        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(
            action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - \
                TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var)

        # Forming input list for the policy
        input_list = obs_vars + [action_var, advantage_var] + old_dist_info_vars_list

        if is_recurrent:
            input_list.append(valid_var)

        # print('NPO: Policy Input list: ', [var for var in input_list])
        # theano.printing.pydotprint(surr_loss, outfile="loss.png",
        #                            var_with_name_simple=True)
        self.optimizers[policy_name].update_opt(
            loss=surr_loss,
            target=self.policies[policy_name],
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data, policy_name, do_optimization=True):
        if do_optimization:
            all_input_values = self.policies[policy_name].get_data(samples_data)
            agent_infos = samples_data["agent_infos"]
            info_list = [agent_infos[k]
                         for k in self.policies[policy_name].distribution.dist_info_keys]
            all_input_values += tuple(info_list)
            if self.policies[policy_name].recurrent:
                all_input_values += (samples_data["valids"],)
            print('optimize_policy:', [var.shape for var in all_input_values])
            loss_before = self.optimizers[policy_name].loss(all_input_values)
            self.optimizers[policy_name].optimize(all_input_values)
            mean_kl = self.optimizers[policy_name].constraint_val(all_input_values)
            loss_after = self.optimizers[policy_name].loss(all_input_values)
            logger.record_tabular(policy_name + '_LossAfter', loss_after)
            logger.record_tabular(policy_name + '_MeanKL', mean_kl)
            logger.record_tabular(policy_name + '_dLoss', loss_before - loss_after)
        else:
            loss_after = 0
            mean_kl = 0
            loss_before = 0
            print('WARNING: %s OPTIMIZATION STOPPED: Max iter: %d' % (policy_name, self.hide_stop_improve_after))
            logger.record_tabular(policy_name + '_LossAfter', 0)
            logger.record_tabular(policy_name + '_MeanKL', 0)
            logger.record_tabular(policy_name + '_dLoss', 0)

        return {policy_name + '_LossAfter': loss_after,
                policy_name + '_MeanKL': mean_kl,
                policy_name + '_dLoss': loss_before - loss_after}

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policies=self.policies,
            baselines=self.baselines,
            env=self.env,
        )

    def shape_is_img(self, shape):
        if len(shape) > 1:
            return True
        else:
            return False

## Not clear
# self.policies[policy_name].distribution.dist_info_keys
