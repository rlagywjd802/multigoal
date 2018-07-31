# from sandbox.vime.algos.npo_expl import NPO
from rllab.core.serializable import Serializable
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from multigoal.rllab_utils.algos.npo_comp import NPO


class TRPO(NPO, Serializable):
    """
    Trust Region Policy Optimization for dual agent scenario (competition)
    """

    def __init__(
            self,
            optimizers=None,     #dictionary of optimizers
            optimizer_args=None, #dictionary of optimizer parameters
            agent_names = ['hide', 'seek'], #names of agents
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizers is None:
            optimizers = {}
            for name in agent_names:
                if optimizer_args is None:
                    optimizer_args = dict()
                optimizers[name] = ConjugateGradientOptimizer(**optimizer_args)
        super(TRPO, self).__init__(optimizers=optimizers, **kwargs)
