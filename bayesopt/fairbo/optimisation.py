from bayesopt.optimisation import Optimiser
from bayesopt.scoring import BaseScorer
from bayesopt.gaussian_processes import BaseGP

from typing import Type
from numpy.typing import ArrayLike

class FairOptmiser(Optimiser):

    def __init__(
        self,
        sensitiveFeatureIndex: int,
        fairnessGP: Type[BaseGP],
        fairnessScorer: Type[BaseScorer],
        *args,
        **kwargs
    )
        super().__init__(self, *args, **kwargs)

        self.sensitiveFeatureIndex = sensitiveFeatureIndex
        self.fairnessGP = fairnessGP
        self.fairnessScorer = fairnessScorer

    def _init_step(self, element: ArrayLike, confidenceLevel: float):

        super()._init_step(self, element, confidenceLevel)

        pass

