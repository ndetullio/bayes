from bayesopt.hyperparameter_space import HyperSpace
from bayesopt.performance.objective import (
    get_loss, 
    Bootstrap
)
from bayesopt.acquisition import (
    get_next_guess, 
    EiAcquisition
)

import numpy as np

class Optimiser:
    
    def __init__(
        self,
        ml_model,
        X,
        y,
        parameterSpace,
        objectiveGP,
        objectiveScore = 'accuracy',
        acquisitionFunction = EiAcquisition(),
        initBudget = 10,
        totalBudget = 100
    ):
    
        self.ml_model = ml_model
        self.X = X
        self.y = y
        self.objectiveGP = objectiveGP
        self.objectiveScore = objectiveScore
        self.initBudget = initBudget
        self.totalBudget = totalBudget
        self.objectiveInput = None
        self.objectiveTarget = []
        self.acquisitionFunction = acquisitionFunction

        if isinstance(parameterSpace, HyperSpace):
            self.Space = parameterSpace
        else:
            self.Space = HyperSpace(parameterSpace)
    
    def _init_step(self, element, confidenceLevel):

        params = self.Space.make_dict(element)
        self.ml_model.set_params(**params)

        loss, _, _ = get_loss(
            self.X,
            self.y,
            self.ml_model,
            self.resampler,
            score = self.objectiveScore,
            confidenceLevel = confidenceLevel
        )

        self.objectiveTarget.append(loss)

    def initialise(
        self,
        resampler,
        confidenceLevel = 0.95,
    ):
        
        # Reinitialise arrays
        self.objectiveInput = None
        self.objectiveTarget = []

        parameterSample = self.Space.sample(
            numSamples = self.initBudget, 
            returnSample = True
        )

        for i, element in enumerate(parameterSample, start=1):
            
            print(f'Running step {i} of the initialisation procedure.')
            self._init_step(element, confidenceLevel)
        
        # Update GP
        self.objectiveInput = parameterSample
        self.objectiveGP.fit(self.objectiveInput, self.objectiveTarget)
        
    def next_guess(self):

        nextGuess = get_next_guess(
            self.objectiveGP,
            self.objectiveInput,
            self.Space,
            acquisition=self.acquisitionFunction
        )

        return nextGuess

    def evaluate(
        self, 
        nextGuess, 
        confidenceLevel = 0.95
    ):

        self.objectiveInput = np.concatenate(
            (self.objectiveInput, nextGuess[np.newaxis].reshape(1,-1)),
            axis = 0
        )

        # Get ground truth for target
        params = self.Space.make_dict(nextGuess) 
        self.ml_model.set_params(**params)

        loss, _, _ = get_loss(
            self.X,
            self.y,
            self.ml_model,
            self.resampler,
            score = self.objectiveScore,
            confidenceLevel = confidenceLevel
        )

        self.objectiveTarget.append(loss)

        # Update GP        
        self.objectiveGP.fit(
            self.objectiveInput, 
            self.objectiveTarget
        )

    def _optimisation_step(self, confidenceLevel):

        nextGuess = self.next_guess()

        self.evaluate(
            nextGuess,
            confidenceLevel=confidenceLevel
        )

    def run_optimisation(
        self,
        resampler = Bootstrap(),
        confidenceLevel = 0.95
    ):

        self.resampler = resampler(len(self.X))

        self.initialise(
            resampler=resampler,
            confidenceLevel=confidenceLevel
        )

        print('Starting optimisation loop ---')
        for _ in range(self.initBudget, self.totalBudget):

            self._optimisation_step(confidenceLevel)


