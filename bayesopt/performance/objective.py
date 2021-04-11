from sklearn.utils import resample
from sklearn.metrics import accuracy_score

import numpy as np
from collections import namedtuple

# class ReIterator:
#     def __init__(self, iterator_factory):
#         self.iterator_factory = iterator_factory

#     def __iter__(self):
#         return self.iterator_factory()

class Bootstrap:

    def __init__(
        self, 
        sampleProportion = 0.8,
        numRounds = 5,
        randomState = None
    ):

        self.sampleProportion = sampleProportion
        self.numRounds = numRounds
        self.randomState = randomState

    def __call__(self, numRows):

        numSamples = int(numRows*self.sampleProportion)
        indices = np.arange(numRows)

        sampler = []
        sample = namedtuple('Sample', 'insample outsample')
        for i in range(self.numRounds):
            if self.randomState is not None:
                self.randomState += 1

            insample = resample(
                indices,
                n_samples = numSamples,
                random_state = self.randomState
            )
            outsample = set(indices) - set(insample)

            sampler.append(sample(list(insample), list(outsample)))

        return sampler


# def bootstrap(numRows, randomState=None, size=0.8, numRounds=5):
    
#     numSamples = int(numRows*size)
#     indices = np.arange(numRows)
    
#     sampler = []
#     sample = namedtuple('Sample', 'insample outsample')
#     for i in range(numRounds):
#         if randomState is not None:
#             randomState += i

#         insample = resample(
#             indices, 
#             n_samples = numSamples, 
#             random_state = randomState
#         )
#         outsample = set(indices) - set(insample)
        
#         sampler.append(sample(list(insample), list(outsample)))
        
#     return sampler

def get_loss(
    X,
    y,
    model, 
    resampler,
    score = 'accuracy', 
    confidenceLevel = 0.95
):
    
    if isinstance(score, str):
        if score == 'accuracy':
            scorefunc = accuracy_score
        else:
            raise ValueError("Score must be accuracy, for now!")
            
    else:
        raise TypeError("Score must be a string, for now!")
        
    losses = []
    for it, sample in enumerate(resampler, start=1):
        
        print(f'Running bootstrap round {it}')

        Xtr = X[sample.insample, ]
        ytr = y[sample.insample]
        Xte = X[sample.outsample, ]
        yte = y[sample.outsample]
        
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        
        losses.append(1.-scorefunc(yte, ypred))
        
    p = ((1.0-confidenceLevel)/2.0) * 100
    lowerConfidenceLimit = max(0.0, np.percentile(losses, p))
    p = (confidenceLevel+((1.0-confidenceLevel)/2.0)) * 100
    upperConfidenceLimit = min(1.0, np.percentile(losses, p))
    
    return np.mean(losses), lowerConfidenceLimit, upperConfidenceLimit