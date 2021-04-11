import numpy as np
from collections import namedtuple

def create_space(parameterSpace):

    Space = namedtuple('Space', 'lower upper type')
    space = {k: Space(t[0], t[1], t[2]) for k, t in parameterSpace.items()}

    return space

def sample_space(space, numSamples=100):
    
    X = np.ndarray((numSamples, len(space)), dtype=object)
    for ind, sp in enumerate(space.values()):
        if sp.type == 'integer':
            X[:, ind] = np.random.randint(sp.lower, sp.upper+1, size=numSamples).astype(int)
        else:
            X[:, ind] = np.random.uniform(sp.lower, sp.upper, size=numSamples)
        
    return X

class HyperSpace:

    def __init__(self, parameterSpace):

        self.space = create_space(parameterSpace)
        self.parameterNames = self.space.keys()
        self.parameterSample = None

    def sample(self, numSamples=100, returnSample=False):

        self.sampleArray = sample_space(
            self.space,
            numSamples = numSamples
        )

        if returnSample:
            return self.sampleArray

    def make_dict(self, values):

        return dict(zip(self.parameterNames, values))

