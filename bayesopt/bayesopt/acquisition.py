from warnings import catch_warnings, simplefilter
from scipy.stats import norm
import numpy as np

class EiAcquisition:

    def __init__(self, numNewSamples=100):

        self.numNewSamples = numNewSamples

    def __call__(self, modelGP, Space, knownX):

        mu = predictor(modelGP, knownX, returnStd = False)
        muOptimum = np.min(mu)

        newX = Space.sample(
            numSamples = self.numNewSamples, 
            returnSample = True
        )
        mu, std = predictor(modelGP, newX, returnStd = True)
        
        improvement = np.zeros_like(mu)
        mask = std > 0
        improve = muOptimum - mu[mask]
        scaled = improve / std[mask]
        cdf = norm.cdf(scaled)
        pdf = norm.pdf(scaled)
        exploit = improve * cdf
        explore = std[mask] * pdf
        improvement[mask] = exploit + explore

        nextGuess = newX[np.argmax(improvement),]
        
        return nextGuess

def predictor(modelGP, X, returnStd = True):
    
    with catch_warnings():
        # ignore warnings
        simplefilter("ignore")
        return modelGP.predict(X, return_std=returnStd)

def get_next_guess(
    modelGP, 
    currentX, 
    Space, 
    acquisition = EiAcquisition()
):
    
    nextGuess = acquisition(
        modelGP, 
        Space, 
        currentX)
    
    return nextGuess
