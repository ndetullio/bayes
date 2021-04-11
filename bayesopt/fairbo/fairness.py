import numpy as np

def get_dsp(sensitiveFeature, prediction):

    if not isinstance(sensitiveFeature, np.ndarray):
        sensitiveFeature = np.asarray(sensitiveFeature)

    if not isinstance(prediction, np.ndarray):
        prediction = np.asarray(prediction)
    
    mask = prediction > 0

    numPositive = np.sum(mask)
    Ppp = sensitiveFeature[mask][sensitiveFeature[mask]]/numPositive
    Ppn = sensitiveFeature[mask][~sensitiveFeature[mask]]/numPositive

    return abs(Ppp-Ppn)