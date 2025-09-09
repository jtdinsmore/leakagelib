import numpy as np
from scipy.interpolate import interp1d
from tensorflow.keras import models
from ..settings import LEAKAGE_DATA_DIRECTORY

def load_tensorflow_model():
    """Get the tensorflow model. This code will fail if your tensorflow is not up to date. The NN was trained with tensorflow version 2.18.0. It should be compatible with many other versions."""
    try:
        return models.load_model(f"{LEAKAGE_DATA_DIRECTORY}/bkg-nn/model1-None.keras")
    except:
        return models.load_model(f"{LEAKAGE_DATA_DIRECTORY}/bkg-nn/model1-None.h5")
    
class Model:
    def __init__(self):
        self.model = load_tensorflow_model()
        
    def __call__(self, tracks):
        return self.model(tracks).numpy()[:,0]
    
def account_for_prior(bg_probs):
    ptcl = np.mean(bg_probs)
    return ptcl*bg_probs / (ptcl*bg_probs + (1 - ptcl)*(1 - bg_probs))