from features import Feature
from rendering.renderoutput import RenderOutput
import numpy as np

class VisibilityBalance(Feature):

    def getName(self):
        return 'VisibilityBalance'
    
    def compute(self, renderOut : RenderOutput):
        meanVis = np.mean(renderOut.visibilities)
        return 1.0 - 2*np.sqrt(np.mean((renderOut.visibilities - meanVis)**2))

