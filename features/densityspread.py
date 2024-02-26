from features import Feature
from rendering.renderoutput import RenderOutput
import numpy as np

class DensitySpread(Feature):

    def getName(self):
        return 'DensitySpread'

    def compute(self, renderOut : RenderOutput):
        meanDensity = np.mean(renderOut.isoDensities)
        return np.sum((renderOut.isoDensities - meanDensity)**2)/meanDensity
