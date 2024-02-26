from features import Feature
from rendering.renderoutput import RenderOutput
import numpy as np
from scipy.signal import convolve2d
from features.featutils import FeatUtils

class NoiseInvariance(Feature):
    def __init__(self):
        self.kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])

    def getName(self):
        return 'NoiseVariance'

    def compute(self, renderOutput : RenderOutput):
        img = FeatUtils.rgb2gray(renderOutput.image)
        h, w = img.shape
        sigma = np.sum(np.sum(np.absolute(convolve2d(img, self.kernel))))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w-2) * (h-2))
        if sigma > 10.0: sigma = 10.0
        return 1.0 - sigma / 10.0



