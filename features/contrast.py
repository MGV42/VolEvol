from features import Feature
from rendering.renderoutput import RenderOutput
import numpy as np

class Contrast(Feature):

    def getName(self):
        return 'Constrast'

    def compute(self, renderOut : RenderOutput):

        stencilMask = self.getCommonStencil(renderOut)
        maskedImg = self.applyMask(renderOut.image, stencilMask)

        if maskedImg.size == 0:
            return 0

        # determine RMS contrast (standard deviation of pixel intensities)
        imgIntensities = np.mean(maskedImg[:,:3]/255, axis = 1)
        meanIntensity = np.mean(imgIntensities)

        return 2*np.sqrt(np.mean((imgIntensities - meanIntensity)**2))

