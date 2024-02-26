from features import Feature
from rendering.renderoutput import RenderOutput
import numpy as np

class Brightness(Feature):
    
    def getName(self):
        return 'Brightness'

    def compute(self, renderOut : RenderOutput):

        stencilMask = self.getCommonStencil(renderOut)
        maskedImg = self.applyMask(renderOut.image, stencilMask) / 255

        if maskedImg.size == 0:
            return 0

        # coefficients for perceptual brightness as deduced by https://alienryderflex.com/hsp.html
        rCoef = 0.299
        gCoef = 0.587
        bCoef = 0.114
        rChannel = maskedImg[:,0]
        gChannel = maskedImg[:,1]
        bChannel = maskedImg[:,2]

        return np.mean(np.sqrt(rCoef * rChannel**2 + bCoef * bChannel**2 + gCoef * gChannel**2))

    

        
