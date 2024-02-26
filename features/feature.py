from abc import abstractmethod
from rendering.renderoutput import RenderOutput
import numpy as np

class Feature:
    
    @abstractmethod 
    def compute(self, renderOut : RenderOutput):
        # determine a score based on data from renderer output
        # score should be determinted such that:
        # - values are from [0, 1]
        # - lower values are "better" (the optimization goal should be to MAXIMIZE the feature score)
        pass
    
    @abstractmethod
    def getName(self):
       pass

    # for features that are determined from histograms
    def getHistogram(self):
        return None

    # for features that are determined from feature maps
    def getFeatureMap(self):
        return None

    def getCommonStencil(self, renderOut : RenderOutput):
        # union of all stencils
        stencilUnion = renderOut.stencils[0]
        if renderOut.noIsourfaces > 1:
            for i in range(1, renderOut.noIsosurfaces):
                stencilUnion = np.bitwise_or(stencilUnion, renderOut.stencils[i])
        return stencilUnion;

    def applyMask(self, arr, mask):
        return arr[mask > 0]