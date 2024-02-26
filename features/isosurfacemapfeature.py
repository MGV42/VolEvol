from abc import abstractmethod
from features import Feature
from rendering.renderoutput import RenderOutput
import numpy as np

# base class of features based on processing isosurface feature maps

class IsosurfaceMapFeature(Feature):
    def __init__(self, isoIdx = None):
        self.isoIdx = isoIdx
        self.featMap = None
        self.hist = None
        
    @abstractmethod
    def computeIsosurfaceScore(self, isoIdx):
        # compute feature score for isosurface at isoIdx
        pass

    def compute(self, renderOut : RenderOutput):
        if self.isoIdx is not None and self.isoIdx >= 0 and self.isoIdx < renderOut.noIsosurfaces:
            # return property of isosurface at isoIdx
            return self.computeIsosurfaceScore(renderOut, self.isoIdx)

        # return visibility-weighted mean score of all isosurfaces
        visSum = np.sum(renderOut.visibilities)
        if visSum == 0: return 0
        return np.sum([self.computeIsosurfaceScore(renderOut, i) * renderOut.visibilities[i] 
                        for i in range(renderOut.noIsosurfaces)])/visSum

    def getHistogram(self):
        return self.hist

    def getFeatureMap(self):
        return self.featMap

