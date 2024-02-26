from features import IsosurfaceMapFeature, FeatUtils
from rendering.renderoutput import RenderOutput
import numpy as np

class GradientShift(IsosurfaceMapFeature):
    def __init__(self, isoIdx = None, histSize = 128):
        super().__init__(isoIdx)
        self.histSize = histSize

    def getName(self):
        return 'GradientShift'
        
    def computeIsosurfaceScore(self, renderOut : RenderOutput, isoIdx : int):
        self.featMap = renderOut.gradMagnitudeMaps[isoIdx]
        featMapMask = renderOut.stencils[isoIdx]
        self.hist = FeatUtils.computeHistogram(self.featMap, self.histSize, featMapMask, True)
        return FeatUtils.getMaxBin(self.hist)
    
    
