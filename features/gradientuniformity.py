from features import IsosurfaceMapFeature, FeatUtils 
from rendering.renderoutput import RenderOutput


class GradientUniformity(IsosurfaceMapFeature):
    def __init__(self, isoIdx = None, histSize = 128):
        super().__init__(isoIdx)
        self.histSize = histSize

    def getName(self):
        return 'GradientUniformity'

    def computeIsosurfaceScore(self, renderOut : RenderOutput, isoIdx : int):
        self.featMap = renderOut.gradOrientationMaps[isoIdx]
        featMapMask = renderOut.stencils[isoIdx]
        self.hist = FeatUtils.computeHistogram(self.featMap, self.histSize, featMapMask, True)
        return 1.0 - FeatUtils.computeUniformity(self.hist)

