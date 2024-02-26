from features import IsosurfaceMapFeature, FeatUtils
from rendering.renderoutput import RenderOutput

class CurvatureEntropy(IsosurfaceMapFeature):
    def __init__(self, isoIdx = None, histSize = 5):
        super().__init__(isoIdx)
        self.histSize = histSize

    def getName(self):
        return 'CurvatureEntropy'

    def computeIsosurfaceScore(self, renderOut : RenderOutput, isoIdx : int):
        self.featMap = renderOut.curvatureMaps[isoIdx]
        featMapMask = renderOut.stencils[isoIdx]
        self.hist = FeatUtils.computeHistogram(self.featMap, self.histSize, featMapMask, False)
        return 1.0 - FeatUtils.computeEntropy(self.hist)
