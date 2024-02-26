from features import Feature
from rendering.renderoutput import RenderOutput
import numpy as np

class SpecificVisibility(Feature):
    # visibility of an isosurface with a specified index

    def getName(self):
        return 'SpecificVisibility'

    def __init__(self, isosurfaceIdx = 0):
        self.isosurfaceIdx = isosurfaceIdx

    def compute(self, renderOut : RenderOutput):
        idx = min(self.isosurfaceIdx, renderOut.noIsosurfaces-1)
        self.featMap = renderOut.visibilityMaps[idx]
        return renderOut.visibilities[idx]

    def getFeatureMap(self):
        return self.featMap
