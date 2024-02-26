from features import Feature
from rendering.renderoutput import RenderOutput
import numpy as np

class MaxVisibility(Feature):
    # visibility of most visible isosurface

    def getName(self):
        return 'MaxVisibility'

    def compute(self, renderOut : RenderOutput):
        maxVisIdx = np.argmax(renderOut.visibilities)
        self.featMap = renderOut.visibilityMaps[maxVisIdx]
        return np.max(renderOut.visibilities)

    def getFeatureMap(self):
        return self.featMap
