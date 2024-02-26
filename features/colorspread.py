from features import Feature
from features.featutils import FeatUtils
from rendering.renderoutput import RenderOutput
import numpy as np
import glm
from itertools import combinations

class ColorSpread(Feature):

    def getName(self):
        return 'ColorSpread'

    def compute(self, renderOut : RenderOutput):
        # mean squared euclidean distance between all isosurface color pairs

        if renderOut.noIsosurfaces == 1:
            return 1

        # we use an explicit implementation to convert RGB to CIELab
        # we want to avoid the dependency on skimage
        # the results are the same as calling skimage.color.rgb2lab(col, illuminant = 'D65')
        # also, the in-house implementation is twice as fast as the skimage one!

        labColors = [FeatUtils.rgb2lab(col) for col in renderOut.isoColors]
        labColors = [np.array([c[0]/100, (c[1]+128)/256, (c[2]+128)/256]) for c in labColors]

        colorPairs = list(combinations(labColors, 2))
        return sum([glm.length2(colors[0] - colors[1]) for colors in colorPairs])/len(colorPairs)



        
        
        

        
