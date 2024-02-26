from constraints import Constraint
from rendering.renderoutput import RenderOutput

class LowVisibilityConstraint(Constraint):
    # filter out images with poorly-visible isosurfaces
    def __init__(self, lowerBound):
        super().__init__(lowerBound)

    def getName(self):
        return 'LowVisibilityConstraint'

    def check(self, renderOut : RenderOutput):
        # check if all isosurfaces with positive visibility are "visible enough"
        for i in range(renderOut.noIsosurfaces):
            if renderOut.visibilities[i] > 0 and renderOut.visibilities[i] < self.lowerBound:
                return False
        return True

    

