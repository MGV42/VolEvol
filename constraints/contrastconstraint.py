from constraints import Constraint
from features import Contrast
from rendering.renderoutput import RenderOutput

class ContrastConstraint(Constraint):
    def __init__(self, lowerBound = None, upperBound = None):
        super().__init__(lowerBound, upperBound)
        self.contrastFeature = Contrast()

    def getName(self):
        return 'ContrastConstraint'

    def check(self, renderOut : RenderOutput):
        contrastVal = self.contrastFeature.compute(renderOut)
        return self.checkBoundaries(contrastVal)


