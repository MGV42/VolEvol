from constraints import Constraint
from features import Brightness
from rendering.renderoutput import RenderOutput

class BrightnessConstraint(Constraint):
    def __init__(self, lowerBound = None, upperBound = None):
        super().__init__(lowerBound, upperBound)
        self.brightnessFeature = Brightness()
    
    def getName(self):
        return 'BrightnessConstraint'

    def check(self, renderOut : RenderOutput):
        brightnessVal = self.brightnessFeature.compute(renderOut)
        return self.checkBoundaries(brightnessVal)
        


