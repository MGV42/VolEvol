from constraints import Constraint
from features import Pique
from rendering.renderoutput import RenderOutput

class LowPiqueConstraint(Constraint):
    # filter images that have low Pique Scores
    def __init__(self, lowerBound):
        super().__init__(lowerBound)
        self.piqueFeature = Pique()
    
    def getName(self):
        return 'LowPiqueConstraint'

    def check(self, renderOut : RenderOutput):
        piqueVal = self.piqueFeature.compute(renderOut)
        return self.checkBoundaries(piqueVal)


