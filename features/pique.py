from features import Feature
from rendering.renderoutput import RenderOutput
from features.piquemethod import PiqueMethod

class Pique(Feature):
    def __init__(self):
        self.piqueMethod = PiqueMethod()

    def getName(self):
        return "Pique"

    def compute(self, renderOut : RenderOutput):
        img = renderOut.image[:,:,:3]
        return 1.0 - self.piqueMethod.getScore(img)/100


