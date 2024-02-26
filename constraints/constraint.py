from abc import ABC, abstractmethod
from rendering.renderoutput import RenderOutput

class Constraint:
    def __init__(self, lowerBound = None, upperBound = None):
        self.lowerBound = lowerBound
        self.upperBound = upperBound

    @abstractmethod 
    def getName(self):
        pass

    @abstractmethod
    def check(self, renderOut : RenderOutput):
        pass

    def checkBoundaries(self, val):
        if self.lowerBound is None:
            if self.upperBound is None or val <= self.upperBound:
                return True
        elif val >= self.lowerBound:
                if self.upperBound is None or val <= self.upperBound:
                    return True
        return False



