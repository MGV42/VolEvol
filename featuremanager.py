from features import *
from constraints import *

class FeatureManager:
    
    def __init__(self):

        self.featureFactory = {
            'Brightness' : Brightness(),
            'ColorSpread' : ColorSpread(),
            'Contrast' : Contrast(),
            'CurvatureEntropy' : CurvatureEntropy(),
            'DensitySpread' : DensitySpread(),
            'GradientShift' : GradientShift(),
            'GradientUniformity' : GradientUniformity(),
            'NoiseInvariance' : NoiseInvariance(),
            'Pique' : Pique(),
            'VisibilityBalance' : VisibilityBalance()
            }

        self.minBrightness = 0.1
        self.minContrast = 0.1
        self.minVisibility = 0.1
        self.minPique = 0.2

        self.constraintFactory = {
            'BrightnessConstraint' : BrightnessConstraint(self.minBrightness),
            'ContrastConstraint' : ContrastConstraint(self.minContrast),
            'LowVisibilityConstraint' : LowVisibilityConstraint(self.minVisibility),
            'LowPiqueConstraint' : LowPiqueConstraint(self.minPique)
            }

        self.objectiveFeatures = [
                                  self.featureFactory['Pique'], 
                                  self.featureFactory['ColorSpread'], 
                                  self.featureFactory['GradientShift']
                                 ]

        self.diversityFeatures = [
                                  self.featureFactory['CurvatureEntropy'],  
                                  self.featureFactory['DensitySpread'], 
                                  self.featureFactory['VisibilityBalance'] 
                                 ]

        self.constraints = [
                            self.constraintFactory['BrightnessConstraint'], 
                            self.constraintFactory['ContrastConstraint'], 
                            self.constraintFactory['LowVisibilityConstraint'],
                            self.constraintFactory['LowPiqueConstraint']
                           ]

    def updateParameters(self, paramDict):
        self.updateFeatures(paramDict['objectiveFeatures']['value'], 
                            paramDict['diversityFeatures']['value'])

    def updateFeatures(self, objectiveFeatureNames, diversityFeatureNames):
        self.objectiveFeatures = [self.featureFactory[feat] for feat in objectiveFeatureNames]
        self.diversityFeatures = [self.featureFactory[feat] for feat in diversityFeatureNames]

    def updateConstraints(constraintNames):
        self.constraints = [self.constraintFactory[constr] for constr in constraintNames]


