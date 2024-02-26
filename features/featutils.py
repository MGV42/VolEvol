import numpy as np
from scipy.signal import butter, filtfilt

class FeatUtils:

    @staticmethod
    def getStencilUnion(stencils):
        stencilUnion = stencils[0]
        if len(stencils) > 1:
            stencilUnion = np.bitwise_or(stencilUnion, stencils[i])
        return stencilUnion

    @staticmethod
    def computeHistogram(featMap, histSize, featMapMask = None, denoise = True):
        featVals = featMap[featMapMask > 0] if featMapMask is not None else featMap
        if featVals.size == 0: 
            return np.zeros(histSize)

        histVals, _ = np.histogram(featVals, bins = histSize, density = True)
        histVals = histVals / np.sum(histVals)

        if denoise:
            filterOrder = 3
            filterCutoff = 0.1
            b, a = butter(filterOrder, filterCutoff)
            histVals = filtfilt(b, a, histVals)

        return histVals

    @staticmethod
    def computeEntropy(vals):
        # entropy of distribution provided by vals
        nonZeroVals = vals[vals > 0]
        nonZeroValsCount = len(nonZeroVals)
        if nonZeroValsCount == 0: return 1.0
        ent = -np.sum(nonZeroVals * np.log(nonZeroVals))
        if nonZeroValsCount > 1: ent /= np.log(nonZeroValsCount)
        return ent

    @staticmethod
    def computeUniformity(vals):
        # determine the "flatness" of the distribution provided by vals
        meanVal = np.mean(vals)
        return np.sqrt(np.sum((vals - meanVal)**2))
        
    @staticmethod
    def getMaxBin(histVals):
        # bin corresponding to histogram max value
        return np.argmax(histVals) / len(histVals)

    @staticmethod
    def rgb2gray(img): # about 2-3 times faster than the opencv implementation from cvtColor
        return 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]

    @staticmethod
    def rgb2lab(inputColor) :
       # colors should be normalized!
       num = 0
       RGB = [0, 0, 0]

       for value in inputColor :
           #value = float(value) / 255

           if value > 0.04045 :
               value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
           else :
               value = value / 12.92

           RGB[num] = value * 100
           num = num + 1

       XYZ = [0, 0, 0,]

       X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
       Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
       Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
       XYZ[ 0 ] = round( X, 4 )
       XYZ[ 1 ] = round( Y, 4 )
       XYZ[ 2 ] = round( Z, 4 )

       XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
       XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
       XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

       num = 0
       for value in XYZ :

           if value > 0.008856 :
               value = value ** ( 0.3333333333333333 )
           else :
               value = ( 7.787 * value ) + ( 16 / 116 )

           XYZ[num] = value
           num = num + 1

       Lab = [0, 0, 0]

       L = ( 116 * XYZ[ 1 ] ) - 16
       a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
       b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

       Lab [ 0 ] = round( L, 4 )
       Lab [ 1 ] = round( a, 4 )
       Lab [ 2 ] = round( b, 4 )

       return Lab
