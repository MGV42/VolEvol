import glm
import numpy as np

class CP:
    def __init__(self, x : float, rgbaVal : glm.vec4):
        self.x = x
        self.rgbaVal = rgbaVal

    def xa(self) -> glm.vec2:
        return glm.vec2(self.x, self.rgbaVal.a)

    def __str__(self):
        return f'CP[{self.x}] : ({self.rgbaVal.r}, {self.rgbaVal.g}, {self.rgbaVal.b}, {self.rgbaVal.a}'

class TransFunc:

    defaultCPColor = glm.vec3(0.8)

    def controlPointsFromArray(self, cpArray):
        '''
        sets control points from an array formatted as:
       [x0, r0, g0, b0, a0, x1, r1, g1, b1, a1, ...]
        ''' 
        self.cp = [CP(x, glm.vec4(r, g, b, a)) 
                   for [x, r, g, b, a] in cpArray.reshape((-1, 5))] 

    def noControlPoints(self):
        return len(self.cp)

    def controlPointsToArray(self):
        cpArr = np.array([[p.x, p.rgbaVal.r, p.rgbaVal.g, p.rgbaVal.b, p.rgbaVal.a] for p in self.cp])
        return cpArr.reshape(-1)

    def __init__(self, noMaxControlPoints = 7):
        self.cp = [CP(0.2, glm.vec4(0.9, 0.3, 0.0, 0.3)),
                   #CP(0.3, glm.vec4(0.6, 0.5, 0.1, 0.3)),
                   CP(0.7, glm.vec4(0.7, 1.0, 1.0, 0.9))]
        self.interpolationFunc = TransFunc.smoothstep_1
        self.noMaxControlPoints = noMaxControlPoints
        self.noValuesPerControlPoint = 5 # modify this if necessary

    def __str__(self):
        tfStr=''
        for p in self.cp:
            tfStr += str(p) + '\n'
        return tfStr

    def findRightIdx(self, x):
        # find index of point to the right of coord x
        for i in range(self.noControlPoints()):
            if self.cp[i].x > x:
                return i
        return -1 # x is after last control point

    def addCP(self, x : float, rgbaVal : glm.vec4):
        # add new control point so that all control points are sorted by x
        
        rightIdx = self.findRightIdx(x)
        
        if rightIdx == -1:
            # add control point after last one
            self.cp.append(CP(x, rgbaVal))
            return self.noControlPoints()-1
                           
        # add control point at beginning or in middle positions
        self.cp.insert(rightIdx, CP(x, rgbaVal))
        return rightIdx

    def removeCP(self, cpIdx):
        # remove control point at index idx
        self.cp.pop(cpIdx)
    
    def comb(n, k):
        return np.factorial(n) / (np.factorial(k) * np.factorial(n-k))

    def linear_0(u):
        return u
    
    def smoothstep_1(u):
        return 3*u**2 - 2*u**3

    def smoothstep_2(u):
        return 6*u**5 - 15*u**4 + 10*u**3

    def smoothstep_3(u):
        return -20*u**7 + 70*u**6 - 84*u**5 + 35*u**4

    def smoothstep_n(u, n):
        return u**(n+1) * sum([TFtransFunc.comb(n+k, k) * TransFunc.comb(2*n+1, n-k) * (-u)**k for k in range(n+1)])

    # in future work: should preallocate list of samples and use np arrays!
    def getData(self, n):
        # take n samples from the transFunc at equidistant intervals on x
        
        samples = []

        if self.noControlPoints == 0:
            return np.zeros(4*n, dtype = 'uint8') # should change this if color is specified in ways other than RGBA values

        dx = 1/(n-1)
        x = [i*dx for i in range(n)] 

        for i in range(n):
            if x[i] < self.cp[0].x:
                yrgba = glm.vec4(self.cp[0].rgbaVal.rgb, 0)                
            elif x[i] >= self.cp[-1].x:
                yrgba = self.cp[-1].rgbaVal
            else:    
                rIdx = self.findRightIdx(x[i]) # inefficient for a sequence of ordered equidistant points, could be faster
                p0 = self.cp[rIdx-1] # control point to the left of x
                p1 = self.cp[rIdx] # control point to the right of x
                u = (x[i] - p0.x) / (p1.x - p0.x)
                v = self.interpolationFunc(u)
                yrgba = p0.rgbaVal + v * (p1.rgbaVal - p0.rgbaVal)

            # samples.append(CP(x[i], yrgba)) # this list is only for drawing even samples
            # for rendering based on transfer function only a list of rgba values is needed
            samples.extend([yrgba.r, yrgba.g, yrgba.b, yrgba.a])

        return (np.array(samples) * 255).astype(np.uint8)

    def getLinearSegmentLengths(self):
        # get linear length of each segment
        noCPs = self.noControlPoints()
        if noCPs <= 1: return [0]
        else:
            return [glm.length(self.cp[i].xa() - self.cp[i+1].xa()) for i in range(noCPs-1)]

    def samplesForDrawing(self, resolution):
        # take samples suitable for drawing transFunc
        # resolution is a parameter used to decide the number of samples for each per spline segment
        # depending on its linear length

        if self.noControlPoints() == 0:
            return [CP(0, glm.vec4(255, 255, 255, 0)), 
                    CP(1, glm.vec4(255, 255, 255, 0))]
        
        samples = []

        if self.cp[0].x > 0:
            samples.extend([CP(0, glm.vec4(self.cp[0].rgbaVal.rgb, 0)), 
                            CP(self.cp[0].x, glm.vec4(self.cp[0].rgbaVal.rgb, 0))])

        if self.noControlPoints() > 1:

            # linear lengths of spline segments inbetween control points
            cpLengths = self.getLinearSegmentLengths()
            totalLength = sum(cpLengths)
            cpLengths = [cpl / totalLength for cpl in cpLengths] # normalization

            samplesPerSegment = [round(resolution * cplen) for cplen in cpLengths]

            for i in range(len(self.cp)-1):
                p0 = self.cp[i]
                p1 = self.cp[i+1]
            
                samples.append(p0)
            
                if samplesPerSegment[i] > 0:
                    du = 1/samplesPerSegment[i]
                    uVals = [j*du for j in range(1, samplesPerSegment[i])]
                    vVals = [self.interpolationFunc(u) for u in uVals]

                    uVals = [p0.x + u*(p1.x - p0.x) for u in uVals]
                    vVals = [p0.rgbaVal + v*(p1.rgbaVal - p0.rgbaVal) for v in vVals]

                    samplesBetweenCPs = [CP(u, v) for u, v in zip(uVals, vVals)] 
                    samples.extend(samplesBetweenCPs)

        samples.append(self.cp[-1])

        if self.cp[-1].x < 1:
            samples.append(CP(1, self.cp[-1].rgbaVal))

        return samples

