# based on https://github.com/michael-rutherford/pypiqe
import numpy as np
import scipy.ndimage
from features.featutils import FeatUtils

class PiqueMethod:
    def __init__(self):
        self.blockSize = 16 # size of block to be analyzed
        self.activityThreshold = 0.1 # threshold for identifying high spatially-prominent blocks
        self.blockImpairedThreshold = 0.1 # threshold for identifying blocks with noticeable artifacts
        self.windowSize = 6 # segment size in a block edge
        self.nSegments = self.blockSize - self.windowSize + 1 # number of segments for each block edge

    def getScore(self, img): # note: img should be in range 0-255

        ipImage = FeatUtils.rgb2gray(img)

        # pad if image size is not divisible by blockSize 
        originalSize = ipImage.shape  # Actual image size
        rows, columns = originalSize[:2]
        rowsPad = rows % self.blockSize
        columnsPad = columns % self.blockSize
        isPadded = False
        if rowsPad > 0 or columnsPad > 0:
            if rowsPad > 0:
                rowsPad = self.blockSize - rowsPad
            if columnsPad > 0:
                columnsPad = self.blockSize - columnsPad
            isPadded = True
            ipImage = np.pad(ipImage, ((0, rowsPad), (0, columnsPad)), mode='symmetric')

        distBlockScores = 0 # accumulation of distorted block scores
        NHSA = 0 # number of high spatial active blocks

        # Normalize image to zero mean and ~unit std
        # used circularly-symmetric Gaussian weighting function sampled out 
        # to 3 standard deviations.
        mu = self.gaussBlur(ipImage)
        sigma = np.sqrt(np.abs(self.gaussBlur(ipImage * ipImage) - mu * mu))
        imnorm = (ipImage - mu) / (sigma + 1)

        # Preallocation for masks
        NoticeableArtifactsMask = np.zeros_like(imnorm, dtype=bool)
        NoiseMask = np.zeros_like(imnorm, dtype=bool)
        ActivityMask = np.zeros_like(imnorm, dtype=bool)

        # Start of block by block processing
        for i in range(0, imnorm.shape[0], self.blockSize):
            for j in range(0, imnorm.shape[1], self.blockSize):

                # Weights Initialization
                WNDC = 0
                WNC = 0

                # Compute block variance
                Block = imnorm[i:i+self.blockSize, j:j+self.blockSize]
                blockVar = np.var(Block, ddof=1)

                # Considering spatially prominent blocks 
                if blockVar > self.activityThreshold:
                    ActivityMask[i:i+self.blockSize, j:j+self.blockSize] = True
                    WHSA = 1
                    NHSA += 1

                    # Analyze Block for noticeable artifacts
                    blockImpaired = self.noticeDistCriterion(Block, 
                                                             self.nSegments, 
                                                             self.blockSize-1, 
                                                             self.windowSize, 
                                                             self.blockImpairedThreshold, 
                                                             self.blockSize)

                    if blockImpaired:
                        WNDC = 1
                        NoticeableArtifactsMask[i:i+self.blockSize, j:j+self.blockSize] = True

                    # Analyze Block for Gaussian noise distortions
                    blockSigma, blockBeta = self.noiseCriterion(Block, self.blockSize-1, blockVar)

                    if blockSigma > 2 * blockBeta:
                        WNC = 1
                        NoiseMask[i:i+self.blockSize, j:j+self.blockSize] = True

                    # Pooling/ distortion assignment
                    distBlockScores += WHSA * WNDC * (1 - blockVar) + WHSA * WNC * blockVar
                    a='a'

        # Quality score computation
        # C is a positive constant, it is included to prevent numerical instability
        C = 1
        Score = ((distBlockScores + C) / (C + NHSA)) * 100

        # if input image is padded then remove those portions from ActivityMask,
        # NoticeableArtifactsMask and NoiseMask and ensure that size of these masks
        # are always M-by-N.
        if isPadded:
            NoticeableArtifactsMask = NoticeableArtifactsMask[:originalSize[0], :originalSize[1]]
            NoiseMask = NoiseMask[:originalSize[0], :originalSize[1]]
            ActivityMask = ActivityMask[:originalSize[0], :originalSize[1]]

        return Score

    def gaussBlur(self, img):
        # opencv version
        #return cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=7/6, borderType=cv2.BORDER_REPLICATE)

        # we use the scipy version instead of the opencv version to avoid the opencv dependency
        # however, the opencv version is noticeably faster
        return scipy.ndimage.gaussian_filter(img, sigma = 7/6, radius = 3, mode = 'reflect')

    

    # Function to analyze block for Gaussian noise distortions
    def noiseCriterion(self, Block, blockSize, blockVar):
        # Compute block standard deviation
        blockSigma = np.sqrt(blockVar)    
        # Compute ratio of center and surround standard deviation
        cenSurDev = self.centerSurDev(Block, blockSize)    
        # Relation between center-surround deviation and the block standard deviation
        blockBeta = np.abs(blockSigma - cenSurDev) / np.maximum(blockSigma, cenSurDev)
    
        return blockSigma, blockBeta

    # Function to compute center surround Deviation of a block
    def centerSurDev(self, Block, blockSize):
        # block center
        center1 = (blockSize+1)//2
        center2 = center1+1
        center = np.concatenate((Block[:, center1-1], Block[:, center2-1]), axis=0)

        # block surround
        Block = np.delete(Block, (center1-1), axis=1)
        Block = np.delete(Block, (center2-1), axis=1)

        # Compute standard deviation of block center and block surround
        center_std = np.std(center, ddof=1)
        surround_std = np.std(Block, ddof=1)
    
        # Ratio of center and surround standard deviation
        cenSurDev = center_std / surround_std
    
        # Check for nan's
        if np.isnan(cenSurDev):
            cenSurDev = 0
    
        return cenSurDev

    # Function to analyze block for noticeable artifacts
    def noticeDistCriterion(self, Block, nSegments, blockSize, windowSize, blockImpairedThreshold, N):

        # Top edge of block
        topEdge = Block[0, :]
        segTopEdge = self.segmentEdge(topEdge, nSegments, blockSize, windowSize)

        # Right side edge of block
        rightSideEdge = Block[:, N-1]
        rightSideEdge = rightSideEdge.T
        segRightSideEdge = self.segmentEdge(rightSideEdge, nSegments, blockSize, windowSize)

        # Down side edge of block
        downSideEdge = Block[N-1, :]
        segDownSideEdge = self.segmentEdge(downSideEdge, nSegments, blockSize, windowSize)

        # Left side edge of block
        leftSideEdge = Block[:, 0]
        leftSideEdge = leftSideEdge.T
        segLeftSideEdge = self.segmentEdge(leftSideEdge, nSegments, blockSize, windowSize)

        # Compute standard deviation of segments in left, right, top and down side edges of a block
        segTopEdge_stdDev = np.std(segTopEdge, axis=1, ddof=1)
        segRightSideEdge_stdDev = np.std(segRightSideEdge, axis=1, ddof=1)
        segDownSideEdge_stdDev = np.std(segDownSideEdge, axis=1, ddof=1)
        segLeftSideEdge_stdDev = np.std(segLeftSideEdge, axis=1, ddof=1)

        # Check for segment in block exhibits impairedness, if the standard deviation of the segment is less than blockImpairedThreshold.
        blockImpaired = 0
        for seg_index in range(segTopEdge.shape[0]):
            if ((segTopEdge_stdDev[seg_index] < blockImpairedThreshold) or 
                (segRightSideEdge_stdDev[seg_index] < blockImpairedThreshold) or 
                (segDownSideEdge_stdDev[seg_index] < blockImpairedThreshold) or 
                (segLeftSideEdge_stdDev[seg_index] < blockImpairedThreshold)):
                blockImpaired = 1
                break

        return blockImpaired

    # Function to segment block edges
    def segmentEdge(self, blockEdge, nSegments, blockSize, windowSize):
        # Segment is defined as a collection of 6 contiguous pixels in a block edge
        segments = np.zeros((nSegments, windowSize))
        for i in range(nSegments):
            segments[i, :] = blockEdge[i: windowSize]
            if windowSize <= (blockSize + 1):
                windowSize += 1
        return segments








