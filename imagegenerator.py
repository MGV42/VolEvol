from rendering.renderer import VolumeRenderer
from evolutionary import MapElites
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class ImageGenerator:
    def __init__(self, renderer):

        self.imageWidth = 512
        self.imageHeight = 512
        self.noDesiredImages = 25
        self.showGeneratedImages = True
        self.outputDir = None

        self.renderer = renderer

        self.allowInfeasibleImages = True # allow infeasible solutions to be rendered into image set

    def updateParameters(self, paramDict):
        self.imageWidth = paramDict['imageWidth']['value']
        self.imageHeight = paramDict['imageHeight']['value']
        self.noDesiredImages = paramDict['noDesiredImages']['value']
        self.showGeneratedImages = paramDict['showGeneratedImages']['value']
        self.outputDir = paramDict['outputDir']['value']

    def drawImageSet(self, solutionSet):
        # render images from solutions provided by evolutionary component
        prevW = self.renderer.w
        prevH = self.renderer.h
        self.renderer.setViewportSize(self.imageWidth, self.imageHeight)
        
        imageSet = []
        for sol in solutionSet:
            renderOut = self.renderer.drawChrom(sol)
            imageSet.append(renderOut.image)

        self.renderer.setViewportSize(prevW, prevH)

        return imageSet

    def saveImageSet(self, imageSet, outputDir):
        # save images in imageSet to folder in outputPath
        if outputDir:
            dt = datetime.now()
            dtStr = f'{dt.year}{dt.month}{dt.day}_{dt.hour}{dt.minute}'
            for i in range(len(imageSet)):
                imgName = dtStr + f'_{i}.png'
                if outputDir[-1] not in ['/', '\\']: outputDir += '/'
                plt.imsave(outputDir + imgName, imageSet[i][:,:,:3], origin = 'lower')

    def showImageSet(self, imageSet, noCols = None, title = 'Images'):
        # show images in pyplot window
        noImages = len(imageSet)
        if noCols is not None:
            noRows = noImages // noCols
            if noCols * noRows < noImages:
                noRows += 1
        else:
            noCols = round(np.sqrt(noImages))
            noRows = noImages // noCols
            if noCols * noRows < noImages:
                noCols += 1
            
        plt.figure(title)
        idx = 0
        for col in range(noCols):
            for row in range(noRows):
                if idx >= noImages: break
                idx += 1
                plt.subplot(noRows, noCols, idx)
                plt.imshow(imageSet[idx-1], origin='lower')
                plt.axis('off')
        plt.tight_layout(pad = 0.1)
        plt.show()

    def imagesFromSolutions(self, feasibleSolutions, infeasibleSolutions):
        # actual number of images is NOT guarranteed to be noDesiredImages
        # could be lower, depending on the numper of available solutions
        
        noFeasibleSolutions = len(feasibleSolutions)
        noInfeasibleSolutions = len(infeasibleSolutions)

        # solution set made up of solutions from the archive
        # prioritize feasible solutions

        if noFeasibleSolutions == self.noDesiredImages:
            solutionSet = feasibleSolutions
        elif noFeasibleSolutions > self.noDesiredImages:
            #randomly choose from unconstrained solutions
            idx = np.random.choice(range(noFeasibleSolutions), self.noDesiredImages, False)
            solutionSet = feasibleSolutions[idx]
        else:
            # the number of feasible solutions is less than the number of desired ones
            solutionSet = feasibleSolutions
            if self.allowInfeasibleImages:
                # fill remaining solutions with non-feasible ones until noDesiredImages
                noSlotsLeft = self.noDesiredImages - noFeasibleSolutions
                if noInfeasibleSolutions <= noSlotsLeft:
                    if noFeasibleSolutions > 0:
                        solutionSet = np.vstack((solutionSet, infeasibleSolutions))
                    else:
                        solutionSet = infeasibleSolutions
                else:
                    idx = np.random.choice(range(noInfeasibleSolutions), noSlotsLeft, False)
                    if noFeasibleSolutions > 0:
                        solutionSet = np.vstack((solutionSet, infeasibleSolutions[idx]))
                    else: 
                        solutionSet = infeasibleSolutions[idx]

        # get images from solutions
        imageSet = self.drawImageSet(solutionSet)

        return imageSet
        


    