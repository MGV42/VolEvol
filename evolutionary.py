
import numpy as np
from rendering.renderer import VolumeRenderer
from ribs.archives import GridArchive, CVTArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from multiprocessing.pool import ThreadPool as WorkerPool
import time

class MapElites:
    def __init__(self, renderer : VolumeRenderer):

        self.renderer = renderer

        # features which constitute the optimization objective(s)
        self.objectiveFeatures = []

        # features which form the diversity space
        self.diversityFeatures = []

        # all features (objective + diversity)
        self.features = []

        self.constraints = []

        self.noObjectiveFeatures = 0
        self.noDiversityFeatures = 0
        self.noFeatures = 0
        self.noConstraints = 0

        # if True, uses CVT archive, else grid archive
        self.useCVTArchive = False 
        self.archiveSize = 200
        
        self.chromSize = self.renderer.getMaxChromSize()
        self.solutionBounds = [(0, 1) for i in range(self.chromSize)]
        # important: it is a bad idea to have few emitters with large batches
        # computational times may increase significantly
        self.noEmitters = 3
        self.batchSize = 10
        self.emitterMean = 0.2; # initial mean value 
        self.emitterSigma = 0.3; # standard deviation
        self.noWorkers = 1

        self.noSolutions = self.noEmitters * self.batchSize

    def updateParameters(self, paramDict):
        self.useCVTArchive = paramDict['useCVTArchive']['value']
        self.archiveSize = paramDict['archiveSize']['value']
        self.noEmitters = paramDict['noEmitters']['value']
        self.batchSize = paramDict['batchSize']['value']
        self.emitterMean = paramDict['emitterMean']['value']
        self.emitterSigma = paramDict['emitterSigma']['value']
        self.noWorkers = paramDict['noWorkers']['value']

        self.chromSize = self.renderer.getMaxChromSize()

        if self.noObjectiveFeatures > 0 and self.noDiversityFeatures > 0:
            self.setArchive()
            self.setEmitters()
            self.scheduler = Scheduler(self.archive, self.emitters)

    def factorize(self, x, n):
        # decompose x into n almost equal factors whose product is almost equal to x

        factors = np.ones(n)
        if x <= 1: return factors

        # find closest product of constant factors, greater than n:
        xprod = 1
        while xprod < x:
            factors += 1
            xprod = np.prod(factors)

        if xprod == x: return factors

        # decrease each factor, one by one, until product is smaller than x
        idx = n-1
        while xprod > x:
            factors[idx] -= 1
            xprod = np.prod(factors)
            idx -= 1
            if idx < 0: idx = n-1

        return factors
    
    def setArchive(self):
        
        ranges = [(0, 1)] * self.noDiversityFeatures 
        if self.useCVTArchive:
            self.archive = CVTArchive(solution_dim = self.chromSize, 
                                      cells = self.archiveSize,
                                      ranges = ranges)
        else: # grid archive
            dims = self.factorize(self.archiveSize, self.noDiversityFeatures)
            self.archive = GridArchive(solution_dim = self.chromSize, 
                                       dims = dims, 
                                       ranges = ranges)

    def setEmitters(self):
        self.emitters = [
            EvolutionStrategyEmitter(
                archive = self.archive, 
                x0 = np.full(self.chromSize, self.emitterMean),
                sigma0 = self.emitterSigma,
                bounds = self.solutionBounds,
                batch_size = self.batchSize
                )
            for _ in range(self.noEmitters)
            ]

        self.noSolutions = self.noEmitters * self.batchSize
        self.renderOutputs = [None] * self.noSolutions

    def setFeatures(self, objectiveFeatures, diversityFeatures):
        
        # objective features and diversity features are computed similarly,
        # therefore it is more efficient to keep them all in the same array
        self.noObjectiveFeatures = len(objectiveFeatures)
        self.noDiversityFeatures = len(diversityFeatures)
        self.noFeatures = self.noObjectiveFeatures + self.noDiversityFeatures
        # the the features array, first we retain all objective features, followed by diversity features
        # we separate them later using array slicing
        self.features = objectiveFeatures + diversityFeatures
        
        ranges = [(0, 1)] * self.noDiversityFeatures

        self.setArchive()
        self.setEmitters()
        self.scheduler = Scheduler(self.archive, self.emitters)

        # for computing feature weights
        self.prevObjectiveFeatScores = np.zeros((self.noSolutions, self.noObjectiveFeatures))

    def setConstraints(self, constraints):
        self.constraints = constraints
        self.noConstraints = len(self.constraints)

    def constraintCheckKernel(self, idxPair):
        solutionIdx, constrIdx = idxPair
        return self.constraints[constrIdx].check(self.renderOutputs[solutionIdx])

    def computeFeatures(self, solutionPop, noWorkers = 1):
        
        featScores = np.zeros((self.noSolutions, self.noFeatures))
        constraintChecks = np.full((self.noSolutions, self.noConstraints), True)
        
        # note: no significant speedup from using list comprehensions, for loops are fine
        for i in range(self.noSolutions):
            self.renderOutputs[i] = self.renderer.drawChrom(solutionPop[i])

        passesConstraints = [True] * self.noSolutions
        for i in range(self.noSolutions):
            for j in range(self.noConstraints):
                constraintChecks[i, j] = self.constraints[j].check(self.renderOutputs[i])
            passesConstraints[i] = all(constraintChecks[i])
        
        if noWorkers is None or noWorkers <= 1:
            for i in range(self.noSolutions):
                for j in range(self.noFeatures):
                    if j >= self.noObjectiveFeatures or passesConstraints[i]:
                        # either the feature is a diversity feature, 
                        # or an objective feature of a solution that passes all constraints
                        featScores[i, j] = self.features[j].compute(self.renderOutputs[i])
                    # else the solution does not pass all constraints 
                    # so its final objective feature score is 0, as initialized

        else: 
            mp = MultiWorker() 
            mp.renderOutputs = self.renderOutputs
            mp.passesConstraints = passesConstraints
            mp.features = self.features
        
            workerPool = WorkerPool(noWorkers)
            featScores = mp.computeFeatureScores(workerPool)
            workerPool.close()
            featScores = np.array(featScores)

        # compute weights of objective features
        featWeights = featScores[:, :self.noObjectiveFeatures] - self.prevObjectiveFeatScores
        featWeights = np.mean(featWeights, axis = 0)
        featWeights[featWeights < 0] = 0
        featWeights += 0.000001 # prevents divide-by-zeros when doing weighted averages
        # no need to normalize the weights

        self.prevObjectiveFeatScores = featScores[:, :self.noObjectiveFeatures]

        # for each solution, return mean objective feature score and diversity feature scores
        return np.average(featScores[:, :self.noObjectiveFeatures], axis = 1, weights = featWeights), \
                       featScores[:, self.noObjectiveFeatures:]
    
    def runIteration(self):

            # generate solution population of size batchSize * noEmitters
            solutionPop = self.scheduler.ask()

            objectiveValues, diversityValues = self.computeFeatures(solutionPop)
            
            self.scheduler.tell(objectiveValues, diversityValues)

    def getArchiveSolutions(self):
        feasibleSolutions = [elite['solution'] for elite in self.archive if elite['objective'] > 0] 
        infeasibleSolutions = [elite['solution'] for elite in self.archive if elite['objective'] <= 0] 
        return np.array(feasibleSolutions), np.array(infeasibleSolutions)
    
    def getRandomSolutionSample(self, noDesiredSolutions):
        # older method of archive sampling, unused but left for convenience
        elites = self.archive.sample_elites(noDesiredSolutions)
        return elites['solution']

class MultiWorker:
    def __init__(self):
        self.renderOutputs = []
        self.passesConstraints = []
        self.features = []
    
    def checkConstraints(self, procPool):
        noRenderOutputs = len(self.renderOutputs)
        noConstraints = len(self.constraints)
        constrIdxPairs = list(iterProduct(range(noRenderOutputs), range(noConstraints)))
        constrChecks = procPool.map(self.constraintCheckKernel, constrIdxPairs)
        return constrChecks

    def computeFeatureScoreKernel(self, idxPair):
        solutionIdx, featIdx = idxPair
        return self.features[featIdx].compute(self.renderOutputs[solutionIdx])

    def computeFeatureScores(self, procPool):
        noRenderOutputs = len(self.renderOutputs)
        noFeatures = len(self.features)
        featIdxPairs = list(iterProduct(range(noRenderOutputs), range(noFeatures)))
        featScores = workerPool.map(self.computeFeatureScoreKernel, featIdxPairs)
        return featScores

    def computeFeatureScoreKernel(self, renderOut):
        return [feat.compute(renderOut) for feat in self.features]

    def computeFeatureScores(self, workerPool):
        return workerPool.map(self.computeFeatureScoreKernel, self.renderOutputs)



            




