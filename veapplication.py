from datasetloader import DatasetLoader
from rendering.renderer import VolumeRenderer
from evolutionary import MapElites
from imagegenerator import ImageGenerator
from featuremanager import FeatureManager
from parameters import VEParameters
from argparse import ArgumentParser
import os

class VEApplication:
    def __init__(self):

        self.dataFile = 'datasets/bonsai.vdf' # default dataset, if none is provided
        self.datasetLoader = DatasetLoader()
        self.dataset = self.datasetLoader.fromVDF(self.dataFile)
        self.noIterations = 10

        self.computeWidth = 128
        self.computeHeight = 128
                
        self.renderer = VolumeRenderer(self.dataset, self.computeWidth, self.computeHeight)
        self.evolAlg = MapElites(self.renderer)
        self.imGen = ImageGenerator(self.renderer)
        self.featManager = FeatureManager()

        self.evolAlg.setFeatures(self.featManager.objectiveFeatures,
                                 self.featManager.diversityFeatures)
        self.evolAlg.setConstraints(self.featManager.constraints)

        self.setupArgParser()

    def updateParameters(self, paramDict):
        self.dataFile = paramDict['dataset']['value']
        self.noIterations = paramDict['noIterations']['value']
        self.computeWidth = paramDict['computeWidth']['value']
        self.computeHeight = paramDict['computeHeight']['value']
        
        self.renderer.updateParameters(paramDict)
        self.evolAlg.updateParameters(paramDict)
        self.renderer.updateParameters(paramDict)
        self.imGen.updateParameters(paramDict)
        self.featManager.updateParameters(paramDict)
        
        self.dataset = self.datasetLoader.fromVDF(self.dataFile)
        self.renderer.setDataset(self.dataset)
        self.renderer.setViewportSize(self.computeWidth, self.computeHeight)

        self.evolAlg.setFeatures(self.featManager.objectiveFeatures,
                                 self.featManager.diversityFeatures)
        self.evolAlg.setConstraints(self.featManager.constraints)

    def run(self):
        
        if self.parseUserArguments():

            self.progressBar(0, self.noIterations, 'Working: ', 'Complete', decimals = 1, barLength = 50) 
            for iter in range(self.noIterations):
                #print('Iteration', iter+1) 
                self.evolAlg.runIteration()
                self.progressBar(iter+1, self.noIterations, 'Working: ', 'Complete', decimals = 1, barLength = 50) 

            feasibleSolutions, infeasibleSolutions = self.evolAlg.getArchiveSolutions()

            images = self.imGen.imagesFromSolutions(feasibleSolutions, infeasibleSolutions)

            if self.imGen.outputDir:
                self.imGen.saveImageSet(images, self.imGen.outputDir)

            if self.imGen.showGeneratedImages:
                self.imGen.showImageSet(images)

    def writeParameters(self, configFile):
        cfg = open(configFile, 'w')
        
        # get unique categories
        categories = [VEParameters[param]['category'] for param in VEParameters.keys()]
        categories = sorted(set(categories))
        for categ in categories:
            cfg.write(f'[{categ}]\n')
            # get all parameters from category
            params = {paramName : VEParameters[paramName] for paramName in VEParameters.keys() 
                      if VEParameters[paramName]['category'] == categ}
            params = dict(sorted(params.items()))
            for p in params:
                cfg.write(f'{p}: {VEParameters[p]["value"]}   #{VEParameters[p]["description"]}\n')
            cfg.write('\n')

        cfg.close()

    def validateParameter(self, paramName, paramValStr):
        # attempt to get parameter value from string representation paramValStr

        if paramName not in VEParameters:
            return None, f'{paramName} is not a valid parameter name'

        paramType = VEParameters[paramName]['dtype']
        paramVal = None

        if paramType == int:
            try:
                paramVal = int(paramValStr)
            except:
                return None, f'{paramName} value {paramValStr} is not a valid integer'
            paramDomain = VEParameters[paramName]['domain']
            if paramVal < paramDomain[0] or paramVal > paramDomain[1]:
                return None, f'{paramName} value {paramValStr} is outside range {paramDomain}'
            return paramVal, ''

        if paramType == float:
            try:
                paramVal = float(paramValStr)
            except:
                return None, f'{paramName} value {paramValStr} is not a valid real number'
            paramDomain = VEParameters[paramName]['domain']
            if paramVal < paramDomain[0] or paramVal > paramDomain[1]:
                return None, f'{paramName} value {paramValStr} is outside range {paramDomain}'
            return paramVal, ''

        if paramType == bool:
            if paramValStr in ['True', 'true']:
                paramVal = True
            elif paramValStr in ['False', 'false']:
                paramVal = False
            else:
                return None, f'{paramName} should be True or False'
            return paramVal, ''

        if paramName in ['objectiveFeatures', 'diversityFeatures']:
            paramVal = paramValStr.strip('][')
            paramVal = paramVal.split(',')
            paramVal = [feat.strip(" '") for feat in paramVal]
            for feat in paramVal:
                if feat not in self.featManager.featureFactory.keys():
                    return None, f'{paramName}: {feat} is not a valid feature. Supported features: {list(self.featManager.featureFactory.keys())}'
            return paramVal, ''

        if paramName == 'dataset':
            if not paramValStr.isspace(): 
                try:
                    open(paramValStr)
                except:
                    return None, f'file not found: {paramValStr}'
                else:
                    return paramValStr, ''

        if paramName == 'outputDir':
            if paramValStr and not paramValStr.isspace():
              if not os.path.isdir(paramValStr):
                return None, f'invalid path: {paramValStr}'
            return paramValStr, ''

        if paramType == str:
            return paramValStr, ''
        
        return None, 'invalid parameter'

    def parametersFromConfigFile(self, configFile):
        cfg = open(configFile, 'r')

        errMsg = ''

        cfgParams = {}

        lineIdx = 0
        for line in cfg:
            lineIdx += 1
            if line.isspace(): # current line contains only whitespaces
               continue
            line = line.strip()
            if line[0] == '[': # current line contains a category
                continue
            if line[0] == '#': # current line is a comment
                continue
            if ':' not in line: #line does not contain a valid parameter definition
                errMsg += f'line {lineIdx} does not contain a valid parameter definition, expected "[parameter]:[value]".\n'
                continue
            if '#' in line: 
                line = line.split('#')[0] # remove any comments from line, if any

            toks = line.split(':')
            paramName = toks[0].strip()
            paramValStr = toks[1].strip()

            paramVal, paramErr = self.validateParameter(paramName, paramValStr)

            if paramVal is None:
                errMsg += f'parameter error in line {lineIdx}: {paramErr}.\n'
            else:
                cfgParams[paramName] = paramVal
        
        cfg.close()

        if not errMsg:
            for pName in cfgParams.keys():
                VEParameters[pName]['value'] = cfgParams[pName]

            self.updateParameters(VEParameters)

        return errMsg

    def setupArgParser(self):
        self.argParser = ArgumentParser()
        
        for paramName in VEParameters:
            paramType = VEParameters[paramName]['dtype']
            description = VEParameters[paramName]['description']

            if paramType == bool:
                self.argParser.add_argument('--' + paramName, type = str, choices = ['True', 'False'], help = description)

            elif paramType == list:
                self.argParser.add_argument('--' + paramName, type = str, nargs = '+', help = description)

            else: # paramType is int or float
                self.argParser.add_argument('--' + paramName, type = paramType, help = description)

        # argument for loading config file
        self.argParser.add_argument('--config', type = str, help = 'configuration file for setting parameter values') 

    def parseUserArguments(self):
        
        # all parameters known to the argument parser
        allArgs = vars(self.argParser.parse_args())
        # parameters specified by the user
        userArgs = {argName : allArgs[argName] for argName in allArgs.keys() if allArgs[argName] is not None}
        
        # check for config file
        if 'config' in userArgs.keys():
            cfgFile = userArgs['config']
            if not os.path.isfile(cfgFile):
                print(f'Error: cannot find config file {cfgFile}.')
                return False
            else:
                # config file exists, try to load parameters from it:
                print(f'\nLoading {cfgFile}')
                errMsg = self.parametersFromConfigFile(cfgFile)
                if errMsg:
                    print(f'\nErrors found in config file {cfgFile}:\n{errMsg}')
                    return False
            
        # check for user-specified command-line parameters
        cliParams = {}
        for argName in userArgs.keys():
            if argName != 'config':
                argValStr = userArgs[argName]
                argVal, errMsg = self.validateParameter(argName, argValStr)
                if argVal is None:
                    print(f'Command-line error: {errMsg}')
                    return False
                cliParams[argName] = argVal

        # all CLI parameters should be ok
        for pName in cliParams.keys():
            VEParameters[pName]['value'] = cliParams[pName]

        self.updateParameters(VEParameters)

        return True

    def progressBar(self,
                    currentIter, noIter, 
                    prefix = '', 
                    suffix = '', 
                    decimals = 1, 
                    barLength = 100, 
                    barFill = '|', 
                    printEnd = '\r'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (currentIter / float(noIter)))
        filledLength = int(barLength * currentIter // noIter)
        bar = barFill * filledLength + '-' * (barLength - filledLength)
        print(f'\r{prefix} [{bar}] {percent}% {suffix}', end = printEnd)
        if currentIter == noIter: print() # new line when complete
            

    
        


        

            
                







