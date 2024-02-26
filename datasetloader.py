from dataset import VolumeDataset
import glm
import os

class DatasetLoader:
    def __init__(self):
        # parameters data_file, size, bytes_per_voxel, big_endian are mandatory
        # name of data set
        # default parameters in case the rest are missing from VDF file
        self.defaultParams = {
                'scaleFactor' : glm.vec3(1.0),
                'viewerPosition' : glm.vec3(0, 1.0, 0),
                'lightPosition' : glm.vec3(0, -1.0, 0), 
                'origin' : glm.vec3(0),
                'mainAxis' : glm.vec3(0, 0, 1),
                'enableNormalization' : True,
                'headerSkip' : 0
                }

        self.predefinedDatasets = { 
            'schaedel' : 'datasets/schaedel.vdf',
            'hurricane' : 'datasets/hurricane.vdf',
            'bonsai' : 'datasets/bonsai.vdf',
            'buckyball' : 'datasets/buckyball.vdf',
            'manix' : 'datasets/manix.vdf',
            'engine' : 'datasets/engine.vdf'
            }
    
    def fromVDF(self, vdfFile):
        paramDict = self.defaultParams.copy()
        (vdfPath, vdfName) = os.path.split(vdfFile)
        vdf = open(vdfFile, 'r')
        
        for line in vdf:
            toks = line.split(':')
            paramName = toks[0].strip()
            paramStr = toks[1].strip()
            
            if paramName == 'dataFile':
                paramDict[paramName] = vdfPath + '/' + paramStr

            elif paramName == 'size':
                vals = paramStr.split(',')
                vals = [int(v) for v in vals]
                paramDict[paramName] = glm.ivec3(vals)

            elif paramName == 'bytesPerVoxel':
                paramDict[paramName] = int(paramStr)

            elif paramName == 'bigEndian':
                paramDict[paramName] = True if paramStr == 'True' else False
                
            elif paramName == 'scaleFactor':
                vals = paramStr.split(',')
                vals = [float(v) for v in vals]
                paramDict[paramName] = glm.vec3(vals)

            elif paramName == 'enableNormalization':
                paramDict[paramName] = True if paramStr == 'True' else False

            elif paramName == 'headerSkip':
                paramDict[paramName] = int(paramStr)

            elif paramName == 'viewerPosition':
                vals = paramStr.split(',')
                vals = [float(v) for v in vals]
                paramDict[paramName] = glm.vec3(vals)

            elif paramName == 'lightPosition':
                vals = paramStr.split(',')
                vals = [float(v) for v in vals]
                paramDict[paramName] = glm.vec3(vals)

            elif paramName == 'origin':
                vals = paramStr.split(',')
                vals = [float(v) for v in vals]
                paramDict[paramName] = glm.vec3(vals)

            elif paramName == 'mainAxis':
                vals = paramStr.split(',')
                vals = [float(v) for v in vals]
                paramDict[paramName] = glm.vec3(vals)

            elif paramName == 'name':
                paramDict[paramName] = paramStr

        # if an explicit name is not provided, we use the name of the vdf file 
        if 'name' not in paramDict.keys():
            paramDict['name'] = vdfName
                
        vdf.close()
        
        return VolumeDataset(
            dataFile = paramDict['dataFile'],
            size = paramDict['size'], 
            scaleFactor = paramDict['scaleFactor'],
            bytesPerVoxel = paramDict['bytesPerVoxel'],
            bigEndian = paramDict['bigEndian'],
            enableNormalization = paramDict['enableNormalization'], 
            headerSkip = paramDict['headerSkip'],
            viewerPosition = paramDict['viewerPosition'],
            lightPosition = paramDict['lightPosition'],
            origin = paramDict['origin'],
            mainAxis = paramDict['mainAxis'],
            name = paramDict['name']
        )


