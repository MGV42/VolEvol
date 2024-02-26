import os
import numpy as np
import glm

class VolumeDataset:
    def __init__(self, dataFile : str, 
                 size : glm.ivec3,
                 scaleFactor : glm.vec3,
                 bytesPerVoxel : int, 
                 bigEndian : bool = True, 
                 enableNormalization : bool = False, 
                 headerSkip : int = 0,
                 viewerPosition : glm.vec3 = glm.vec3(1, 0, 0),
                 lightPosition : glm.vec3 = glm.vec3(1, 0, 0),
                 origin : glm.vec3 = glm.vec3(0),
                 mainAxis : glm.vec3 = glm.vec3(0, 0, 1),
                 name = 'volume data'
                 ):
        
        # purpose of headerSkip: if a volume data file has a header, 
        # we can skip the first headerSkip bytes when reading voxel data

        self.sizeX = size.x
        self.sizeY = size.y
        self.sizeZ = size.z
        self.scaleFactor = scaleFactor
        self.viewerPosition = viewerPosition
        self.lightPosition = lightPosition
        self.origin = origin
        self.mainAxis = mainAxis
        self.name = name

        noVoxels = self.sizeX * self.sizeY * self.sizeZ
        self.dataType = None
        
        if not os.path.exists(dataFile):
            print(f'Error: file not found: {dataFile}')
            return

        fileSize = os.path.getsize(dataFile)

        dataSize = noVoxels * bytesPerVoxel + headerSkip;

        if fileSize != dataSize:
            print(f'Error reading {dataFile} : file size {fileSize} does not match specified data size {dataSize}')

        endianChar = '>' if bigEndian else '<'

        if bytesPerVoxel == 1: 
            self.dataType = 'B' # unsigned byte
            endianChar = ''
        if bytesPerVoxel == 2: self.dataType = f'{endianChar}u2' # unsigned short
        if bytesPerVoxel == 4: self.dataType = f'{endianChar}f' # probably 32 bit float

        self.voxelData = np.fromfile(dataFile, dtype = self.dataType, offset = headerSkip)
        
        if enableNormalization:
            maxVoxel = np.max(self.voxelData)
            self.voxelData = self.voxelData.astype(np.float32)/maxVoxel
            self.dataType = f'{endianChar}f' # 32 bit float
            

        


