import numpy as np
import ctypes
import OpenGL.GL as gl
import glfw
import glm
import rendering.zpr as zpr
from dataset import VolumeDataset
from rendering.shader import GLSLShader
from rendering.transfunc import TransFunc
from rendering.renderoutput import RenderOutput
import time

class VolumeRenderer:

    def getGLVersionStr():
        rendererStr = gl.glGetString(gl.GL_RENDERER).decode()
        versionStr = gl.glGetString(gl.GL_VERSION).decode()
        return f'OpenGL renderer: {rendererStr}, version: {versionStr}'

    def getMaxNoFBOColorAttachments():
        return gl.glGetIntegerv(gl.GL_MAX_DRAW_BUFFERS)

    def getMaxNoShaderAccessibleTextures():
        return gl.glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS)

    def getMaxNoConcurrentTextureUnits():
        return gl.glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS)

    def __init__(self, dataset : VolumeDataset, w : int, h : int):

        self.dataset = dataset
        self.transFunc = TransFunc()
        self.dataTex = 0
        self.transFuncDataSize = 1024
        self.transFuncData = self.transFunc.getData(self.transFuncDataSize)
        self.transFuncTex = 0
        self.cubeVBO = 0
        self.cubeVAO = 0
        
        self.volMin = glm.vec3(0)
        self.volMax = glm.vec3(1)
        self.modelMat = glm.mat4(1)
        self.viewMat = glm.mat4(1)
        self.projMat = glm.mat4(1)

        # noise for stochastic jittering
        self.noiseSize = glm.ivec2(64, 64)
        self.noiseTex = 0 
        self.rayJitterStrength = 0.1 # amount of ray jittering
        
        self.cubeSize = self.dataset.scaleFactor
        self.viewerPos = self.dataset.viewerPosition
        self.lookAtPos = self.dataset.origin
        self.viewerUpDir = self.dataset.mainAxis
        self.lightPos = self.dataset.lightPosition

        self.useFixedLightPos = False
        self.useFixedViewerPos = False

        self.enableLighting = True
        self.enableShadows = False
        self.enableSurfaceHighlights = False
        self.surfaceHighlightStrength = 0.2
        self.shadowOpacity = 1.0

        self.diffuseLightIntensity = 0.5
        self.specularLightIntensity = 0.8
        self.specularPower = 0.5
        self.lightAmplification = 0.5
        
        self.backColor = glm.vec4(1.0)

        self.interactSensitivity = 5.0

        self.texelType = {'B' : gl.GL_UNSIGNED_BYTE, 
                          'u2' : gl.GL_UNSIGNED_SHORT,
                          '<u2' : gl.GL_UNSIGNED_SHORT,
                          '>u2' : gl.GL_UNSIGNED_SHORT,
                          'f' : gl.GL_FLOAT,
                          '<f' : gl.GL_FLOAT,
                          '>f' : gl.GL_FLOAT}
       
        self.shader = GLSLShader('raycaster', 
                                    'rendering/raycaster.vert', 
                                    'rendering/raycaster_iso.frag')

        '''
        Note: for efficiency, we use a single fbo with multiple color attachemnts
        color attachment 0 is used for rendering the image and has an unint8 texture attached
        the other color attachments are used for isosurface features and have float textures attached
        the max number of concurrent isosurfaces is the max number of color attachments - 1

        GPUs with OpenGL 3.X support allow (at least?) 8 fbo color attachments
        '''
        
        self.maxNoIsosurfaces = 5
        self.minNoIsosurfaces = 1

        self.maxDensity = 0.7
        self.minDensity = 0.05

        self.rayStepSize = 0.001
        self.shadowRayStepSize = 0.005

        self.mainAxisMaxAngle = 2*np.pi
        self.secondaryAxisMaxAngle = np.pi/3

        self.ambientLightIntensity = 0.1
        self.diffuseLightIntensity = 1.0
        self.specularLightIntensity = 0.8
        self.lightAmplification = 1.5;

        self.lightsourceMaxAngle = np.pi/2 # max deviation of light source direction from viewer direction

        self.fbo = 0
        self.renderTex = 0 # texture to render main images

        self.renderOutput = RenderOutput()
        
        self.depthTex = 0 # texture for depth buffer
        self.w = w
        self.h = h

        # context for offscreen rendering using glfw
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        offscreenContext = glfw.create_window(self.w, self.h, '', None, None)
        glfw.make_context_current(offscreenContext)

        self.initGL()

        # end constructor

    def degreesToRadians(self, angle):
        return angle * np.pi / 180 

    def updateParameters(self, paramDict):
        self.enableLighting = paramDict['enableLighting']['value']
        self.enableSurfaceHighlights = paramDict['enableSurfaceHighlights']['value']
        self.enableShadows = paramDict['enableShadows']['value']
        self.mainAxisMaxAngle = self.degreesToRadians(paramDict['mainAxisMaxAngle']['value'])
        self.secondaryAxisMaxAngle = self.degreesToRadians(paramDict['secondaryAxisMaxAngle']['value'])
        self.rayStepSize = paramDict['rayStepSize']['value']
        self.shadowRayStepSize = paramDict['shadowRayStepSize']['value']
        self.ambientLightIntensity = paramDict['ambientLightIntensity']['value']
        self.diffuseLightIntensity = paramDict['diffuseLightIntensity']['value']
        self.specularLightIntensity = paramDict['specularLightIntensity']['value']
        self.specularPower = paramDict['specularPower']['value']
        self.lightAmplification = paramDict['lightAmplification']['value']
        self.surfaceHighlightStrength = paramDict['surfaceHighlightStrength']['value']
        self.shadowOpacity = paramDict['shadowOpacity']['value']

        self.updateRenderer()

    def updateRenderer(self):
        # update settings / structures / buffers that depend on parameters
        self.maxNoRenderableIsosurfaces = VolumeRenderer.getMaxNoFBOColorAttachments() - 1
        self.maxNoIsosurfaces = min(self.maxNoIsosurfaces, self.maxNoRenderableIsosurfaces)
        self.featureTex = [0] * self.maxNoIsosurfaces # textures to render isosurface features
        self.setupFBO()
        self.setupShader()

    def setDataset(self, newDataset : VolumeDataset):
        # change dataset without rebuilding entire renderer
        self.dataset = newDataset
        self.cubeSize = self.dataset.scaleFactor
        self.setupProxyCube(self.cubeSize.x, self.cubeSize.y, self.cubeSize.z)
        self.viewerPos = self.dataset.viewerPosition
        self.lookAtPos = self.dataset.origin
        self.viewerUpDir = self.dataset.mainAxis
        self.lightPos = self.dataset.lightPosition
        
        self.viewMat = glm.lookAt(self.viewerPos, self.lookAtPos, self.viewerUpDir)
        
        self.setupDatasetTexture()
        self.setupShader()

    def rebuildShader(self):
        self.shader.build()

    def setViewportSize(self, w, h):
        self.w = w
        self.h = h
        self.setupFBO()
        self.setupShader()

    def setupDatasetTexture(self):
        gl.glDeleteTextures(1, [self.dataTex])
        self.dataTex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_3D, self.dataTex)

        gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, gl.GL_RED, 
                        self.dataset.sizeX, self.dataset.sizeY, self.dataset.sizeZ, 
                        0, gl.GL_RED, self.texelType[self.dataset.dataType], 
                        self.dataset.voxelData)

        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    def setupNoiseTexture(self):
        noise2D = np.random.randint(0, 256, 
                                    size = [self.noiseSize[0], self.noiseSize[1]], 
                                    dtype = np.uint8)

        gl.glDeleteTextures(1, [self.noiseTex])
        self.noiseTex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.noiseTex)
        
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RED, 
                        self.noiseSize[0], self.noiseSize[1], 
                        0, gl.GL_RED, gl.GL_UNSIGNED_BYTE, 
                        noise2D)
        
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

    def setupProxyCube(self, sizeX, sizeY, sizeZ):
        LX = sizeX / 2
        LY = sizeY / 2
        LZ = sizeZ / 2
        tmin = 0
        tmax = 1.0

        cubeData = np.array([
            #front
            -LX, -LY, LZ, tmin, tmin, tmax, LX, -LY, LZ, tmax, tmin, tmax, LX, LY, LZ, tmax, tmax, tmax,
            -LX, -LY, LZ, tmin, tmin, tmax, LX, LY, LZ, tmax, tmax, tmax, -LX, LY, LZ, tmin, tmax, tmax,  
            #back
            LX, -LY, -LZ, tmax, tmin, tmin, -LX, -LY, -LZ, tmin, tmin, tmin, -LX, LY, -LZ, tmin, tmax, tmin,
            LX, -LY, -LZ, tmax, tmin, tmin, -LX, LY, -LZ, tmin, tmax, tmin, LX, LY, -LZ, tmax, tmax, tmin,
            #left
            -LX, -LY, LZ, tmin, tmin, tmax, -LX, LY, LZ, tmin, tmax, tmax, -LX, LY, -LZ, tmin, tmax, tmin,
            -LX, -LY, LZ, tmin, tmin, tmax, -LX, LY, -LZ, tmin, tmax, tmin, -LX, -LY, -LZ, tmin, tmin, tmin,
            #right
            LX, -LY, LZ, tmax, tmin, tmax, LX, -LY, -LZ, tmax, tmin, tmin, LX, LY, -LZ, tmax, tmax, tmin,
            LX, -LY, LZ, tmax, tmin, tmax, LX, LY, -LZ, tmax, tmax, tmin, LX, LY, LZ, tmax, tmax, tmax,
            #top
            LX, LY, LZ, tmax, tmax, tmax, -LX, LY, -LZ, tmin, tmax, tmin, -LX, LY, LZ, tmin, tmax, tmax, 
            LX, LY, LZ, tmax, tmax, tmax, LX, LY, -LZ, tmax, tmax, tmin, -LX, LY, -LZ, tmin, tmax, tmin,
            #bottom
            -LX, -LY, LZ, tmin, tmin, tmax, -LX, -LY, -LZ, tmin, tmin, tmin, LX, -LY, -LZ, tmax, tmin, tmin,
            -LX, -LY, LZ, tmin, tmin, tmax, LX, -LY, -LZ, tmax, tmin, tmin, LX, -LY, LZ, tmax, tmin, tmax
            ], dtype='float32')

        self.cubeVBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.cubeVBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, cubeData, gl.GL_STATIC_DRAW)
        self.cubeVAO = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.cubeVAO)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(0)
        gl.glEnableVertexAttribArray(1)
        
    def worldToVolume(self, worldCoords):
        volToCube = zpr.scaleMat(self.cubeSize) * zpr.translateMat(glm.vec3(-0.5))
        volCoords = glm.inverse(self.modelMat * volToCube) * glm.vec4(worldCoords, 1.0)
        return volCoords.xyz

    def setupTransFuncTexture(self):
        gl.glDeleteTextures(1, self.transFuncTex)
        self.transFuncTex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_1D, self.transFuncTex)

        gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, gl.GL_RGBA, self.transFuncDataSize, 
                        0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, self.transFuncData)

        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    def setupShader(self):
        self.shader.use()

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_3D, self.dataTex)
        self.shader.uniformInt('dataTex', 0)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.noiseTex)
        self.shader.uniformInt('noiseTex', 1)
        
        self.shader.uniformFloatArray('isoValues', self.transFunc.controlPointsToArray())
        self.shader.uniformInt('noIsosurfaces', self.transFunc.noControlPoints())

        self.shader.uniformMat4('mvp', self.projMat * self.viewMat * self.modelMat)
        self.shader.uniformVec3('viewerPos', self.worldToVolume(self.viewerPos))
        self.shader.uniformVec3('lightPos', self.worldToVolume(self.lightPos))
        self.shader.uniformVec3('volMin', self.volMin)
        self.shader.uniformVec3('volMax', self.volMax)
        self.shader.uniformVec4('backColor', self.backColor)
        self.shader.uniformVec2('viewportSize', glm.ivec2(self.w, self.h))
        self.shader.uniformIVec2('noiseSize', self.noiseSize)
        self.shader.uniformFloat('rayJitterStrength', self.rayJitterStrength)
        self.shader.uniformFloat('rayStepSize', self.rayStepSize)
        self.shader.uniformFloat('shadowRayStepSize', self.shadowRayStepSize)
        self.shader.uniformFloat('shadowOpacity', self.shadowOpacity)
        
        self.shader.uniformFloat('ambientLightIntensity', self.ambientLightIntensity)
        self.shader.uniformFloat('diffuseLightIntensity', self.diffuseLightIntensity)
        self.shader.uniformFloat('specularLightIntensity', self.specularLightIntensity)
        self.shader.uniformFloat('specularPower', self.specularPower)
        self.shader.uniformFloat('lightAmplification', self.lightAmplification)

        self.shader.uniformInt('enableLighting', self.enableLighting)
        self.shader.uniformInt('enableShadows', self.enableShadows)
        self.shader.uniformInt('enableSurfaceHighlights', self.enableSurfaceHighlights)
        self.shader.uniformFloat('surfaceHighlightStrength', self.surfaceHighlightStrength)

    def setupFBO(self): 
        # set fbo and render/depth/stencil textures
        # glTexImage calls are in the bindFBOTextures() method
        gl.glDeleteTextures(1, [self.renderTex])
        self.renderTex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.renderTex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        
        gl.glDeleteTextures(1, [self.depthTex])
        self.depthTex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depthTex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        for i in range(self.maxNoIsosurfaces):
            self.featureTex[i] = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.featureTex[i])
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        gl.glDeleteFramebuffers(1, [self.fbo])
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, 
                                  gl.GL_TEXTURE_2D, self.renderTex, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, 
                                  gl.GL_TEXTURE_2D, self.depthTex, 0)

        drawBuffers = [gl.GL_COLOR_ATTACHMENT0]
        for i in range(self.maxNoIsosurfaces):
            colorBuff = gl.GL_COLOR_ATTACHMENT1 + i
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, colorBuff, gl.GL_TEXTURE_2D, self.featureTex[i], 0)
            drawBuffers.append(colorBuff)

        gl.glDrawBuffers(len(drawBuffers), drawBuffers)

        self.resizeGL(self.w, self.h)

    def bindFBOTextures(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.renderTex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.w, self.h, 
                        0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depthTex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT16, self.w, self.h, 
                        0, gl.GL_DEPTH_COMPONENT, gl.GL_UNSIGNED_SHORT, None)

        for i in range(self.maxNoIsosurfaces):
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.featureTex[i])
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, self.w, self.h, 
                            0, gl.GL_RGBA, gl.GL_FLOAT, None)

    def initGL(self):
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.setupTransFuncTexture()
        self.setupNoiseTexture()
        
        self.shader.build()
        self.setDataset(self.dataset)
        self.updateRenderer()
        
        gl.glClearColor(0, 0, 0, 0)
        
        #print(VolumeRenderer.getGLVersionStr())
        
    def drawGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # we need to separately clear the color of the render color attachment so as to 
        # avoid transparent pixels in the rendered images
        gl.glClearBufferuiv(gl.GL_COLOR, 0, glm.value_ptr(self.backColor))
        
        self.setupShader()
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)

    def resizeGL(self, w, h):
        
        self.bindFBOTextures()

        gl.glViewport(0, 0, w, h)
        self.projMat = glm.perspective(45, w/h, 0.1, 100)
        self.viewMat = glm.lookAt(self.viewerPos, self.lookAtPos, self.viewerUpDir)

    def interactZoom(self, dy):
        self.modelMat *= zpr.zoom(dy, self.viewMat * self.modelMat, self.interactSensitivity)

    def interactPan(self, dx, dy):
        self.modelMat *= zpr.pan(dx, dy, self.viewMat * self.modelMat, self.interactSensitivity)

    def interactRotate(self, dx, dy):
        self.modelMat *= zpr.rotate(dx, dy, self.viewMat * self.modelMat, self.interactSensitivity)

    def updateTransFunc(self):
        self.transFuncData = self.transFunc.getData(self.transFuncDataSize)
        gl.glBindTexture(gl.GL_TEXTURE_1D, self.transFuncTex)
        gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, gl.GL_RGBA, self.transFuncDataSize, 
                        0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, self.transFuncData)

    def getImage(self):
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        imgBuff = gl.glReadPixels(0, 0, self.w, self.h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE) 
        return np.frombuffer(imgBuff, dtype = 'uint8').reshape(self.h, self.w, 4)

    def getIsosurfaceFeatureMaps(self):
        featureMapList = [] # one featuremap per isosurface containing all 4 features as RGBA values
        for i in range(self.transFunc.noControlPoints()):
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1 + i)
            imgBuff = gl.glReadPixels(0, 0, self.w, self.h, gl.GL_RGBA, gl.GL_FLOAT)
            feat = np.frombuffer(imgBuff, dtype = 'float32').reshape(self.h, self.w, 4)
            featureMapList.append(feat)
        return np.array(featureMapList)

    def normalizeMap(self, featureMap):
        fmax = np.max(featureMap)
        fmin = np.min(featureMap)
        if fmin == fmax: return featureMap
        return (featureMap - fmin) / (fmax - fmin)

    def draw(self):
        self.drawGL()
        self.renderOutput = RenderOutput()
        self.renderOutput.image = self.getImage()
        self.renderOutput.noIsosurfaces = self.transFunc.noControlPoints()
        self.renderOutput.isoDensities = [p.x for p in self.transFunc.cp]
        self.renderOutput.isoColors = [np.array(p.rgbaVal.rgb) for p in self.transFunc.cp]
        self.renderOutput.isoOpacities = [p.rgbaVal.a for p in self.transFunc.cp]
        featMaps = self.getIsosurfaceFeatureMaps()
        self.renderOutput.gradMagnitudeMaps = featMaps[:,:,:,0]
        self.renderOutput.gradOrientationMaps = featMaps[:,:,:,1]
        self.renderOutput.curvatureMaps = featMaps[:,:,:,2]
        for i in range(self.renderOutput.noIsosurfaces):
            self.renderOutput.curvatureMaps[i] = self.normalizeMap(self.renderOutput.curvatureMaps[i])
        np.nan_to_num(self.renderOutput.curvatureMaps, copy = False) # TODO: should check why there are nan values in curvature maps
        self.renderOutput.visibilityMaps = featMaps[:,:,:,3]
        self.renderOutput.stencils = np.where(self.renderOutput.visibilityMaps > 0, 1, 0)
        self.renderOutput.visibilities = np.zeros(self.renderOutput.noIsosurfaces) 
        for i in range(self.renderOutput.noIsosurfaces):
            maskedVisMap = self.renderOutput.visibilityMaps[i][self.renderOutput.stencils[i] > 0]
            if maskedVisMap.size > 0:
                self.renderOutput.visibilities[i] = np.mean(maskedVisMap)

        return self.renderOutput

    ### update renderer using values from chromosome
    ## chrom structure: 
    # no_tf_control_points
    # values of control points (density, color, alpha)
    # x, y, z rotation angles if useFixedViewerPos == false
    # lx, ly, lz light rotation angles if useFixedLightPos == false
    ### all values in chrom are scaled to [0, 1]
    def drawChrom(self, chrom):
        idx = 0 # index used to iterate through chromosome
        # minimum number of isosurfaces should be 1
        self.noControlPoints = self.minNoIsosurfaces + round(chrom[idx] * (self.maxNoIsosurfaces - self.minNoIsosurfaces))
        idx += 1
        
        noControlPointValues = self.noControlPoints * self.transFunc.noValuesPerControlPoint
        maxNoControlPointValues = self.maxNoIsosurfaces * self.transFunc.noValuesPerControlPoint
        allControlPointValues = np.array(chrom[idx : idx + noControlPointValues])
        self.transFunc.controlPointsFromArray(allControlPointValues)
        for i in range(self.noControlPoints):
            self.transFunc.cp[i].x = self.minDensity + self.transFunc.cp[i].x * (self.maxDensity - self.minDensity)
        idx += maxNoControlPointValues

        if self.useFixedViewerPos == False:
            maxRotAngles = self.viewerUpDir * self.mainAxisMaxAngle/2  + \
                            (1 - self.viewerUpDir) * self.secondaryAxisMaxAngle/2 
            minRotAngles = -maxRotAngles
            
            rotAngles = minRotAngles + glm.vec3(chrom[idx], chrom[idx+1], chrom[idx+2]) * (maxRotAngles - minRotAngles)

            self.modelMat = glm.mat4(1)
            self.modelMat *= zpr.rotateMat(rotAngles.x, glm.vec3(1, 0, 0))
            self.modelMat *= zpr.rotateMat(rotAngles.y, glm.vec3(0, 1, 0))
            self.modelMat *= zpr.rotateMat(rotAngles.z, glm.vec3(0, 0, 1))
            idx += 3

        if self.useFixedLightPos == False:
            lightAngles = -self.lightsourceMaxAngle/2 + \
                          glm.vec3(chrom[idx], chrom[idx+1], chrom[idx+2]) * \
                          self.lightsourceMaxAngle
            self.lightPos = zpr.rotateMat(lightAngles.x, glm.vec3(1, 0, 0)) * \
                            zpr.rotateMat(lightAngles.y, glm.vec3(0, 1, 0)) * \
                            zpr.rotateMat(lightAngles.z, glm.vec3(0, 0, 1)) * \
                            self.viewerPos
            idx += 3

        return self.draw()

    def getMaxChromSize(self):
        chromSize = 1 + self.maxNoIsosurfaces * self.transFunc.noValuesPerControlPoint
        if self.useFixedViewerPos == False: chromSize += 3
        if self.useFixedLightPos == False: chromSize += 3
        return chromSize

        


    


