import os
import glm
import OpenGL.GL as gl
import OpenGL.GLU as glu

class GLSLShader:
    def __init__(self, name, vertFile, fragFile):
        self.name = name
        self.programHandle = 0
        self.vertHandle = 0
        self.fragHandle = 0
        self.vertFile = vertFile
        self.fragFile = fragFile
        
    def getVersion():
        version = gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        return version

    def readSourceFile(sourceFile):
        if not os.path.exists(sourceFile):
            print(f'Error: file not found: {sourceFile}')
            return ''
        inFile = open(sourceFile, 'r')
        return inFile.read()

    def shaderTypeToString(shaderType):
        if shaderType == gl.GL_VERTEX_SHADER: return 'vertex'
        if shaderType == gl.GL_FRAGMENT_SHADER: return 'fragment'
        return 'unknown'

    def compile(self, shaderType, shaderSource):
        shaderHandle = gl.glCreateShader(shaderType)
        shaderTypeName = GLSLShader.shaderTypeToString(shaderType)
        if shaderHandle == 0:
            print(f'Error creating {shaderTypeName} shader from {self.name}')
            return 0
        
        gl.glShaderSource(shaderHandle, shaderSource)
        gl.glCompileShader(shaderHandle)
        
        isCompiled = gl.glGetShaderiv(shaderHandle, gl.GL_COMPILE_STATUS)
        if isCompiled == gl.GL_FALSE:
            infolog = gl.glGetShaderInfoLog(shaderHandle)
            gl.glDeleteShader(shaderHandle)
            print(f'{self.name} {shaderTypeName} output:\n{infolog.decode()}\n')

        return shaderHandle

    def use(self):
        gl.glUseProgram(self.programHandle)

    def link(self):
        gl.glAttachShader(self.programHandle, self.vertHandle)
        gl.glAttachShader(self.programHandle, self.fragHandle)
        gl.glLinkProgram(self.programHandle)
        isLinked = gl.glGetProgramiv(self.programHandle, gl.GL_LINK_STATUS)
        if isLinked == gl.GL_FALSE:
            infolog = gl.glGetProgramInfoLog(self.programHandle)
            gl.glDeleteProgram(self.programHandle)
            gl.glDeleteShader(self.vertHandle)
            gl.glDeleteShader(self.fragHandle)
            self.vertHandle = 0
            self.fragHandle = 0
            print(f'{self.name} program output:\n{infolog.decode()}\n')
            return
        gl.glDetachShader(self.programHandle, self.vertHandle)
        gl.glDetachShader(self.programHandle, self.fragHandle)

    def cleanup(self):
        if self.vertHandle: 
            gl.glDeleteShader(self.vertHandle)
            self.vertHandle = 0
        if self.fragHandle: 
            gl.glDeleteShader(self.fragHandle)
            self.fragHandle = 0
        if self.programHandle:
            gl.glDeleteProgram(self.programHandle)
            self.programHandle = 0
    
    def build(self):
        self.cleanup()
        vertSource = GLSLShader.readSourceFile(self.vertFile)
        fragSource = GLSLShader.readSourceFile(self.fragFile)
        self.programHandle = gl.glCreateProgram()
        if self.programHandle == 0:
            print(f'Error: failed to create shader program for: {self.name}\n')
            return
        self.vertHandle = self.compile(gl.GL_VERTEX_SHADER, vertSource)
        self.fragHandle = self.compile(gl.GL_FRAGMENT_SHADER, fragSource)
        self.link()

    def uniformInt(self, uniformName : str, val : int):
        loc = gl.glGetUniformLocation(self.programHandle, uniformName)
        gl.glUniform1i(loc, val)

    def uniformFloat(self, uniformName : str, val : float):
        loc = gl.glGetUniformLocation(self.programHandle, uniformName)
        gl.glUniform1f(loc, val)

    def uniformFloatArray(self, uniformName : str, arr):
        loc = gl.glGetUniformLocation(self.programHandle, uniformName)
        gl.glUniform1fv(loc, len(arr), arr)

    def uniformVec2(self, uniformName : str, vec : glm.vec2):
        loc = gl.glGetUniformLocation(self.programHandle, uniformName)
        gl.glUniform2f(loc, vec.x, vec.y)

    def uniformIVec2(self, uniformName : str, vec : glm.ivec2):
        loc = gl.glGetUniformLocation(self.programHandle, uniformName)
        gl.glUniform2i(loc, vec.x, vec.y)

    def uniformVec3(self, uniformName : str, vec : glm.vec3):
        loc = gl.glGetUniformLocation(self.programHandle, uniformName)
        gl.glUniform3f(loc, vec.x, vec.y, vec.z)

    def uniformVec4(self, uniformName : str, vec : glm.vec4):
        loc = gl.glGetUniformLocation(self.programHandle, uniformName)
        gl.glUniform4f(loc, vec.x, vec.y, vec.z, vec.w)

    def uniformMat4(self, uniformName : str, matrix : glm.mat4):
        loc = gl.glGetUniformLocation(self.programHandle, uniformName)
        gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(matrix))

    

    
