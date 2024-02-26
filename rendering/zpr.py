import glm
import math

# note: glm uses column ordering for matrices

def translateMat(v : glm.vec3):
    '''
    generate translation matrix from v 
    '''
    return glm.mat4(1, 0, 0, 0,
                    0, 1, 0, 0, 
                    0, 0, 1, 0,
                    v.x, v.y, v.z, 1)

def scaleMat(v : glm.vec3):
    '''
    generate scale matrix from v 
    '''
    return glm.mat4(v.x, 0, 0, 0,
                    0, v.y, 0, 0,
                    0, 0, v.z, 0,
                    0, 0, 0, 1)

def rotateMat(angle, axis : glm.vec3):
    '''
    generate rotation matrix around axis by angle
    formula taken from: https://ai.stackexchange.com/questions/14041/how-can-i-derive-the-rotation-matrix-from-the-axis-angle-rotation-vector
    '''
    c = math.cos(angle)
    s = math.sin(angle)
    ax = axis.x
    ay = axis.y
    az = axis.z

    return glm.mat4(c+ax**2*(1-c), ay*ax*(1-c)+az*s, az*ax*(1-c)-ay*s, 0,
                    ax*ay*(1-c)-az*s, c+ay**2*(1-c), az*ay*(1-c)+ax*s, 0,
                    ax*az*(1-c)+ay*s, ay*az*(1-c)-ax*s, c+az**2*(1-c), 0,
                    0, 0, 0, 1)

def zoom(dy, modelViewMat, sensitivity = 1.0):
    dv = glm.vec3(0, 0, dy)
    z = glm.mat3(glm.inverse(modelViewMat)) * dv * 0.001 * sensitivity
    return translateMat(z)

def pan(dx, dy, modelViewMat, sensitivity = 1.0):
    dv = glm.vec3(dx, -dy, 0)
    p = glm.mat3(glm.inverse(modelViewMat)) * dv * 0.001 * sensitivity
    return translateMat(p)

def rotate(dx, dy, modelViewMat, sensitivity = 1.0):
    dv = glm.vec3(dy, dx, 0)
    angle = glm.length(dv) * 0.001 * sensitivity 
    if angle > 2*glm.pi(): angle = 0
    if angle < 0: angle = 2*glm.pi()
    rotAxis = glm.normalize(glm.mat3(glm.inverse(modelViewMat)) * dv)
    return rotateMat(angle, rotAxis)