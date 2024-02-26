#version 420

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inTexCoord;

out vec3 texCoord;

uniform mat4 mvp;

void main()
{
	texCoord = inTexCoord;
	gl_Position = mvp * vec4(inPos, 1.0);
}