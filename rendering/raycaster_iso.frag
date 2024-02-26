#version 420

#define MAX_NO_ISOSURFACES 7
#define NO_ISOSURFACE_VALUES 5
#define GRAD_DELTA 0.015

in vec3 texCoord;

layout(location = 0) out vec4 fragColor;
out vec4 featValue[MAX_NO_ISOSURFACES];

uniform float[MAX_NO_ISOSURFACES * NO_ISOSURFACE_VALUES] isoValues;
uniform sampler3D dataTex;
uniform sampler2D noiseTex;
uniform ivec2 noiseSize;
uniform vec3 viewerPos;
uniform vec3 lightPos;
uniform vec3 volMin;
uniform vec3 volMax;
uniform vec4 backColor;
uniform ivec2 viewportSize;
uniform int noIsosurfaces;
uniform bool enableLighting;
uniform bool enableShadows;
uniform bool enableSurfaceHighlights;
uniform float surfaceHighlightStrength;
uniform float rayJitterStrength;
uniform float rayStepSize;
uniform float shadowRayStepSize;
uniform float shadowOpacity;
uniform float ambientLightIntensity;
uniform float diffuseLightIntensity;
uniform float specularLightIntensity;
uniform float specularPower;
uniform float lightAmplification;


void composit(inout vec4 dst, in vec4 src)
{
	dst.rgb += (1.0 - dst.a) * src.a * src.rgb;
	dst.a += (1.0 - dst.a) * src.a;
}

float jitter(float jitterScale)
{
	vec2 noisePos = gl_FragCoord.xy / noiseSize;
	return texture(noiseTex, noisePos).r * jitterScale;
}

float sampleDataAt(vec3 pos) //todo: use this instead of texture() calls
{
	return texture(dataTex, pos).r;
}

vec3 gradient(vec3 position)
{
	vec3 sample1, sample2;
	float delta = GRAD_DELTA;
	sample1.x = texture(dataTex, position - vec3(delta, 0.0, 0.0)).r;
    sample2.x = texture(dataTex, position + vec3(delta, 0.0, 0.0)).r;
    sample1.y = texture(dataTex, position - vec3(0.0, delta, 0.0)).r;
    sample2.y = texture(dataTex, position + vec3(0.0, delta, 0.0)).r;
    sample1.z = texture(dataTex, position - vec3(0.0, 0.0, delta)).r;
    sample2.z = texture(dataTex, position + vec3(0.0, 0.0, delta)).r;
    return sample2 - sample1;
}

// hessian matrix
mat3 hessian(vec3 position)
{
	mat3 hess; 

	float delta = GRAD_DELTA;
	vec3 g1x = gradient(position - vec3(delta, 0.0, 0.0));
	vec3 g2x = gradient(position + vec3(delta, 0.0, 0.0));
    vec3 g1y = gradient(position - vec3(0.0, delta, 0.0));
    vec3 g2y = gradient(position + vec3(0.0, delta, 0.0));
    vec3 g1z = gradient(position - vec3(0.0, 0.0, delta));
    vec3 g2z = gradient(position + vec3(0.0, 0.0, delta));

	hess[0] = g2x - g1x;
	hess[1] = g2y - g1y;
	hess[2] = g2z - g1z;

	return transpose(hess);	
}

// approximate curvature of implicit surface
// https://en.wikipedia.org/wiki/Gaussian_curvature
float Kcurvature(vec3 position)
{
	mat3 hess = hessian(position);
	vec3 grad = gradient(position);
	mat4 upperMat;
	upperMat[0] = vec4(hess[0], grad.x);
	upperMat[1] = vec4(hess[1], grad.y);
	upperMat[2] = vec4(hess[2], grad.z);
	upperMat[3] = vec4(grad, 0.0);
	return determinant(upperMat) / pow(length(grad), 4);
}

float matTrace(mat3 m)
{
	return m[0][0] + m[1][1] + m[2][2];
}

float matFrobeniusNorm(mat3 m)
{
	float F = 0;
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			F += m[i][j] * m[i][j];
	return sqrt(F);
}

float KindlmannCurvature(vec3 position)
{
	mat3 H = hessian(position);
	vec3 g = gradient(position);
	vec3 n = normalize(g);
	mat3 P = mat3(1.0) - outerProduct(n, n);
	mat3 G = -P*H*P;
	float T = matTrace(G);
	float F = matFrobeniusNorm(G);
	float quadSolutionTerm = sqrt(2*F*F - T*T);
	float F2 = F*F;
	float k1 = (T + quadSolutionTerm)/2.0;
	float k2 = (T - quadSolutionTerm)/2.0;
	
	float K = (k1 + k2)/2;

	if (isnan(K)) return 0;

	return K; 
}

vec3 phongLighting(vec3 normalVec, vec3 lightVec, vec3 viewerVec, 
				vec3 ambientColor, vec3 diffuseColor, 
				vec3 specularColor, float specExponent)
{
	float dist = length(lightVec);
	vec3 N = normalize(normalVec);
	vec3 L = normalize(lightVec);
	vec3 C = normalize(viewerVec);
	
	diffuseColor *= max(0.0, dot(N, L));
	
	vec3 halfVec = normalize(L + C);
	float nxHalf = max(0.0, dot(N, halfVec));
	specularColor *= pow(nxHalf, specExponent);

	return ambientColor + diffuseColor + specularColor;
}

bool insideVolume(vec3 position, vec3 volMin, vec3 volMax)
{
	vec3 temp1 = sign(position - volMin);
	vec3 temp2 = sign(volMax - position);
	float inside = dot(temp1, temp2);
	if(inside < 3.0) return false;
	return true;
}

vec4 encodeFloatRGBA(float x)
{
	uint xint = floatBitsToUint(x);
	vec4 res;
	res.a = (xint >> 24 & 0xFF) / 255.0;
	res.b = (xint >> 16 & 0xFF) / 255.0;
	res.g = (xint >> 8 & 0xFF) / 255.0;
	res.r = (xint & 0xFF) / 255.0;
	return res;
}

struct IsosurfaceFeatures
{
	float gradMag;
	float gradOrientation;
	float curvature;
	float visibility;
};

struct Ray
{
	vec3 pos; // current position of ray
	vec3 dir; // ray direction
	float stepSize; // step size used when advancing ray
	vec3 advance; // amount the ray advances in each iteration, equal to stepSize * dir
};

Ray createRay(vec3 pos, vec3 dir, float stepSize)
{
	Ray ray;
	ray.pos = pos;
	ray.dir = dir;
	ray.stepSize = stepSize;
	ray.advance = dir * stepSize;
	return ray;
}

// check if ray intersects isosurface with index idx
// if intersection occurs, write intersection position intersPos
// otherwise, intersPos is irrelevant
bool rayIsosurfaceIntersection(Ray ray, int isoIdx, out vec3 intersPos)
{
	float dataSample = sampleDataAt(ray.pos);
	float nextDataSample = sampleDataAt(ray.pos + ray.advance);
	float isoDataVal = isoValues[5*isoIdx];
	if ((dataSample - isoDataVal) * (nextDataSample - isoDataVal) < 0)
	{
		// ray intersects isosurface i
		// approximate intersection position halfway between previous and next ray positions
		intersPos = ray.pos - ray.advance * 0.5;
		return true;
	}
	return false;
}

void main()
{
	vec3 rayInitPos = texCoord;
	vec3 rayInitDir = normalize(rayInitPos - viewerPos);
	
	if (rayJitterStrength > 0)
		rayInitPos += jitter(rayJitterStrength*0.01) * rayInitDir;	

	vec4 dst = vec4(0.0);
	
	int maxRaySamples = 2048;

	//surface features
	IsosurfaceFeatures isoFeat[MAX_NO_ISOSURFACES];
	for (int i = 0; i < MAX_NO_ISOSURFACES; i++)
	{
		isoFeat[i] = IsosurfaceFeatures(0, 0, 0, 0);
	}
	
	float occlusion = 0.0;
	float lightAbsorption = 0.0;

	vec3 lightContourColor = vec3(1.0);
	vec3 darkContourColor = vec3(0.0);

	// bit enconding of which isosurfaces have been intersected at least once
	// bit i is 1 if at least one intersection with isosurface i occurred
	// useful for determining various properties on first intersection only
	uint alreadyIntersected = 0; 

	//create ray for current fragment
	Ray ray = createRay(rayInitPos, rayInitDir, rayStepSize);

	for(int j = 0; j < maxRaySamples; j++)
	{
		// advance ray
		ray.pos += ray.advance;
		
		for (int i = 0; i < noIsosurfaces; i++)
		{
			vec3 isoPos; // position where the ray intersects isosurface i
			
			if (rayIsosurfaceIntersection(ray, i, isoPos))
			{
				// ray intersects isosurface i
				
				vec4 src = vec4(isoValues[5*i+1], isoValues[5*i+2], isoValues[5*i+3], isoValues[5*i+4]);
				float occlusionAlpha = src.a; // alpha value used to estimate occlusion

				vec3 grad = gradient(isoPos);

				// compute isosurface features

				if ((alreadyIntersected & (1 << i)) == 0)
				{
					// gradient magnitude
					isoFeat[i].gradMag = length(grad);

					// gradient angle
					if (isoFeat[i].gradMag == 0)
					{
						isoFeat[i].gradOrientation = 0;
					}
					else
					{
						vec3 gradNorm = normalize(grad);
						vec3 viewDir = normalize(isoPos - viewerPos);
						// cos similarity between gradient direction and viewer direction on isosurface i
						float cosSim = abs(dot(gradNorm, viewDir));
						isoFeat[i].gradOrientation = 1 - cosSim;						
					}
					
					// curvature
					float K = KindlmannCurvature(isoPos);
					isoFeat[i].curvature = abs(K);
										
					// visibility
					float vis = occlusionAlpha - occlusion;
					isoFeat[i].visibility = max(vis, 0);

					occlusion += occlusionAlpha;

					// curvature-based highlights
					if (enableSurfaceHighlights)
					{
						float kcoeff = K * 2.0 * surfaceHighlightStrength;
						if (kcoeff > 0)
							src.rgb = mix(src.rgb, lightContourColor, kcoeff);
						if (kcoeff < 0)
							src.rgb = mix(src.rgb, darkContourColor, -kcoeff);
					}
				}
					
				alreadyIntersected |= (1 << i); // mark isosurface i as having been intersected once

				if (enableShadows)
				{
					vec3 shRayInitPos = isoPos;
					vec3 shRayInitDir = normalize(lightPos - shRayInitPos);
					//create shadow ray
					Ray shRay = createRay(shRayInitPos, shRayInitDir, shadowRayStepSize);
					
					bool endShRay = false;
					lightAbsorption = 0.0;

					// start shadow ray loop
					for (int s = 0; s < maxRaySamples; s++)
					{
						// advance shadow ray
						shRay.pos += shRay.advance;

						if (!insideVolume(shRay.pos, volMin, volMax))
						break;
						
						for (int k = 0; k < noIsosurfaces; k++)
						{
							vec3 shIsoPos; // position of intersection between shadow ray and isosurface k

							if (rayIsosurfaceIntersection(shRay, k, shIsoPos))
							{
								float isoAlpha = isoValues[5*k+4];
								lightAbsorption += isoAlpha; // add shadow ray contribution

								if (lightAbsorption >= 0.9)
								{
									endShRay = true;
									break;
								}
							}
							
							if (endShRay)
								break;
						}
					}
					// end shadow ray loop

					src.rgb *= 1.0 - lightAbsorption * shadowOpacity;
				}

				float dataVal = sampleDataAt(isoPos); 

				if (enableLighting && dataVal > 0.01)
				{
					vec3 N = gradient(isoPos); 
					vec3 L = isoPos - lightPos;
					vec3 V = isoPos - viewerPos;
					
					vec3 ambient = vec3(ambientLightIntensity);
					vec3 diffuse = vec3(diffuseLightIntensity) * (1.0 + lightAmplification);
					vec3 specular = vec3(specularLightIntensity) * (1.0 + lightAmplification);
					float specExponent = specularPower * 256;
					src.rgb *= phongLighting(N, L, V, ambient, diffuse, specular, specExponent);
				}

				// add contribution of isosurface i to current pixel
				composit(dst, src);

			} // end isosurface intersection test

		} // end iterating through isosurfaces
		

		if (!insideVolume(ray.pos, volMin, volMax))
			break;

	} // end ray loop

	fragColor = vec4(mix(backColor.rgb, dst.rgb, dst.a), backColor.a);
	
	for (int i = 0; i < noIsosurfaces; i++)
		featValue[i] = vec4(isoFeat[i].gradMag, 
							isoFeat[i].gradOrientation, 
							isoFeat[i].curvature,
							isoFeat[i].visibility);

}
