
VEParameters = {

    'dataset' : {
        'value' : 'datasets/bonsai.vdf',
        'dtype' : str,
        'domain' : None,
        'category' : 'data',
        'description' : 'volume dataset provided via a Volume Descriptor Format (.vdf) file'
        },
    
    'outputDir' : {
        'value' : '',
        'dtype' : str,
        'domain' : None,
        'category' : 'data',
        'description': 'output folder to store generated images'
        },

    'archiveSize' : {
        'value' : 200,
        'dtype' : int,
        'domain' : [10, 10000],
        'category' : 'evolutionary',
        'description' : 'maximum number of elite solutions in archive'
        },

    'batchSize' : {
        'value' : 10,
        'dtype' : int,
        'domain' : [5, 100],
        'category' : 'evolutionary',
        'description' : 'number of solutions managed by each emitter'
        },

    'emitterMean' : {
        'value' : 0.2,
        'dtype' : float,
        'domain' : [0.0, 1.0],
        'category' : 'evolutionary',
        'description' : 'initial mean value used by emitters'
        },

    'emitterSigma' : {
        'value' : 0.3,
        'dtype' : float,
        'domain' : [0.01, 1.0],
        'category' : 'evolutionary',
        'description' : 'standard deviation used by emitters'
        },

    'lightSourceMaxAngle' : {
        'value' : 60,
        'dtype' : float,
        'domain' : [1, 180],
        'category' : 'evolutionary',
        'description' : 'maximum angle to rotate light source in a cone around viewer direction'
        },

    'mainAxisMaxAngle' : {
        'value' : 180,
        'dtype' : float,
        'domain' : [1, 360],
        'category' : 'evolutionary',
        'description' : 'maximum angle to rotate viewer around main volume axis'
        },

    'maxNoIsosurfaces' : {
        'value' : 5,
        'dtype' : int,
        'domain' : [1, 7],
        'category' : 'evolutionary',
        'description' : 'maximum number of isosurfaces to search for'
        },

    'minNoIsosurfaces' : {
        'value' : 1,
        'dtype' : int,
        'domain' : [1, 7],
        'category' : 'evolutionary',
        'description' : 'minimum number of isosurfaces to search for'
        },

    'noEmitters' : {
        'value' : 3,
        'dtype' : int,
        'domain' : [1, 32],
        'category' : 'evolutionary',
        'description' : 'number of emitters used to generate and update solutions'
        },

    'noIterations' : {
        'value' : 10,
        'dtype' : int,
        'domain' : [10, 1000],
        'category' : 'evolutionary',
        'description' : 'number of evolutionary iterations'
        },

    'noWorkers' : {
        'value' : 1,
        'dtype' : int,
        'domain' : [1, 16],
        'category' : 'evolutionary',
        'description' : 'number of workers to use when computing features'
        },

    'secondaryAxisMaxAngle' : {
        'value' : 60,
        'dtype' : float,
        'domain' : [1, 360],
        'category' : 'evolutionary',
        'description' : 'maximum angle to rotate viewer around secondary volume axes'
        },

    'useCVTArchive' : {
        'value' : False,
        'dtype' : bool, 
        'domain' : [True, False],
        'category' : 'evolutionary',
        'description' : 'use CVT archive instead of grid archive'
        },

    'diversityFeatures' : {
        'value' : ['VisibilityBalance', 'DensitySpread', 'CurvatureEntropy'],
        'dtype' : list,
        'domain' : None,
        'category' : 'feature',
        'description' : 'features that define the diversity space'
        },

    'objectiveFeatures' : {
        'value' : ['Pique', 'ColorSpread', 'GradientShift'],
        'dtype' : list,
        'domain' : None,
        'category' : 'feature',
        'description' : 'features to be used as objectives'
        },

    'computeWidth' : {
        'value' : 128,
        'dtype' : int, 
        'domain' : [32, 1024],
        'category' : 'image',
        'description' : 'width of images used for feature computation'
        },

    'computeHeight' : {
        'value' : 128,
        'dtype' : int, 
        'domain' : [32, 1024],
        'category' : 'image',
        'description' : 'height of images used for feature computation'
        },

    'imageWidth': {
        'value' : 512,
        'dtype' : int,
        'domain' : [32, 4096],
        'category' : 'image',
        'description' : 'width of generated images, in pixels'
        },

    'imageHeight' : {
        'value' : 512, 
        'dtype' : int,
        'domain' : [32, 4096],
        'category' : 'image',
        'description' : 'height of generated images, in pixels'
        },

    'noDesiredImages' : {
        'value' : 25,
        'dtype' : int, 
        'domain' : [1, 256],
        'category' : 'image', 
        'description': 'number of desired output images (actual number may be lower)'
        },

    'showGeneratedImages' : {
        'value' : True,
        'dtype' : bool,
        'domain' : [True, False],
        'category' : 'image',
        'description' : 'display generated images in Pyplot window'
        },

    'ambientLightIntensity' : {
        'value' : 0.3, 
        'dtype' : float, 
        'domain' : [0.0, 1.0],
        'category' : 'rendering',
        'description' : 'intensity of ambient light'
        },

    'diffuseLightIntensity' : {
        'value' : 0.6, 
        'dtype' : float, 
        'domain' : [0.0, 1.0],
        'category' : 'rendering',
        'description' : 'intensity of diffuse light'
        },

    'enableLighting' : {
        'value' : True,
        'dtype' : bool,
        'domain' : [True, False],
        'category' : 'rendering',
        'description' : 'enable illumination when rendering' 
        },

    'enableShadows' : {
        'value' : False,
        'dtype' : bool,
        'domain' : [True, False],
        'category' : 'rendering',
        'description' : 'enable casting of shadows by objects'
        },

    'enableSurfaceHighlights' : {
        'value' : False,
        'dtype' : bool,
        'domain' : [True, False],
        'category' : 'rendering',
        'description' : 'enable the highlighting of variations on object surfaces'
        },

    'lightAmplification' : {
        'value' : 0.2,
        'dtype' : float,
        'domain' : [0.0, 1.0],
        'category' : 'rendering',
        'description' : 'amplification of overall lightsource brightness'
        },

    'rayJitterStrength' : {
        'value' : 0.0,
        'dtype' : float,
        'domain' : [0.0, 1.0],
        'category' : 'rendering',
        'description' : 'amount of randomization of sampling ray starting positions'
        },

    'rayStepSize' : {
        'value' : 0.001,
        'dtype' : float,
        'domain' : [0.001, 0.1],
        'category' : 'rendering',
        'description' : 'distance between consecutive samples along sampling ray'
        },

    'shadowOpacity' : {
        'value' : 1.0,
        'dtype' : float,
        'domain' : [0.0, 1.0],
        'category' : 'rendering',
        'description' : 'opacity of casted shadows'
        },

    'shadowRayStepSize' : {
        'value' : 0.005,
        'dtype' : float,
        'domain' : [0.001, 0.1],
        'category' : 'rendering',
        'description' : 'distance between consecutive samples along shadow ray'
        },

    'specularLightIntensity' : {
        'value' : 0.8, 
        'dtype' : float, 
        'domain' : [0.0, 1.0],
        'category' : 'rendering',
        'description' : 'intensity of specular light'
        },
    
    'specularPower' : {
        'value' : 0.6,
        'dtype' : float,
        'domain' : [0.0, 1.0],
        'category' : 'rendering',
        'description' : 'glossiness of surfaces with specular highlights'
        },

    'surfaceHighlightStrength' : {
        'value' : 0.3,
        'dtype' : float, 
        'domain' : [0.0, 1.0],
        'category' : 'rendering',
        'description' : 'visibility of surface highlights'
        },
    
    }




