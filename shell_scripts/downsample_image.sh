#!/bin/bash 

# h_min w_min h_max w_max
# atlas image 2351 x 14240
# so crop out height (12240~14240)

# first one : 12600 300 14100 1600
# second one : 200 300 1700 1800

export DEM_PATH=/home/lunar4/jnskkmhr/IsaacSimEnvironmentGeneration/DEM/NAC_DTM_ATLAS1_E473N0448.tiff
export DEM_SAVEPATH=/home/lunar4/jnskkmhr/IsaacSimEnvironmentGeneration/DEM/NAC_DTM_ATLAS1_E473N0448_crop_gaussian_double.tiff
python3 downsample_image.py --img_path $DEM_PATH \
                            --down_scale 1.0 \
                            --save_path $DEM_SAVEPATH \
                            --crop_range 200 300 1700 1800 \
                            --upscale 2 \