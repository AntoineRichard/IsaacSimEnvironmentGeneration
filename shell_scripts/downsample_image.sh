#!/bin/bash 

# h_min w_min h_max w_max
python3 downsample_image.py --img_path DEM/LRO_NAC_DEM_73N350E_150cmp.tif \
                            --down_scale 1.0 \
                            --save_path DEM/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500.tif \
                            --crop_range 3500 2000 4000 2500 \

python3 downsample_image.py --img_path DEM/LRO_NAC_Shade_73N350E_150cmp_texture.png \
                            --down_scale 1.0 \
                            --save_path DEM/LRO_NAC_DEM_73N350E_150cmp_texture_3500_4000_2000_2500.tif \
                            --crop_range 3500 2000 4000 2500 \