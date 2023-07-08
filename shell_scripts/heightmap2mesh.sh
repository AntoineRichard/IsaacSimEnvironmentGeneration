#!/bin/bash
# cell based method
# export DEM_PATH=/home/lunar4/jnskkmhr/IsaacSimEnvironmentGeneration/DEM/NAC_DTM_ATLAS1_E473N0448_downsample_v2.tiff
export DEM_PATH=$1
# export DEM_PATH=/home/lunar4/jnskkmhr/IsaacSimEnvironmentGeneration/DEM/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500.tif
python3 Heightmap2Mesh/heightmap_to_meshes.py --dems $DEM_PATH \
                                              --save_paths ~/jnskkmhr/omn_asset/Terrain/ATLAS_Terrain_v4 \
                                              --save_extension OBJ \
                                              --generate_uvs 1 \
                                              --xy_resolution 1.0 \
                                              --z_scale 1.0 \
                                              --decimation_mode cell \
                                              --subtile_size 100 \
                                              --cut_subtiles 1 \