#!/bin/bash
# cell based method
python3 Heightmap2Mesh/heightmap_to_meshes.py --dems DEM/LRO_NAC_DEM_73N350E_150cmp_0_500_3000_3500.tif \
                                              --save_paths ~/jnskkmhr/omn_asset/Terrain2/LRO_NAC_DEM_73N350E_150cmp_0_500_3000_3500 \
                                              --save_extension OBJ \
                                              --generate_uvs 1 \
                                              --xy_resolution 0.667 \
                                              --z_scale 1.0 \
                                              --decimation_mode cell \
                                              --subtile_size 60 \
                                              --cut_subtiles 1 \