#!/bin/bash
# cell based method
python3 Heightmap2Mesh/heightmap_to_meshes.py --dems DEM/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500.tif \
                                              --save_paths ~/jnskkmhr/omn_asset/Terrain/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500 \
                                              --textures DEM/LRO_NAC_DEM_73N350E_150cmp_texture_3500_4000_2000_2500.tif \
                                              --save_extension OBJ \
                                              --generate_uvs 1 \
                                              --xy_resolution 0.667 \
                                              --z_scale 1.0 \
                                              --decimation_mode cell \
                                              --num_cells_x 500 \
                                              --num_cells_y 500 \
                                              --subtile_size 15.0 \
                                              --cut_subtiles 1 \