#!/bin/bash
# cell based method
python3 Heightmap2Mesh/heightmap_to_meshes.py --dems DEM/generated_tile.npy \
                                              --save_paths ~/jnskkmhr/omn_asset/Terrain/generated_tile \
                                              --save_extension OBJ \
                                              --generate_uvs 1 \
                                              --xy_resolution 1.0 \
                                              --z_scale 1.0 \
                                              --decimation_mode cell \
                                              --num_cells_x 976 \
                                              --num_cells_y 976 \
                                              --subtile_size 1.0 \
                                              --cut_subtiles 1 \