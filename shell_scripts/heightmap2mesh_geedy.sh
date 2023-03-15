#!/bin/bash
# cell based method
python3 Heightmap2Mesh/heightmap_to_meshes.py --dems DEM/LRO_NAC_DEM_73N350E_150cmp_2000_2300.tif \
                                              --save_paths ./mesh/LRO_NAC_DEM_73N350E_150cmp_2000_2300_greedy \
                                              --textures DEM/LRO_NAC_DEM_73N350E_150cmp_texture_2000_2300.png \
                                              --save_extension OBJ \
                                              --generate_uvs 1 \
                                              --xy_resolution 0.667 \
                                              --z_scale 1.0 \
                                              --decimation_mode greedy \
                                            #   --subtile_size 1.0 \
                                            #   --cut_subtiles 1 \ 