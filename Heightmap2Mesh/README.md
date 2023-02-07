# Heightmap to Mesh

The modules inside this code aim at easily generating meshes from Digital Elevation Maps (DEMs) and matching RGB textures.

## How to use

To use this code you need at least a DEM saved as either as a tif, png or npy.
Optionally, you can also provide a texture map, whose shape is matching the one of the DEM.
(If the sizes don't match the provided texture will be stretched).
 - The DEM or the list of DEMs are set using the `--dems` argument.
 - The textures or the list of textures are set using the `--textures` argument.
 - The path or list of paths to which the generated meshes should be saved to are set by the `--save_paths` argument.

Often DEMs are scaled. hence, to alleviate the need to rescale the mesh after the generation, the code allows the user to set scaling information.
There are two parameters for that:
 - `xy_resolution`: the resolution of the DEM in pixels per meter. (PPM)
 - `z_scale`: the scaling factor to be applied onto the z axis of the DEM. Note that the minimum height will be offset to 0.

Then you need to set the decimation algorithm you want to use. This is achieved by setting the `--decimation_mode` argument, as of now there are 4 options:
 - `default`: No decimation, takes the raw image and generates a mesh from it.
 - `point`: The point-based strategy sets the cell at the minimum height of the cell's points. Practical decimation tool as it creates regular cells. 
    - `num_cells_x`: The number of cell to be used along the x axis. This will force the resulting mesh to only have N cells along the x axis.
    - `num_cells_y`: The number of cell to be used along the x axis. This will force the resulting mesh to only have N cells along the y axis.
    - `cells_fx`: The decimation ratio along the x axis. Equivalent to rescaling fxfy in opencv.
    - `cells_fy`: The decimation ratio along the y axis. Equivalent to rescaling fxfy in opencv.
 - `cell`: The cell-based strategies sample the interior of the cell and place the cell at the minimum height of the cell's sampled interior points. Practical decimation tool as it creates regular cells.
    - `num_cells_x`: The number of cell to be used along the x axis. This will force the resulting mesh to only have N cells along the x axis.
    - `num_cells_y`: The number of cell to be used along the x axis. This will force the resulting mesh to only have N cells along the y axis.
    - `cells_fx`: The decimation ratio along the x axis. Equivalent to rescaling fxfy in opencv.
    - `cells_fy`: The decimation ratio along the y axis. Equivalent to rescaling fxfy in opencv.
 - `greedy`: Approximates a height field with a triangle mesh (triangulated irregular network - TIN) using a greedy insertion algorithm similar to that described by Garland and Heckbert in their paper "Fast Polygonal Approximations of Terrain and Height Fields". The most storage efficient algorithm. Associated parameters:
    - `absolute_error`: the maximum absolute error allowed when reconstructing the mesh.

With this decided you can select in which format you want the mesh to be saved to. This is done by using the `--save_extension`. Currently it supports two modes:
 - `STL`: A very common format to save 3D objects to. It does not support textures.
 - `OBJ`: A common format to save 3D objects to. It CAN support textures.

Please note that if you want to apply textures afterwards it may be beneficial to generate the UV maps during the mesh generation process.
This can be done by adding the following argument: `--generate_uvs 1`

Bellow are examples of how to use this module:
Single DEM, STL, with a 10pixels per meter resolution, and a downscale of z by a factor of 2:
```
python3 heightmap_to_meshes.py --dems PATH_TO_DEM\
                               --save_paths FOLDER_TO_SAVE_TO\
                               --save_extension STL\
                               --xy_resolution 10\
                               --z_scale 0.5
```
Single DEM, rescaled, OBJ, no UVs:
```
python3 heightmap_to_meshes.py --dems PATH_TO_DEM\
                               --save_paths FOLDER_TO_SAVE_TO\
                               --save_extension OBJ\
                               --xy_resolution 10\
                               --z_scale 0.5
```
Single DEM, recaled, OBJ, with UVs. The code will generate a dummy texture so you can visualize the tiles properly:
```
python3 heightmap_to_meshes.py --dems PATH_TO_DEM\
                               --save_paths FOLDER_TO_SAVE_TO\
                               --save_extension OBJ\
                               --generate_uvs 1\
                               --xy_resolution 10\
                               --z_scale 0.5
```
Single DEM, recaled, OBJ, with UVs and texture.
```
python3 heightmap_to_meshes.py --dems PATH_TO_DEM\
                               --textures PATH_TO_TEX\
                               --save_paths FOLDER_TO_SAVE_TO\
                               --save_extension OBJ\
                               --generate_uvs 1\
                               --xy_resolution 10\
                               --z_scale 0.5
```
Single DEM, rescaled, OBJ, UVs, cell decimation:
```
python3 heightmap_to_meshes.py --dems PATH_TO_DEM\
                               --save_paths FOLDER_TO_SAVE_TO\
                               --save_extension OBJ\
                               --generate_uvs 1\
                               --xy_resolution 10\
                               --z_scale 0.5\
                               --decimation_mode cell
                               --cells_fx 0.2\
                               --cells_fy 0.2
```
Single DEM, rescaled, OBJ, UVs, point decimation:
```
python3 heightmap_to_meshes.py --dems PATH_TO_DEM\
                               --save_paths FOLDER_TO_SAVE_TO\
                               --save_extension OBJ\
                               --generate_uvs 1\
                               --xy_resolution 10\
                               --z_scale 0.5\
                               --decimation_mode point\
                               --num_cells_x 1000\
                               --num_cells_y 1000
```
Batch DEM, rescaled, OBJ, UVs, greedy decimation:
```
python3 heightmap_to_meshes.py --dems PATH_TO_DEM1 PATH_TO_DEM2 PATH_TO_DEM3\
                               --save_paths FOLDER_TO_SAVE_TO1 FOLDER_TO_SAVE_TO2 FOLDER_TO_SAVE_TO3\
                               --save_extension OBJ\
                               --generate_uvs 1\
                               --xy_resolution 10\
                               --z_scale 0.5\
                               --decimation_mode point\
                               --num_cells_x 1000\
                               --num_cells_y 1000
```