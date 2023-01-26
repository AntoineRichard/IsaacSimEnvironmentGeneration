import dataclasses
import argparse
import warnings

@dataclasses.dataclass
class HM2MeshConfig:
    """
    Config for the HeightMap to Mesh classes.
    """
    # The path to the heightmap/dem to be loaded.
    hm_path: str = None
    # The path under which the file will be saved.
    save_path: str = None
    # The path to the texture file to be used.
    tex_path: str = None
    # The z scaling factor.
    z_scale: float = 1.0
    # The xy scaling factor.
    xy_resolution: float = 1.0
    # The extension to be used when saving the file.
    save_extension: str = "STL"
    # Whether or not the UVs should be generated when saving as OBJ.
    generate_uvs: bool = False
    # The maximum absolute error when using greedy decimation.
    absolute_error: float = 0.1
    # The number of cells to be used on the x and y axis when using cell or point decimation.
    num_cells: tuple = (0,0)
    # The the ratio of celles to be used on the x and y axis when using cell or point decimation.
    fxfy: tuple = (-1,-1)
    # Whether or not the mesh should be cutted into smaller meshes.
    cut_subtiles: bool = False
    # The size of the smaller meshes.
    subtile_size: float = 10

def loadFromArgs(args: argparse.Namespace):
    """
    Loads the arguments from the argparse parser.
    Inputs:
        args (obj): 
    """
    return HM2MeshConfig(hm_path=args.dems,
                         save_path=args.save_paths,
                         tex_path=args.textures,
                         z_scale=args.z_scale,
                         xy_resolution=args.xy_resolution,
                         save_extension=args.save_extension,
                         generate_uvs=args.generate_uvs,
                         absolute_error=args.absolute_error,
                         num_cells=(args.num_cells_x, args.num_cells_y),
                         fxfy=(args.cells_fx, args.cells_fy),
                         cut_subtiles=args.cut_subtiles,
                         subtile_size=args.subtile_size)

def loadFromDict(dct: dict):
    return HM2MeshConfig(**dct)

def parseArgs():
    parser = argparse.ArgumentParser("Generates meshes out of Digital Elevation Models (DEMs) or Heightmaps.")
    parser.add_argument("--dems", type=str, nargs="+", default=None, help="List of dems to be converted to meshes.")
    parser.add_argument("--save_paths", type=str, nargs="+", default=None, help="List of dems to be converted to meshes.")
    parser.add_argument("--textures", type=str, nargs="+", default=None, help="List of textures to be mapped onto the meshes.")
    parser.add_argument("--z_scale", type=float, default=1.0, help="The scaling factor to be applied on the z axis.")
    parser.add_argument("--xy_resolution", type=float, default=1.0, help="The resolution of the xy axes. In pixels per unit of distance.")
    parser.add_argument("--absolute_error", type=float, default=0.1, help="Used in greedy decimation mode. The maximum error between the reference mesh and the decimated mesh.")
    parser.add_argument("--num_cells_x", type=float, default=0.0, help="Used in cell or point decimation mode. The number of cell to be used along the x axis. This will force the resulting mesh to only have N cells along the x axis.")
    parser.add_argument("--num_cells_y", type=float, default=0.0, help="Used in cell or point decimation mode. The number of cell to be used along the x axis. This will force the resulting mesh to only have N cells along the y axis.")
    parser.add_argument("--cells_fx", type=float, default=-1.0, help="Used in cell or point decimation mode. The decimation ratio along the x axis. Equivalent to rescaling fxfy in opencv.")
    parser.add_argument("--cells_fy", type=float, default=-1.0, help="Used in cell or point decimation mode. The decimation ratio along the y axis. Equivalent to rescaling fxfy in opencv.")
    parser.add_argument("--decimation_mode", type=str, default="default", help="The type of decimation to be used when generating the mesh. Possible choises are: default, cell, point, greedy.")
    parser.add_argument("--save_extension", type=str, default="default", help="The extension to be used when saving the files. Possible choices are STL or OBJ.")
    parser.add_argument("--generate_uvs", type=int, default=1, help="Whether or not the UVs should be generated. Possible choices are 0 or 1. 0: no, 1: yes.")
    parser.add_argument("--cut_subtiles", type=int, default=0, help="Whether or not the heightmap should be cutted into smaller meshes. Possible choices are 0 or 1. 0: no, 1: yes.")
    parser.add_argument("--subtile_size", type=float, default=10.0, help="The size of the subtiles to be generated when cutting the main mesh.")
    args, unknown_args = parser.parse_known_args()
    print(args)
    return args, unknown_args

def checkArguments(args: argparse.Namespace):
    print(args.decimation_mode)
    assert args.decimation_mode in ["default", "greedy", "point", "cell"], "Requested decimation mode does not exist."
    assert args.subtile_size > 0, "Subtile_sizes must be larger than 0."
    if not args.textures is None:
        if len(args.textures) != len(args.dems):
            if len(args.textures) == 1:
                warnings.warn("Only 1 texture provided. Applying this texture to all the maps.")
            else:
                raise ValueError("The number of textures and dems is different. Please ensure the same number of dems and texures is provided.")
    if args.save_paths is None:
        raise ValueError("No save_paths provided. Please provide a least one save_paths.")
    else:
        if len(args.save_paths) != len(args.dems):
            raise ValueError("The number of save_paths and dems is different. Please ensure the same number of dems and save_paths is provided.")

    assert args.absolute_error > 0, "The absolute error must be a strictly positive value."
    assert args.num_cells_x >= 0, "The number of cells along the x axis must be positive."
    assert args.num_cells_y >= 0, "The number of cells along the y axis must be positive."
    assert (args.cells_fx > 0) or (args.cells_fx == -1), "The x scaling factor must be strictly positive or set to -1."
    assert (args.cells_fy > 0) or (args.cells_fy == -1), "The y scaling factor must be strictly positive or set to -1."
    assert args.save_extension.lower() in ["stl", "obj"], "The saving format must be either OBJ or STL."
    assert args.z_scale > 0, "The z scaling factor must be strictly positive."
    assert args.xy_resolution > 0, "The xy scaling factor must be strictly positive."