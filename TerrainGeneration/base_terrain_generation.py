import argparse
import os
import asyncio
import omni
from omni.isaac.kit import SimulationApp

def assembleMap(folder, output_path, texture_path, terrain_root = "/World/terrain", texture_name="Sand"):
    files = os.listdir(folder)
    pu.newStage()
    stage = omni.usd.get_context().get_stage()
    pu.createXform(stage, "/World")
    pu.createXform(stage, terrain_root)
    material = pu.loadTexture(stage, texture_path, texture_name, 'World/Looks')
    semantic_label = None
    for file in files:
        extenstion = file.split('.')[-1]
        if extenstion.lower() != "usd":
            continue
        name = file.split('.')[0]
        y_coord = int(name.split('_')[1])
        x_coord = int(name.split('_')[2])
        if extenstion.lower() == "usd":
            file_path = os.path.join(folder, file)
            z_offset = -44.4 # it seems there is offset between origin of terrain xprim and mesh (lower left corner has (0, 0, 0) position, but mesh is shifted by 44.4 in z axis, so we need to offset it by 44.4 to make it aligned with world origin
            pu.createObject(os.path.join(terrain_root, name), stage, file_path, Gf.Vec3d(x_coord, y_coord, z_offset), semantic_label=semantic_label)
            # terrain_mesh = stage.GetPrimAtPath(os.path.join(terrain_root, name, 'grp1'))
            # pu.applyMaterial(terrain_mesh, material)
    # pu.setDefaultPrim(stage, terrain_root)
    pu.saveStage(output_path)
    pu.closeStage()

def processFolders(folders, map_path, tex_path):
    for folder in folders:
        assert os.path.exists(folder), "Path to folder: "+folder+" does not exist. Please correct it."
        name = folder.split("/")[-1]
        assembleMap(folder, output_path=os.path.join(map_path,name+".usd"), texture_path=tex_path)
        

if __name__ == "__main__":
    kit = SimulationApp()
    from pxr import UsdGeom, Usd, Gf
    import pxr_utils as pu

    parser = argparse.ArgumentParser("Convert OBJ/STL assets to USD")
    parser.add_argument(
        "--folders", type=str, nargs="+", default=None, help="List of folders to convert (space seperated)."
    )
    parser.add_argument(
        "--maps_path", type=str, default=None, help="The path where the maps will be stored."
    )
    parser.add_argument("--tex_path", type=str, default=None)
    args, unknown_args = parser.parse_known_args()

    if args.folders is None:
        raise ValueError(f"No folders specified via --folders argument")

    processFolders(args.folders, args.maps_path, args.tex_path)

    kit.close()