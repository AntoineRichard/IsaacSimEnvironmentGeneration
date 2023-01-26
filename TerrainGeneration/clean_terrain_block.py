import argparse
import asyncio
import omni
import os
from omni.isaac.kit import SimulationApp

def cleanTerrainBlock(source_path, output_path, enable_smooth):
    pu.loadStage(source_path)
    stage = omni.usd.get_context().get_stage()
    pu.deletePrim(stage, "/World/Looks")

    world_prim = stage.GetPrimAtPath("/World")
    if enable_smooth:
        for prim in Usd.PrimRange(world_prim):
            if prim.IsA(UsdGeom.Mesh):
                pu.enableSmoothShade(prim)

    childs = world_prim.GetChildren()
    map_path = childs[0].GetPath()
    map_name = childs[0].GetName()
    pu.addCollision(stage, map_path)
    pu.movePrim(map_path, "/"+map_name)
    pu.setDefaultPrim(stage, "/"+map_name)
    pu.deletePrim(stage, "/World")
    pu.saveStage(output_path)
    pu.closeStage()

def processFolders(folders, save_in_place=False, enable_smooth=False):
    for folder in folders:
        assert os.path.exists(folder), "Path to folder: "+folder+" does not exist. Please correct it."
        if save_in_place:
            output_folder = folder
        else:
            output_folder = folder + "_clean"
            os.makedirs(output_folder, exist_ok=True)

        files = os.listdir(folder)
        for file in files:
            extenstion = file.split('.')[-1]
            if extenstion.lower() == "usd":
                input_path = os.path.join(folder, file)
                output_path = os.path.join(output_folder, file)
                cleanTerrainBlock(input_path, output_path, enable_smooth)

if __name__ == "__main__":
    kit = SimulationApp()
    from pxr import UsdGeom, Usd
    import pxr_utils as pu

    parser = argparse.ArgumentParser("Convert OBJ/STL assets to USD")
    parser.add_argument(
        "--folders", type=str, nargs="+", default=None, help="List of folders to convert (space seperated)."
    )
    parser.add_argument(
        "--save_in_place", action="store_true", help="If specified, directly replaces the already existiing blocks."
    )
    parser.add_argument(
        "--enable_smooth", action="store_true", help="If specified, applies smooth shaders to the cleaned objects."
    )
    args, unknown_args = parser.parse_known_args()

    if args.folders is None:
        raise ValueError(f"No folders specified via --folders argument")

    processFolders(args.folders, args.save_in_place, args.enable_smooth)

    kit.close()