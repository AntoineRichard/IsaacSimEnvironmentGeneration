import argparse
import asyncio
import omni
import os
from omni.isaac.kit import SimulationApp


async def convert(in_file, out_file, load_materials=False):
    # This import causes conflicts when global
    import omni.kit.asset_converter
    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # setup converter and flags
    converter_context.ignore_materials = not load_materials
    # converter_context.ignore_animation = False
    # converter_context.ignore_cameras = True
    # converter_context.single_mesh = True
    # converter_context.smooth_normals = True
    # converter_context.preview_surface = False
    # converter_context.support_point_instancer = False
    # converter_context.embed_mdl_in_usd = False
    converter_context.use_meter_as_world_unit = True
    # converter_context.create_world_as_default_root_prim = False
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


def asset_convert(folders, load_materials):
    supported_file_formats = ["stl", "obj", "fbx"]
    print(folders)
    for input_folder in folders:
        assert os.path.exists(input_folder), "The request directory: "+input_folder+" does not exist. Check the provided path."
        output_folder = input_folder + "_USD"
        os.makedirs(output_folder, exist_ok=True)

    for input_folder in folders:
        assert os.path.exists(input_folder), "The request directory: "+input_folder+" does not exist. Check the provided path."
        print("Converting folder "+input_folder+"...")
        models = os.listdir(input_folder)
        for model in models:
            model_name = '.'.join(model.split('.')[:-1])
            model_format = (model.split('.')[-1]).lower()
            if model_format in supported_file_formats:
                input_model_path = os.path.join(input_folder, model)
                output_model_path = os.path.join(input_folder + '_USD', model_name + ".usd")
                if not os.path.exists(output_model_path):
                    status = asyncio.get_event_loop().run_until_complete(
                        convert(input_model_path, output_model_path, load_materials)
                    )
                    if not status:
                        print(f"ERROR Status is {status}")
                    print("---Added "+model_name+".usd")


if __name__ == "__main__":
    kit = SimulationApp()

    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension("omni.kit.asset_converter")

    parser = argparse.ArgumentParser("Convert OBJ/STL assets to USD")
    parser.add_argument(
        "--folders", type=str, nargs="+", default=None, help="List of folders to convert (space seperated)."
    )
    parser.add_argument(
        "--load_materials", action="store_true", help="If specified, materials will be loaded from meshes"
    )
    args, unknown_args = parser.parse_known_args()

    if args.folders is None:
        raise ValueError(f"No folders specified via --folders argument")

    # Ensure Omniverse Kit is launched via SimulationApp before asset_convert() is called
    print(args.folders)
    asset_convert(args.folders, args.load_materials)

    # cleanup
    kit.close()
