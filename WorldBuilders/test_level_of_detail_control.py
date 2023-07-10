from omni.isaac.kit import SimulationApp
import numpy as np
import os
import json
import random
import asyncio
import argparse
import cv2

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.nucleus import get_server_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.extensions import enable_extension
from pxr_utils import createInstancerAndCache, setInstancerParameters, setRotateXYZ, setInstancerVisibilityFromFrustrum
from Mixer import *
from Types import *
import omni
from omni.kit.viewport.utility import get_active_viewport
from pxr import UsdGeom, Sdf, UsdLux, Gf, UsdShade

enable_extension('omni.isaac.ros2_bridge') #ROS2 Bridge

async def save_stage(stage, is_save, save_path):
    if is_save:
        print("*"*40)
        print(f'save to {save_path}')
        print("*"*40)
        stage.save_as_stage(save_path, None) #world.stage = pxr.Usd.Stage

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arg()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    root_dir = "/home/lunar4/jnskkmhr/omn_asset" # working directory
    asset_path = "rock_model" # asset root dir

    world = World(stage_units_in_meters=1.0)
    world.scene.add_ground_plane(size=1000)
    stage = omni.usd.get_context().get_stage()

    # light
    distant_light = UsdLux.DistantLight.Define(stage, "/sun")
    distant_light.GetIntensityAttr().Set(7000)
    xform = UsdGeom.Xformable(distant_light)
    setRotateXYZ(xform, Gf.Vec3d(0,85.0,0))

    assets_small = ["/home/lunar4/jnskkmhr/omn_asset/rock_model/apollo-lunar-sample-143211404_USD/14321-1404_SFM_Web-Resolution-Model_Coordinate-Registered.usd"]
    semantic_label_list_small = [None]
    createInstancerAndCache(stage, "/Rocks_small", assets_small, semantic_label_list_small, add_collision=True)

    # load sample dem file
    data = np.zeros((1000, 1000), dtype=np.float64)
    H, W = data.shape
    mpp = 1.0
    xmin = 0.0
    xmax = H * mpp
    ymin = 0.0
    ymax = W * mpp


    # Create Robot
    asset_root = get_server_path()
    robot_path = os.path.join(asset_root, "lunar_sim/EX1_d435i.usd")
    robot_prim = add_reference_to_stage(robot_path, "/robot/EX1_d435i")
    xform_api = UsdGeom.XformCommonAPI(robot_prim)
    xform_api.SetTranslate(Gf.Vec3d(20.0, 0, 0))
    xform_api.SetRotate((0, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    # Create camera
    CAMERA_STAGE_PATH = "/robot/EX1_d435i/EX1/mast_cam_cover_link/D435i_OG_ros2/realsense/camera_base/left_camera_link/left_camera_optical_link"
    with open("WorldBuilders/camera_config/d435i_vga.json", "r") as f:
        d435i_calib = json.load(f)["D435i_VGA"]["rgb"]
    camera_prim = UsdGeom.Camera.Define(stage, os.path.join(CAMERA_STAGE_PATH, "Camera"))
    xform_api = UsdGeom.XformCommonAPI(camera_prim)
    xform_api.SetTranslate(Gf.Vec3d(0, 0, 0))
    xform_api.SetRotate((180, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    horizontal_apeture = d435i_calib["calibration"]["sensor_width"]
    vertical_apeture = d435i_calib["calibration"]["sensor_height"]
    focal_length = d435i_calib["calibration"]["focal_length"]
    focus_distance = d435i_calib["calibration"]["focus_distance"]
    clipping_range = d435i_calib["calibration"]["clipping_range"]

    camera_prim.GetHorizontalApertureAttr().Set(horizontal_apeture)
    camera_prim.GetVerticalApertureAttr().Set(vertical_apeture)
    camera_prim.GetProjectionAttr().Set("perspective")
    camera_prim.GetFocalLengthAttr().Set(focal_length)
    camera_prim.GetFocusDistanceAttr().Set(focus_distance)
    camera_prim.GetClippingRangeAttr().Set(Gf.Vec2d((clipping_range)))


    # Spawn objects
    scale_min = 3.0 
    scale_max = 5.0
    # sampler1 = NormalSampler_T(mean=(pos1_flat_x, pos1_flat_y), std=std, randomization_space=2, seed=seed)
    sampler = ThomasClusterSampler_T(lambda_parent=0.02, lambda_daughter=5, sigma=0.05, randomization_space=2, use_rejection_sampling=False, seed=seed)
    plane = Plane_T(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, output_space=2)
    image_layer = Image_T(output_space=1)
    image_clipper = ImageClipper_T(randomization_space=1, resolution=(H, W), mpp_resolution=mpp, data=data) #only z
    normalmap_layer = NormalMap_T(output_space=4)
    normalmap_clipper = NormalMapClipper_T(randomization_space=4, resolution=(H, W), mpp_resolution=mpp, data=data)
    uni1 = UniformSampler_T(randomization_space=1)
    line1 = Line_T(xmin=scale_min, xmax=scale_max)

    req_pos_xy = UserRequest_T(p_type = Position_T(), sampler=sampler, layer=plane, axes=["x","y"])
    req_pos_z = UserRequest_T(p_type = Position_T(), sampler=image_clipper, layer=image_layer, axes=["z"])
    req_ori = UserRequest_T(p_type = Orientation_T(), sampler=normalmap_clipper, layer=normalmap_layer, axes=["x", "y", "z", "w"])
    req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line1, axes=["xyz"])
    requests = [req_pos_xy, req_pos_z, req_ori, req_scale]
    mixer = RequestMixer(requests)


    # pass all the attribute from mixer to instancer
    num = 1
    attributes = mixer.executeGraph(num)
    position = attributes["xformOp:translation"]
    scale = attributes["xformOp:scale"]
    orientation = attributes["xformOp:orientation"]
    setInstancerParameters(stage, "/Rocks_small", pos=position, quat=orientation, scale=scale)

    # add render callback
    def render_callback(event):
        setInstancerVisibilityFromFrustrum(stage, \
                                           "/Rocks_small", \
                                           os.path.join(CAMERA_STAGE_PATH, "Camera")
                                           )
    world.add_render_callback("setLoD", render_callback)

    viewport = get_active_viewport()
    viewport.set_active_camera("/camera/Camera")
    world.reset()
    world.play()
    while simulation_app.is_running():
        world.step(render=True)
    
    simulation_app.close()