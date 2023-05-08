from omni.isaac.kit import SimulationApp
import numpy as np
import os
import random
import asyncio
import argparse

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core import SimulationContext
from pxr_utils import createInstancerAndCache, setInstancerParameters, \
                      setTranslate, setRotateXYZ, createObject, addCollision, \
                      loadTexture, createStandaloneInstance, createXform, setDefaultPrim, createCamera
from Mixer import *
from Types import *
import omni
from pxr import UsdGeom, Sdf, UsdLux, Gf, UsdShade
from utils import bilinear_interpolation, bicubic_interpolation, nearest_interpolation

async def save_stage(stage, is_save, save_path):
    if is_save:
        print("*"*40)
        print(f'save to {save_path}')
        print("*"*40)
        stage.save_as_stage(save_path, None) #world.stage = pxr.Usd.Stage

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--is_save', action='store_true')
    parser.add_argument('--dem_delta', type=float, required=True, help='DEM height delta')
    parser.add_argument('--cam_delta', type=float, required=True, help='cam height delta')
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

    # terrain
    terrain_prim_path = os.path.join(root_dir, "Map/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500_USD_clean.usd")
    omni.usd.get_context().open_stage(terrain_prim_path, None)
    simulation_context = SimulationContext(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    addCollision(stage, "/World/terrain")

    # light
    distant_light = UsdLux.DistantLight.Define(stage, "/sun")
    distant_light.GetIntensityAttr().Set(7000)
    xform = UsdGeom.Xformable(distant_light)
    setRotateXYZ(xform, Gf.Vec3d(0,85.0,0))

    # assets = os.listdir(os.path.join(root_dir, asset_path))
    # assets = [os.path.join(os.path.join(root_dir, asset_path), asset) for asset in assets if asset.split('.')[-1]=="usd"]
    assets_small = ["/home/lunar4/jnskkmhr/omn_asset/rock_model/apollo-lunar-sample-1001715_USD/10017-15_SFM_Web-Resolution-Model_Coordinate-Registered.usd", 
            "/home/lunar4/jnskkmhr/omn_asset/rock_model/apollo-lunar-sample-1201311_USD/12013-11_SFM_Web-Resolution-Model_Coordinate-Registered.usd", 
            "/home/lunar4/jnskkmhr/omn_asset/rock_model/apollo-lunar-sample-700178_USD/70017-8_SFM_Web-Resolution-Model_Coordinate-Registered.usd"]
    semantic_label_list_small = ["rock","rock","rock"]
    createInstancerAndCache(stage, "/Rocks_small", assets_small, semantic_label_list_small)

    assets_large = ["/home/lunar4/jnskkmhr/omn_asset/rock_model/apollo-lunar-sample-143211404_USD/14321-1404_SFM_Web-Resolution-Model_Coordinate-Registered.usd", 
            "/home/lunar4/jnskkmhr/omn_asset/rock_model/apollo-lunar-sample-606390_USD/60639-0_Web-Resolution_Coordinate-Registered.usd"]
    semantic_label_list_large = ["rock","rock"]
    createInstancerAndCache(stage, "/Rocks_large", assets_large, semantic_label_list_large)

    # load sample dem file
    # 1.5 m/pix
    img_path = "/home/lunar4/jnskkmhr/IsaacSimEnvironmentGeneration/DEM/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500.npy"
    data = np.load(img_path).astype(np.float64)
    base_height = data[0, 0]
    up_scale = 4
    data = bilinear_interpolation(data, up_scale)
    H, W = data.shape
    data = np.flip(data, 0)
    data = data - base_height - args.dem_delta
    mpp = 1.5/up_scale
    xmin = 0.0
    xmax = H * mpp
    ymin = 0.0
    ymax = W * mpp

    # nx,ny = np.gradient(data)
    # slope_x = np.arctan2(nx,1) #theta_x = tan^-1(nx)
    # slope_y = np.arctan2(ny,1) #theta_y = tan^-1(ny)
    # magnitude = np.hypot(nx,ny) #slope norm (sqrt(slope_x^2 + slope_y^2))
    # flat_rank = np.argsort(magnitude.reshape(-1), axis=0) #sort by magnitude

    # camera parameters
    # camera extrinsic
    cam_pos_x = 690.0
    cam_pos_y = 210.0
    # cam_pos_x = 500
    # cam_pos_y = 500
    cam_pos_z = data[int(cam_pos_y/mpp), int(cam_pos_x/mpp)]+args.cam_delta

    # calibration parameter
    # see https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html
    # focal_length = 1.88 # in mm (according to RealSense docs -> but seems odd)
    # fov = [69.4, 42.5]
    # focus_distance = 10.0
    # clipping_range = [0.0105, 10.0] # in m
    with open("WorldBuilders/camera_config/d435i.json", "r") as f:
        calib_params = json.load(f)

    # define camera at "/camera/Camera"
    camera = createCamera(
        stage=stage, 
        prim_path="/camera", 
        camera_name="Camera", 
        translation=Gf.Vec3d(pos1_flat_x, pos1_flat_y, data[y1, x1]+args.cam_delta), 
        orientation=Gf.Vec3d(80.0, 0.0, 0.0), 
        **calib_params, 
        )

    # setting of rock placement
    # create sampler of rocks

    # small rocks
    scale_min = 0.3 
    scale_max = 1.0
    sampler1 = ThomasClusterSampler_T(lambda_parent=0.05, lambda_daughter=10, sigma=0.05, randomization_space=2, use_rejection_sampling=False, seed=seed)
    plane1 = Plane_T(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, output_space=2)
    image_layer1 = Image_T(output_space=1)
    image_clipper1 = ImageClipper_T(randomization_space=1, resolution=(H, W), mpp_resolution=mpp, data=data) #only z
    normalmap_layer1 = NormalMap_T(output_space=4)
    normalmap_clipper1 = NormalMapClipper_T(randomization_space=4, resolution=(H, W), mpp_resolution=mpp, data=data) #orientation
    uni1 = UniformSampler_T(randomization_space=1)
    line1 = Line_T(xmin=scale_min, xmax=scale_max)

    req_pos_xy1 = UserRequest_T(p_type = Position_T(), sampler=sampler1, layer=plane1, axes=["x","y"])
    req_pos_z1 = UserRequest_T(p_type = Position_T(), sampler=image_clipper1, layer=image_layer1, axes=["z"])
    req_ori1 = UserRequest_T(p_type = Orientation_T(), sampler=normalmap_clipper1, layer=normalmap_layer1, axes=["x", "y", "z", "w"])
    req_scale1 = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line1, axes=["xyz"])
    requests1 = [req_pos_xy1, req_pos_z1, req_ori1, req_scale1]
    mixer_1 = RequestMixer(requests1)

    # large rocks
    scale_min = 6.0 
    scale_max = 8.0
    sampler2 = ThomasClusterSampler_T(lambda_parent=0.005, lambda_daughter=2, sigma=0.05, randomization_space=2, use_rejection_sampling=False, seed=seed)
    plane2 = Plane_T(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, output_space=2)
    image_layer2 = Image_T(output_space=1)
    image_clipper2 = ImageClipper_T(randomization_space=1, resolution=(H, W), mpp_resolution=mpp, data=data) #only z
    normalmap_layer2 = NormalMap_T(output_space=4)
    normalmap_clipper2 = NormalMapClipper_T(randomization_space=4, resolution=(H, W), mpp_resolution=mpp, data=data)
    uni2 = UniformSampler_T(randomization_space=1)
    line2 = Line_T(xmin=scale_min, xmax=scale_max)

    # put all the attribute into mixer
    req_pos_xy2 = UserRequest_T(p_type = Position_T(), sampler=sampler2, layer=plane2, axes=["x","y"])
    req_pos_z2 = UserRequest_T(p_type = Position_T(), sampler=image_clipper2, layer=image_layer2, axes=["z"])
    req_ori2 = UserRequest_T(p_type = Orientation_T(), sampler=normalmap_clipper2, layer=normalmap_layer2, axes=["x", "y", "z", "w"])
    req_scale2 = UserRequest_T(p_type = Scale_T(), sampler=uni2, layer=line2, axes=["xyz"])
    requests2 = [req_pos_xy2, req_pos_z2, req_ori2, req_scale2]
    mixer_2 = RequestMixer(requests2)


    # pass all the attribute from mixer to instancer
    num1 = 1
    attributes1 = mixer_1.executeGraph(num1)
    position1 = attributes1["xformOp:translation"]
    scale1 = attributes1["xformOp:scale"]
    orientation1 = attributes1["xformOp:orientation"]

    setInstancerParameters(stage, "/Rocks_small", pos=position1, quat=orientation1, scale=scale1)
    addCollision(stage, "/Rocks_small")

    num2 = 1
    attributes2 = mixer_2.executeGraph(num2)
    position2 = attributes2["xformOp:translation"]
    scale2 = attributes2["xformOp:scale"]
    orientation2 = attributes2["xformOp:orientation"]

    setInstancerParameters(stage, "/Rocks_large", pos=position2, quat=orientation2, scale=scale2)
    addCollision(stage, "/Rocks_large")

    # position = np.concatenate([position1, position2], axis=0)
    # scale = np.concatenate([scale1, scale2], axis=0)
    # orientation = np.concatenate([orientation1, orientation2], axis=0)

    # setInstancerParameters(stage, "/Rocks", pos=position, quat=orientation, scale=scale)
    # addCollision(stage, "/Rocks")

    save_path = os.path.join(root_dir, "Map/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500_D435i_stage1.usd")
    asyncio.ensure_future(save_stage(stage, args.is_save, save_path))
    simulation_context.play()

    while(True):
        simulation_context.step(render=True)
    simulation_app.close()