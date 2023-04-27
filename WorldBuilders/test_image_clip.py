from omni.isaac.kit import SimulationApp
import numpy as np
import os
import random
import asyncio

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

def assembleMap(folder, stage, terrain_root = "/terrain", texture_path="../../../Textures/Sand.mdl", texture_name="Sand"):
    files = os.listdir(folder)
    createXform(stage, terrain_root)
    texture_path = "/home/lunar4/jnskkmhr/omn_asset/Terrain/Textures/Sand.mdl"
    material = loadTexture(stage, texture_path, texture_name, terrain_root+'/Looks/')
    for file in files:
        extenstion = file.split('.')[-1]
        if extenstion.lower() != "usd":
            continue
        name = file.split('.')[0]
        y_coord = int(name.split('_')[1])
        x_coord = int(name.split('_')[2])
        if extenstion.lower() == "usd":
            file_path = os.path.join(folder, file)
            createObject(os.path.join(terrain_root, name), stage, file_path, Gf.Vec3d(x_coord, y_coord, 0))
    terrain = stage.GetPrimAtPath(terrain_root)
    #pu.applyMaterial(terrain, material)
    setDefaultPrim(stage, terrain_root)

def processFolders(folders, stage):
    for folder in folders:
        assert os.path.exists(folder), "Path to folder: "+folder+" does not exist. Please correct it."
        name = folder.split("/")[-1]
        assembleMap(folder, stage)

async def save_stage(stage, is_save, save_path):
    if is_save:
        print(f'save to {save_path}')
        stage.save_as_stage(save_path, None) #world.stage = pxr.Usd.Stage

seed = 11
random.seed(seed)
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
setRotateXYZ(xform, Gf.Vec3d(20,85,0))

# assets = os.listdir(os.path.join(root_dir, asset_path))
# assets = [os.path.join(os.path.join(root_dir, asset_path), asset) for asset in assets if asset.split('.')[-1]=="usd"]
assets = ["/home/lunar4/jnskkmhr/omn_asset/rock_model/apollo-lunar-sample-1001715_USD/10017-15_SFM_Web-Resolution-Model_Coordinate-Registered.usd", 
          "/home/lunar4/jnskkmhr/omn_asset/rock_model/apollo-lunar-sample-1201311_USD/12013-11_SFM_Web-Resolution-Model_Coordinate-Registered.usd", 
          "/home/lunar4/jnskkmhr/omn_asset/rock_model/apollo-lunar-sample-1201311_USD/12013-11_SFM_Web-Resolution-Model_Coordinate-Registered"]
semantic_label_list = ["rock", "rock", "rock"]
createInstancerAndCache(stage, "/Rocks", assets, semantic_label_list)

# load sample dem file
# 1.5 m/pix
img_path = "/home/lunar4/jnskkmhr/IsaacSimEnvironmentGeneration/DEM/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500.npy"
data = np.load(img_path)
H, W = data.shape

base_height = data[0, 0]
offset = 0.5
data = np.flip(data, 0)
data = data - base_height - offset
mpp = 1.5
xmin = 0.0
xmax = H * mpp
ymin = 0.0
ymax = W * mpp

nx,ny = np.gradient(data)
slope_x = np.arctan2(nx,1) #theta_x = tan^-1(nx)
slope_y = np.arctan2(ny,1) #theta_y = tan^-1(ny)
magnitude = np.hypot(nx,ny) #slope norm (sqrt(slope_x^2 + slope_y^2))
flat_rank = np.argsort(magnitude.reshape(-1), axis=0) #sort by magnitude

rank_id1 = random.randint(150, 200)
rank_id2 = random.randint(150, 200)

pos1_flat = flat_rank[rank_id1]
x1 = pos1_flat % W
y1 = pos1_flat // W
pos1_flat_x = mpp * x1
pos1_flat_y = mpp * y1

pos2_flat = flat_rank[rank_id2]
x2 = pos1_flat % W
y2 = pos1_flat // W
pos2_flat_x = mpp * x2
pos2_flat_y = mpp * y2

# camera
camera_prim, _ = createXform(stage, "/camera")
camera_prim = UsdGeom.Xformable(camera_prim)
setTranslate(camera_prim, (pos1_flat_x, pos1_flat_y, data[y1, x1]-8.0))
setRotateXYZ(camera_prim, (0, 0, 0))
# camera = createCamera(prim_path="/camera/Camera", translation=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0), focus_distance=100.0, focal_length=2.4) #not working with replicator
camera = createCamera(stage=stage, prim_path="/camera/Camera", translation=Gf.Vec3d(0.0, 0.0, 0.0), orientation=Gf.Vec3d(90.0, 0.0, 0.0), focus_distance=100.0, focal_length=32.0)


# create sampler of rocks
sampler1 = NormalSampler_T(mean=(pos1_flat_x, pos1_flat_y), std=(xmax, ymax), randomization_space=2, seed=seed)
# sampler = UniformSampler_T(min=(xmin, ymin), max=(xmax, ymax), randomization_space=2, seed=77)
plane1 = Plane_T(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, output_space=2)
image_layer1 = Image_T(output_space=1)
image_clipper1 = ImageClipper_T(randomization_space=1, resolution=(H, W), mpp_resolution=mpp, data=data) #only z
normalmap_layer1 = NormalMap_T(output_space=4)
normalmap_clipper1 = NormalMapClipper_T(randomization_space=4, resolution=(H, W), mpp_resolution=mpp, data=data)
uni1 = UniformSampler_T(randomization_space=1)
line1 = Line_T(xmin=1.0, xmax=10.0)

req_pos_xy1 = UserRequest_T(p_type = Position_T(), sampler=sampler1, layer=plane1, axes=["x","y"])
req_pos_z1 = UserRequest_T(p_type = Position_T(), sampler=image_clipper1, layer=image_layer1, axes=["z"])
req_ori1 = UserRequest_T(p_type = Orientation_T(), sampler=normalmap_clipper1, layer=normalmap_layer1, axes=["x", "y", "z", "w"])
req_scale1 = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line1, axes=["xyz"])
requests1 = [req_pos_xy1, req_pos_z1, req_ori1, req_scale1]
mixer_1 = RequestMixer(requests1)


sampler2 = NormalSampler_T(mean=(pos2_flat_x, pos2_flat_y), std=(xmax, ymax), randomization_space=2, seed=seed)
plane2 = Plane_T(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, output_space=2)
image_layer2 = Image_T(output_space=1)
image_clipper2 = ImageClipper_T(randomization_space=1, resolution=(H, W), mpp_resolution=mpp, data=data) #only z
normalmap_layer2 = NormalMap_T(output_space=4)
normalmap_clipper2 = NormalMapClipper_T(randomization_space=4, resolution=(H, W), mpp_resolution=mpp, data=data)
uni2 = UniformSampler_T(randomization_space=1)
line2 = Line_T(xmin=1.0, xmax=10.0)

req_pos_xy2 = UserRequest_T(p_type = Position_T(), sampler=sampler2, layer=plane2, axes=["x","y"])
req_pos_z2 = UserRequest_T(p_type = Position_T(), sampler=image_clipper2, layer=image_layer2, axes=["z"])
req_ori2 = UserRequest_T(p_type = Orientation_T(), sampler=normalmap_clipper2, layer=normalmap_layer2, axes=["x", "y", "z", "w"])
req_scale2 = UserRequest_T(p_type = Scale_T(), sampler=uni2, layer=line2, axes=["xyz"])
requests2 = [req_pos_xy2, req_pos_z2, req_ori2, req_scale2]
mixer_2 = RequestMixer(requests2)


num = 500
attributes1 = mixer_1.executeGraph(num)
position1 = attributes1["xformOp:translation"]
scale1 = attributes1["xformOp:scale"]
orientation1 = attributes1["xformOp:orientation"]

attributes2 = mixer_2.executeGraph(num)
position2 = attributes2["xformOp:translation"]
scale2 = attributes2["xformOp:scale"]
orientation2 = attributes2["xformOp:orientation"]

position = np.concatenate([position1, position2], axis=0)
scale = np.concatenate([scale1, scale2], axis=0)
orientation = np.concatenate([orientation1, orientation2], axis=0)

setInstancerParameters(stage, "/Rocks", pos=position, quat=orientation, scale=scale)
addCollision(stage, "/Rocks")

simulation_context.play()

while(True):
    simulation_context.step(render=True)

is_save = False
save_path = os.path.join(root_dir, "Map/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500_stage.usd")
save_stage(stage, is_save, save_path)
simulation_app.close()