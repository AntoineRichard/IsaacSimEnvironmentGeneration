from omni.isaac.kit import SimulationApp
import numpy as np
import os

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from pxr_utils import createInstancerAndCache, setInstancerParameters, setRotateXYZ, createObject, addCollision, loadTexture, createStandaloneInstance, createXform, setDefaultPrim
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

root_dir = "/home/lunar4/jnskkmhr/omn_asset" # working directory
asset_path = "rock_model" # asset root dir

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# load base terrain prim
# terrain_prim_path = os.path.join(root_dir, "Terrain2/LRO_NAC_DEM_73N350E_150cmp_0_500_3000_3500_USD_clean.usd")
# createObject("/terrain", stage, terrain_prim_path)
# addCollision(stage, "/terrain")
# texture_path = "/home/lunar4/jnskkmhr/omn_asset/Terrain/Textures/Sand.mdl"
# texture_name = "Sand"
# material = loadTexture(stage, texture_path, texture_name, '/terrain/Looks')

terrain_folder = ["/home/lunar4/jnskkmhr/omn_asset/Terrain/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500_USD_clean"]
processFolders(terrain_folder, stage)

distant_light = UsdLux.DistantLight.Define(stage, "/sun")
distant_light.GetIntensityAttr().Set(7000)
xform = UsdGeom.Xformable(distant_light)
setRotateXYZ(xform, Gf.Vec3d(0,85,0))

# assets = os.listdir(os.path.join(root_dir, asset_path))
# assets = [os.path.join(os.path.join(root_dir, asset_path), asset) for asset in assets if asset.split('.')[-1]=="usd"]
assets = ["/home/lunar4/jnskkmhr/omn_asset/rock_model/apollo-lunar-sample-1001715_USD/10017-15_SFM_Web-Resolution-Model_Coordinate-Registered.usd", 
          "/home/lunar4/jnskkmhr/omn_asset/rock_model/apollo-lunar-sample-143211404_USD/14321-1404_SFM_Web-Resolution-Model_Coordinate-Registered.usd"]
createInstancerAndCache(stage, "/rocks", assets)

# load sample dem file
# 1.5 m/pix
img_path = "/home/lunar4/jnskkmhr/IsaacSimEnvironmentGeneration/DEM/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500.npy"
data = np.load(img_path)
H, W = data.shape

data = np.flip(data, 0)
# base_height = data[0, 0]
base_height = np.min(data)+1.0 # 0.5 is offset since data - base_height does not give you exact relative position of rock against terrain. (rock is still little bit high)
data = data - base_height
mpp = 1.5
xmin = 0.0
xmax = H * mpp
ymin = 0.0
ymax = W * mpp

sampler1 = NormalSampler_T(mean=(xmax/2, 2*ymax/3), std=(xmax/2, ymax/2), randomization_space=2, seed=77)
# sampler = UniformSampler_T(min=(xmin, ymin), max=(xmax, ymax), randomization_space=2, seed=77)
plane1 = Plane_T(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, output_space=2)
image_layer1 = Image_T(output_space=1)
image_clipper1 = ImageClipper_T(randomization_space=1, resolution=(H, W), mpp_resolution=mpp, data=data) #only z
normalmap_layer1 = NormalMap_T(output_space=4)
normalmap_clipper1 = NormalMapClipper_T(randomization_space=4, resolution=(H, W), mpp_resolution=mpp, data=data)
uni1 = UniformSampler_T(randomization_space=1)
line1 = Line_T(xmin=5.0, xmax=20.0)

req_pos_xy1 = UserRequest_T(p_type = Position_T(), sampler=sampler1, layer=plane1, axes=["x","y"])
req_pos_z1 = UserRequest_T(p_type = Position_T(), sampler=image_clipper1, layer=image_layer1, axes=["z"])
req_ori1 = UserRequest_T(p_type = Orientation_T(), sampler=normalmap_clipper1, layer=normalmap_layer1, axes=["x", "y", "z", "w"])
req_scale1 = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line1, axes=["xyz"])
requests1 = [req_pos_xy1, req_pos_z1, req_ori1, req_scale1]
mixer_1 = RequestMixer(requests1)


sampler2 = NormalSampler_T(mean=(3*xmax/4, 3*ymax/4), std=(xmax/2, ymax/2), randomization_space=2, seed=77)
plane2 = Plane_T(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, output_space=2)
image_layer2 = Image_T(output_space=1)
image_clipper2 = ImageClipper_T(randomization_space=1, resolution=(H, W), mpp_resolution=mpp, data=data) #only z
normalmap_layer2 = NormalMap_T(output_space=4)
normalmap_clipper2 = NormalMapClipper_T(randomization_space=4, resolution=(H, W), mpp_resolution=mpp, data=data)
uni2 = UniformSampler_T(randomization_space=1)
line2 = Line_T(xmin=5.0, xmax=20.0)

req_pos_xy2 = UserRequest_T(p_type = Position_T(), sampler=sampler2, layer=plane2, axes=["x","y"])
req_pos_z2 = UserRequest_T(p_type = Position_T(), sampler=image_clipper2, layer=image_layer2, axes=["z"])
req_ori2 = UserRequest_T(p_type = Orientation_T(), sampler=normalmap_clipper2, layer=normalmap_layer2, axes=["x", "y", "z", "w"])
req_scale2 = UserRequest_T(p_type = Scale_T(), sampler=uni2, layer=line2, axes=["xyz"])
requests2 = [req_pos_xy2, req_pos_z2, req_ori2, req_scale2]
mixer_2 = RequestMixer(requests2)


num = 1000
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

setInstancerParameters(stage, "/rocks", pos=position, quat=orientation, scale=scale)

world.play()

while(True):
    world.step(render=True)

simulation_app.close()
