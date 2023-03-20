from omni.isaac.kit import SimulationApp
import numpy as np
import os

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from pxr_utils import createInstancerAndCache, setInstancerParameters, setRotateXYZ, createObject, addCollision
from Mixer import *
import omni
from pxr import UsdGeom, Sdf, UsdLux, Gf, UsdShade

root_dir = "/home/lunar4/jnskkmhr/omn_asset" # working directory
asset_path = "rock_model" # asset root dir

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

distant_light = UsdLux.DistantLight.Define(stage, "/sun")
distant_light.GetIntensityAttr().Set(10000)
xform = UsdGeom.Xformable(distant_light)
setRotateXYZ(xform, Gf.Vec3d(0,85,0))

assets = os.listdir(os.path.join(root_dir, asset_path))
assets = [os.path.join(os.path.join(root_dir, asset_path), asset) for asset in assets if asset.split('.')[-1]=="usd"]
createInstancerAndCache(stage, "/rocks", assets)

# load base terrain prim
terrain_prim_path = os.path.join(root_dir, "Terrain/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500_USD_clean.usd")
createObject("/terrain", stage, terrain_prim_path)
addCollision(stage, "/terrain")

# load sample dem file
# 1.5 m/pix
img_path = "/home/lunar4/jnskkmhr/IsaacSimEnvironmentGeneration/DEM/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500.npy"
data = np.load(img_path)
data = np.flip(data, 0)
# data = np.load("/home/lunar4/jnskkmhr/DEM/generated_tile.npy")
H, W = data.shape
mpp = 1.5

xmin = 0.0
xmax = H * mpp
ymin = 0.0
ymax = W * mpp

# sampler = NormalSampler_T(mean=(0.0, 0.0), std=(xmax/2, ymax/2), randomization_space=2, seed=77)
sampler = UniformSampler_T(min=(xmin, ymin), max=(xmax, ymax), randomization_space=2, seed=77)
plane = Plane_T(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, output_space=2)

image_layer = Image_T(output_space=1)
image_clipper = ImageClipper_T(randomization_space=1, resolution=(H, W), mpp_resolution=mpp, data=data) #only z

normalmap_layer = NormalMap_T(output_space=4)
normalmap_clipper = NormalMapClipper_T(randomization_space=4, resolution=(H, W), mpp_resolution=mpp, data=data)

uni1 = UniformSampler_T(randomization_space=1)
line = Line_T(xmin=30.0, xmax=32.0)

req_pos1 = UserRequest_T(p_type = Position_T(), sampler=sampler, layer=plane, axes=["x","y"])
req_pos2 = UserRequest_T(p_type = Position_T(), sampler=image_clipper, layer=image_layer, axes=["z"])
req_ori = UserRequest_T(p_type = Orientation_T(), sampler=normalmap_clipper, layer=normalmap_layer, axes=["x", "y", "z", "w"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line, axes=["xyz"])
requests = [req_pos1, req_pos2, req_ori, req_scale]
mixer = RequestMixer(requests)

num = 10000
attributes = mixer.executeGraph(num)
position = attributes["xformOp:translation"]
scale = attributes["xformOp:scale"]
orientation = attributes["xformOp:orientation"]
setInstancerParameters(stage, "/rocks", pos=position, quat=orientation, scale=scale)

while(True):
    world.step(render=True)

simulation_app.close()
