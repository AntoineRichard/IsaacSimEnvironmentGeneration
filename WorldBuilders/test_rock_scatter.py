from omni.isaac.kit import SimulationApp
import numpy as np
import os
from glob import glob

simulation_app = SimulationApp({"headless": False})
import omni
from omni.isaac.core import World, PhysicsContext
from omni.isaac.core.utils.semantics import add_update_semantics
from pxr_utils import createInstancerAndCache, setInstancerParameters, setRotateXYZ, createObject, addCollision, loadTexture, createStandaloneInstance, createXform, setDefaultPrim, applyMaterial
from Mixer import *
from Types import *
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
    # applyMaterial(terrain, material.get_path())
    # add_update_semantics(prim=terrain, semantic_label="terrain") #apply semantic
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

# physics
# physics_context = PhysicsContext(backend="torch")
# physics_context.set_solver_type("TGS")
# physics_context.set_gravity(-9.8) # on Moon 1/6 G

terrain_folder = ["/home/lunar4/jnskkmhr/omn_asset/Terrain/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500_USD_clean"]
processFolders(terrain_folder, stage)
addCollision(stage, "/terrain")

distant_light = UsdLux.DistantLight.Define(stage, "/sun")
distant_light.GetIntensityAttr().Set(7000)
xform = UsdGeom.Xformable(distant_light)
setRotateXYZ(xform, Gf.Vec3d(0,85,0))

root_dir = "/home/lunar4/jnskkmhr/omn_asset" # working directory
asset_root = "rock_model" # asset root dir
assets_path = glob(os.path.join(root_dir, asset_root, "*_USD"))
assets = [glob(os.path.join(asset_path, "*.usd"))[0] for asset_path in assets_path]
# semantic_label_list = [f"rock_{id}" for id in range(len(assets))]
createInstancerAndCache(stage, "/rocks", assets)
# load sample dem file
# 1.5 m/pix
img_path = "/home/lunar4/jnskkmhr/IsaacSimEnvironmentGeneration/DEM/LRO_NAC_DEM_73N350E_150cmp_3500_4000_2000_2500.npy"
data = np.load(img_path)
H, W = data.shape

data = np.flip(data, 0)
offset = 0.8
base_height = np.min(data)+offset
data = data - base_height
mpp = 1.5
xmin = 0.0
xmax = H * mpp
ymin = 0.0
ymax = W * mpp

sampler = NormalSampler_T(mean=(2*xmax/3, 2*ymax/3), std=(3*xmax, 3*ymax), randomization_space=2, seed=77)
# sampler = ThomasClusterSampler_T(lambda_parent=0.0005, lambda_daughter=10.0, sigma=0.1, randomization_space=2, seed=77)
# sampler = UniformSampler_T(min=(xmin, ymin), max=(xmax, ymax), randomization_space=2, seed=77)
plane = Plane_T(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, output_space=2)
image_layer = Image_T(output_space=1)
image_clipper = ImageClipper_T(randomization_space=1, resolution=(H, W), mpp_resolution=mpp, data=data) #only z
normalmap_layer = NormalMap_T(output_space=4)
normalmap_clipper = NormalMapClipper_T(randomization_space=4, resolution=(H, W), mpp_resolution=mpp, data=data)
uni = UniformSampler_T(randomization_space=1)
line = Line_T(xmin=3.0, xmax=6.0)

req_pos_xy = UserRequest_T(p_type = Position_T(), sampler=sampler, layer=plane, axes=["x","y"])
req_pos_z = UserRequest_T(p_type = Position_T(), sampler=image_clipper, layer=image_layer, axes=["z"])
req_ori = UserRequest_T(p_type = Orientation_T(), sampler=normalmap_clipper, layer=normalmap_layer, axes=["x", "y", "z", "w"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni, layer=line, axes=["xyz"])
requests = [req_pos_xy, req_pos_z, req_ori, req_scale]
mixer = RequestMixer(requests)


num = 1000
attributes = mixer.executeGraph(num)
position = attributes["xformOp:translation"]
scale = attributes["xformOp:scale"]
orientation = attributes["xformOp:orientation"]

setInstancerParameters(stage, "/rocks", pos=position, quat=orientation, scale=scale)


world.play()

while(True):
    world.step(render=True)

simulation_app.close()
