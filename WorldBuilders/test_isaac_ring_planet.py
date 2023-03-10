from omni.isaac.kit import SimulationApp
import numpy as np
import os

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from pxr_utils import createInstancerAndCache, setInstancerParameters, setRotateXYZ
from Mixer import *
import omni
from pxr import UsdGeom, Sdf, UsdLux, Gf, UsdShade
from omni.isaac.core.materials import PreviewSurface

my_world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()


assets = os.listdir("/home/antoine/Documents/Moon/IsaacSimulationFramework/SimulationEnvironments/Moon/Rocks/RoundRocks_USD_clean")
assets = [os.path.join("/home/antoine/Documents/Moon/IsaacSimulationFramework/SimulationEnvironments/Moon/Rocks/RoundRocks_USD_clean", asset) for asset in assets]
createInstancerAndCache(stage, "/rings1", assets)


color = np.array([235/255.0, 122/255.0, 52/255.0])
visual_prim_path = "/Looks/visual_material"
visual_material = PreviewSurface(prim_path=visual_prim_path, color=color)

sphere_geom = UsdGeom.Sphere.Define(stage, "/planet")
sphere_geom.GetRadiusAttr().Set(30)
sphere_prim = stage.GetPrimAtPath("/planet")
sphere_prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int)
sphere_prim.GetAttribute("refinementLevel").Set(3)
sphere_prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool)
sphere_prim.GetAttribute("refinementEnableOverride").Set(True)
UsdShade.MaterialBindingAPI(sphere_prim).Bind(visual_material.material, UsdShade.Tokens.strongerThanDescendants)

distant_light = UsdLux.DistantLight.Define(stage, "/sun")
distant_light.GetIntensityAttr().Set(1000)
xform = UsdGeom.Xformable(distant_light)
setRotateXYZ(xform, Gf.Vec3d(0,85,0))

matern2_polar = MaternClusterPointSampler_T(lambda_parent=0.5, lambda_daughter=200.0, cluster_radius=0.15, randomization_space=2, use_rejection_sampling=False, warp=(1, 0.2))
disk1 = Disk_T(center=(0,0), radius_min = 57.0, radius_max=60.0, theta_min = 0,theta_max=2*np.pi, output_space=2)
disk2 = Disk_T(center=(0,0), radius_min = 60.5, radius_max=62.0, theta_min = 0,theta_max=2*np.pi, output_space=2)
disk3 = Disk_T(center=(0,0), radius_min = 63.0, radius_max=64.5, theta_min = 0,theta_max=2*np.pi, output_space=2)
disk4 = Disk_T(center=(0,0), radius_min = 65.0, radius_max=67.0, theta_min = 0,theta_max=2*np.pi, output_space=2)
disk5 = Disk_T(center=(0,0), radius_min = 69.0, radius_max=79.0, theta_min = 0,theta_max=2*np.pi, output_space=2)
disk6 = Disk_T(center=(0,0), radius_min = 80.5, radius_max=84.0, theta_min = 0,theta_max=2*np.pi, output_space=2)
disk7 = Disk_T(center=(0,0), radius_min = 85.0, radius_max=87.0, theta_min = 0,theta_max=2*np.pi, output_space=2)
uni1 = UniformSampler_T(randomization_space=1)
line = Line_T(xmin=0.02, xmax=0.2)

req_pos1 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk1, axes=["x","y"])
req_pos2 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk2, axes=["x","y"])
req_pos3 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk3, axes=["x","y"])
req_pos4 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk4, axes=["x","y"])
req_pos5 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk5, axes=["x","y"])
req_pos6 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk6, axes=["x","y"])
req_pos7 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk7, axes=["x","y"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line, axes=["xyz"])
requests1 = [req_pos1, req_scale]
requests2 = [req_pos2, req_scale]
requests3 = [req_pos3, req_scale]
requests4 = [req_pos4, req_scale]
requests5 = [req_pos5, req_scale]
requests6 = [req_pos6, req_scale]
requests7 = [req_pos7, req_scale]


mixer1 = RequestMixer(requests1)
mixer2 = RequestMixer(requests2)
mixer3 = RequestMixer(requests3)
mixer4 = RequestMixer(requests4)
mixer5 = RequestMixer(requests5)
mixer6 = RequestMixer(requests6)
mixer7 = RequestMixer(requests7)


while(True):
    attributes1 = mixer1.executeGraph(10)
    position1 = attributes1["xformOp:translation"]
    scale1 = attributes1["xformOp:scale"]
    attributes2 = mixer2.executeGraph(10)
    position2 = attributes2["xformOp:translation"]
    scale2 = attributes2["xformOp:scale"]
    attributes3 = mixer3.executeGraph(10)
    position3 = attributes3["xformOp:translation"]
    scale3 = attributes3["xformOp:scale"]
    attributes4 = mixer4.executeGraph(10)
    position4 = attributes4["xformOp:translation"]
    scale4 = attributes4["xformOp:scale"]
    attributes5 = mixer5.executeGraph(10)
    position5 = attributes5["xformOp:translation"]
    scale5 = attributes5["xformOp:scale"]
    attributes6 = mixer6.executeGraph(10)
    position6 = attributes6["xformOp:translation"]
    scale6 = attributes6["xformOp:scale"]
    attributes7 = mixer7.executeGraph(10)
    position7 = attributes7["xformOp:translation"]
    scale7 = attributes7["xformOp:scale"]
    scale = np.concatenate([scale1, scale2, scale3, scale4, scale5, scale6, scale7],axis=0)
    position = np.concatenate([position1, position2, position3, position4, position5, position6, position7],axis=0)
    setInstancerParameters(stage, "/rings1", position, scale=scale)
    for i in range(100):
        my_world.step(render=True)

simulation_app.close()
