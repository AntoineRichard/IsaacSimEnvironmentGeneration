from omni.isaac.kit import SimulationApp
import numpy as np
import os

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from pxr_utils import createInstancerAndCache, setInstancerParameters
from Mixer import *
import omni

my_world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

assets = os.listdir("/home/antoine/Documents/Moon/IsaacSimulationFramework/SimulationEnvironments/Moon/Rocks/RoundRocks_USD_clean")
assets = [os.path.join("/home/antoine/Documents/Moon/IsaacSimulationFramework/SimulationEnvironments/Moon/Rocks/RoundRocks_USD_clean", asset) for asset in assets]
createInstancerAndCache(stage, "/rocks", assets)


matern3_polar = MaternClusterPointSampler_T(lambda_parent=0.0005, lambda_daughter=100.0, cluster_radius=0.1, randomization_space=3, use_rejection_sampling=False, warp=(0.1, 2*np.pi, 2*np.pi))
sphere = Sphere_T(center=(0,0,0), radius_min = 10.0, radius_max=100.0, theta_min = 0,theta_max=2*np.pi, phi_min=0, phi_max=2*np.pi, output_space=3)
uni1 = UniformSampler_T(randomization_space=1)
line = Line_T(xmin=0.5, xmax=1)
req_pos1 = UserRequest_T(p_type = Position_T(), sampler=matern3_polar, layer=sphere, axes=["x","y","z"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line, axes=["xyz"])
requests = [req_pos1, req_scale]
mixer = RequestMixer(requests)

while(True):
    attributes = mixer.executeGraph(10)
    position = attributes["xformOp:translation"]
    scale = attributes["xformOp:scale"]
    setInstancerParameters(stage, "/rocks", position, scale=scale)
    for i in range(1):
        my_world.step(render=True)

simulation_app.close()
