from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from Mixer import *

uni2 = UniformSampler_T(randomization_space=2)
uni1 = UniformSampler_T(randomization_space=1)
plane = Plane_T(xmin=-0.5,xmax=0.5,ymin=-2,ymax=2)
line = Line_T(xmin=1, xmax=2)
line2 = Line_T(xmin=0.5, xmax=1)

sphere = Sphere_T(center=(0,0,0), radius_min = 10.0, radius_max=100.0, theta_min = 0,theta_max=2*np.pi, phi_min=0, phi_max=2*np.pi, output_space=3)

skip01 = True
skip02 = True
skip03 = True
# Example 0.1: 2 requests.
# Duplicate axes requested in two different requests.
# Will throw an error.
if not skip01:
    req_pos2 = UserRequest_T(p_type = Position_T(), sampler=uni2, layer=plane, axes=["x","y"])
    req_pos1 = UserRequest_T(p_type = Position_T(), sampler=uni1, layer=line, axes=["y"])

    requests = [req_pos1, req_pos2]

    mixer = RequestMixer(requests)
    attributes = mixer.executeGraph(10)

# Example 0.2: 1 request.
# Duplicate axes requested in a single request.
# Will throw an error.
if not skip02:
    req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line2, axes=["xzz"])

    requests = [req_scale]

    mixer = RequestMixer(requests)
    attributes = mixer.executeGraph(10)

# Example 3.3: 1 request
# The user request to project a plane on the y axis alone which is not possible.
# Will throw an error.
if not skip03:
    req_pos1 = UserRequest_T(p_type = Position_T(), sampler=uni2, layer=plane, axes=["y"])
    requests = [req_pos1]

    mixer = RequestMixer(requests)
    attributes = mixer.executeGraph(10)

# Example 1.1: 3 requests.
# The requests set all the axes.
# req_pos1 is sampling along a plane for axis x and y.
# req_pos2 is sampling along a line for axis z.
# They are merged automatically.
# req_scale specifies "xyz" this means that it will randomize for a single axis and duplicate the values to all axes.
# Or simply put, the scaling is going to be uniform along the three axes.
# The axes are listed in order, this is an ideal scenario.
req_pos1 = UserRequest_T(p_type = Position_T(), sampler=uni2, layer=plane, axes=["x","y"])
req_pos2 = UserRequest_T(p_type = Position_T(), sampler=uni1, layer=line, axes=["z"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line2, axes=["xyz"])

requests = [req_pos1, req_pos2, req_scale]

mixer = RequestMixer(requests)
attributes = mixer.executeGraph(10)

# Example 1.2: 3 requests.
# The requests set all the axes.
# req_pos1 is sampling along a line for axis z.
# req_pos2 is sampling along a plane for axis x and y. 
# Note how x and y are flipped. They will be reorganized automatically by the mixer.
# In the scale_request note how the axes are in a incorrect order. This doesn't change anything.
# This means that you cannot order the axes through request. You must shape your layers properly.
req_pos1 = UserRequest_T(p_type = Position_T(), sampler=uni1, layer=line, axes=["z"])
req_pos2 = UserRequest_T(p_type = Position_T(), sampler=uni2, layer=plane, axes=["y","x"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line2, axes=["zxy"])

requests = [req_pos1, req_pos2, req_scale]

mixer = RequestMixer(requests)
attributes = mixer.executeGraph(10)

# Example 2: 3 requests.
# The requests set all the axes.
# yz are merged on the position. Just like the scale this means that the mixer will make duplicate the y axis values to make the z axis.
req_pos1 = UserRequest_T(p_type = Position_T(), sampler=uni2, layer=plane, axes=["x","yz"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line2, axes=["xyz"])

requests = [req_pos1, req_scale]

mixer = RequestMixer(requests)
attributes = mixer.executeGraph(10)


# Example 3.1: 2 requests
# the z axis in the position is not set by the user. It should default to its default value 0.
# the y axis in the scale is not set by the user. It should default to its default value 1.
req_pos1 = UserRequest_T(p_type = Position_T(), sampler=uni2, layer=plane, axes=["x","y"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line2, axes=["xz"])
requests = [req_pos1, req_scale]

mixer = RequestMixer(requests)
attributes = mixer.executeGraph(10)

# Example 3.2: 2 requests
# the x and z axes in the position is not set by the user. It should default to its default value 0.
# the y and z axes in the scale is not set by the user. It should default to its default value 1.
req_pos1 = UserRequest_T(p_type = Position_T(), sampler=uni1, layer=line, axes=["y"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line2, axes=["x"])
requests = [req_pos1, req_scale]

mixer = RequestMixer(requests)
attributes = mixer.executeGraph(10)


# Example 4.1: 2 requests
# The position is using a matern cluster point process, there can be only one point process per set of requests.
# When using point processes, the number of points requested inside executeGraph is not used. 
# The point process will generate as many points as its density function requires.
# In our case here it's driven by two factors, the area of the sphere, lambda_parent and lambda_daughter.
# Lambda parent represents the density of clusters to be generated by unit of volume.
# Approx 2000 with our current settings.
# Then lambda_daughter is going to generate about 100 points per cluster in a radius of 0.1.
# Right now the radius is not scaled to the sphere but is instead applied on a unit space ((0,1),(0,1),(0,1)) for our sphere.
# Please note that this is true for all the polar primitives (Disk, Sphere, Cylinder, Cone, Torus).
# This is related to the way the pdf is projected onto the geometric primitives.
matern3_polar = MaternClusterPointSampler_T(lambda_parent=0.0005, lambda_daughter=100.0, cluster_radius=0.1, randomization_space=3, use_rejection_sampling=False, warp=(0.1, 2*np.pi, 2*np.pi))
req_pos1 = UserRequest_T(p_type = Position_T(), sampler=matern3_polar, layer=sphere, axes=["x","y","z"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line2, axes=["xyz"])
requests = [req_pos1, req_scale]

mixer = RequestMixer(requests)
attributes = mixer.executeGraph(10)


# Exampler 5: multiple requests merged.
matern2_polar = MaternClusterPointSampler_T(lambda_parent=0.5, lambda_daughter=200.0, cluster_radius=0.15, randomization_space=2, use_rejection_sampling=False, warp=(1, 0.2))
disk1 = Disk_T(center=(0,0), radius_min = 57.0, radius_max=60.0, theta_min = 0,theta_max=2*np.pi, output_space=2)
disk2 = Disk_T(center=(0,0), radius_min = 65.0, radius_max=67.0, theta_min = 0,theta_max=2*np.pi, output_space=2)
disk3 = Disk_T(center=(0,0), radius_min = 69.0, radius_max=79.0, theta_min = 0,theta_max=2*np.pi, output_space=2)
disk4 = Disk_T(center=(0,0), radius_min = 80.5, radius_max=84.0, theta_min = 0,theta_max=2*np.pi, output_space=2)
disk5 = Disk_T(center=(0,0), radius_min = 85.0, radius_max=87.0, theta_min = 0,theta_max=2*np.pi, output_space=2)
uni1 = UniformSampler_T(randomization_space=1)
line = Line_T(xmin=0.02, xmax=0.2)

req_pos1 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk1, axes=["x","y"])
req_pos2 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk2, axes=["x","y"])
req_pos3 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk3, axes=["x","y"])
req_pos4 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk4, axes=["x","y"])
req_pos5 = UserRequest_T(p_type = Position_T(), sampler=matern2_polar, layer=disk5, axes=["x","y"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line, axes=["xyz"])
requests1 = [req_pos1, req_scale]
requests2 = [req_pos2, req_scale]
requests3 = [req_pos3, req_scale]
requests4 = [req_pos4, req_scale]
requests5 = [req_pos5, req_scale]

mixer1 = RequestMixer(requests1)
mixer2 = RequestMixer(requests2)
mixer3 = RequestMixer(requests3)
mixer4 = RequestMixer(requests4)
mixer5 = RequestMixer(requests5)

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
scale = np.concatenate([scale1, scale2, scale3, scale4, scale5],axis=0)
points = np.concatenate([position1, position2, position3, position4, position5],axis=0)

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(points[:,0],points[:,1], points[:,2], "o")
ax.set_title("MaternClusterPointProcess in a Sphere")
plt.show()