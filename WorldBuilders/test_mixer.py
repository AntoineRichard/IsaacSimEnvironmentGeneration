from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from Mixer import *

uni2 = UniformSampler_T(randomization_space=2)
uni1 = UniformSampler_T(randomization_space=1)
plane = Plane_T(xmin=-0.5,xmax=0.5,ymin=-2,ymax=2)
line = Line_T(xmin=1, xmax=2)
line2 = Line_T(xmin=0.5, xmax=1)

skip01 = True
skip02 = True
# Example 0.1: 2 requests.
# Duplicate axes requested in two different requests.
# Will throw an error
if not skip01:
    req_pos2 = UserRequest_T(p_type = Position_T(), sampler=uni2, layer=plane, axes=["x","y"])
    req_pos1 = UserRequest_T(p_type = Position_T(), sampler=uni1, layer=line, axes=["y"])

    requests = [req_pos1, req_pos2]

    mixer = RequestMixer(requests)
    attributes = mixer.executeGraph(10)

# Example 0.2: 1 request.
# Duplicate axes requested in a single request.
# Will throw an error
if not skip02:
    req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line2, axes=["xzz"])

    requests = [req_scale]

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


# Example 3: 2 requests
# the z axis in the position is not set by the user. It should default to its default value 0.
# the y axis in the scale is not set by the user. It should default to its default value 1.
req_pos1 = UserRequest_T(p_type = Position_T(), sampler=uni2, layer=plane, axes=["x","y"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line2, axes=["xz"])
requests = [req_pos1, req_scale]

mixer = RequestMixer(requests)
attributes = mixer.executeGraph(10)