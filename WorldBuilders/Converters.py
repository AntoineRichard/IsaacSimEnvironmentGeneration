from Types import *
from pxr import Usd
import numpy as np

def convert(prim: Usd.Prim) -> Layer_T:
    layer = None
    if prim.GetPrimTypeInfo.GetTypeName() == "Mesh":
        num_points = len(prim.GetAttribute("Points").Get())
        if num_points == 4:
            layer = PlaneMeshConverter(prim)
        elif num_points == 129:
            layer = DiskMeshConverter(prim)
        elif num_points == 8:
            layer = CubeMeshConverter(prim)
        elif num_points == 130:
            layer = CylinderMeshConverter(prim)
        elif num_points == 482:
            layer = SphereMeshConverter(prim)
        elif num_points == 193:
            layer = ConeMeshConverter(prim)
        elif num_points == 1089:
            layer = TorusMeshConverter(prim)
        else:
            layer = PlaneMeshConverter(prim)   
    else:
        if prim.GetPrimTypeInfo.GetTypeName() == "Cube":
            layer = CubeConverter(prim)
        elif prim.GetPrimTypeInfo.GetTypeName() == "Cylinder":
            layer = CylinderConverter(prim)
        elif prim.GetPrimTypeInfo.GetTypeName() == "Sphere":
            layer = SphereConverter(prim)
        elif prim.GetPrimTypeInfo.GetTypeName() == "Cone":
            layer = ConeConverter(prim)
        else:
            layer = PlaneMeshConverter(prim)
    return layer

def getScale(prim: Usd.Prim) -> tuple:
    try:
        scale = prim.GetAttribute("xformOp:scale").Get()
    except:
        scale = (1,1,1)
    return scale

def getOrientation(prim: Usd.Prim) -> tuple:
    try:
        orient = prim.GetAttribute("xformOp:orient").Get()
    except:
        orient = (1,0,0,0)
    return orient

def getTranslation(prim: Usd.Prim) -> tuple:
    try:
        trans = prim.GetAttribute("xformOp:translation").Get()
    except:
        trans = (0,0,0)
    return trans

def getTransform(prim: Usd.Prim) -> Transformation3D_T:
    q = getOrientation(prim)
    t = getTranslation(prim)
    quat = Quaternion_T(x=q[1], y=q[2], z=q[3], w=q[0])
    trans = Translation3D_T(x=t[0], y=t[1], z=t[2])
    return Transformation3D_T(orientation=quat, translation=trans)

def PlaneMeshConverter(prim: Usd.prim) -> Plane_T:
    scale = getScale(prim)
    xmin = -0.5 * scale[0]
    xmax = 0.5 * scale[0]
    ymin = -0.5 * scale[1]
    ymax = 0.5 * scale[1]
    T = getTransform(prim)
    return Plane_T(xmin=xmin,
                   xmax=xmax,
                   ymin=ymin,
                   ymax=ymax,
                   transform=T,
                   output_space=3)

def DiskMeshConverter(prim: Usd.Prim) -> Disk_T:
    scale = getScale(prim)
    r1 = np.min([scale[0], scale[1]])
    min_radius = 0
    max_radius = 0.5 * r1
    min_theta = 0
    max_theta = np.pi*2
    alpha = scale[0]/r1
    beta = scale[1]/r1
    T = getTransform(prim)
    return Disk_T(min_radius=min_radius,
                  max_radius=max_radius,
                  theta_min=min_theta,
                  theta_max=max_theta,
                  alpha=alpha,
                  beta=beta,
                  transform=T,
                  output_space=3)

def CubeMeshConverter(prim: Usd.Prim) -> Cube_T:
    scale = getScale(prim)
    xmin = -0.5 * scale[0]
    xmax = 0.5 * scale[0]
    ymin = -0.5 * scale[1]
    ymax = 0.5 * scale[1]
    zmin = -0.5 * scale[2]
    zmax = 0.5 * scale[2]
    T = getTransform(prim)
    return Cube_T(xmin=xmin,
                   xmax=xmax,
                   ymin=ymin,
                   ymax=ymax,
                   zmin=zmin,
                   zmax=zmax,
                   transform=T,
                   output_space=3)

def CylinderMeshConverter(prim: Usd.Prim) -> Cylinder_T:
    scale = getScale(prim)
    r1 = np.min([scale[0], scale[1]])
    min_radius = 0
    max_radius = 0.5 * r1
    min_height = -0.5 * scale[2]
    max_height = 0.5 * scale[2]
    min_theta = 0
    max_theta = np.pi*2
    alpha = scale[0]/r1
    beta = scale[1]/r1
    T = getTransform(prim)
    return Cylinder_T(min_radius=min_radius,
                  max_radius=max_radius,
                  min_height=min_height,
                  max_height=max_height,
                  theta_min=min_theta,
                  theta_max=max_theta,
                  alpha=alpha,
                  beta=beta,
                  transform=T,
                  output_space=3)

def SphereMeshConverter(prim: Usd.Prim) -> Sphere_T:
    scale = getScale(prim)
    r1 = np.min([scale[0], scale[1], scale[2]])
    min_radius = 0
    max_radius = 0.5 * r1
    min_theta = 0
    max_theta = np.pi*2
    min_phi = 0
    max_phi = np.pi*2

    alpha = scale[0]/r1
    beta = scale[1]/r1
    ceta = scale[2]/r1
    T = getTransform(prim)
    return Sphere_T(min_radius=min_radius,
                  max_radius=max_radius,
                  theta_min=min_theta,
                  theta_max=max_theta,
                  phi_min=min_phi,
                  phi_max=max_phi,
                  alpha=alpha,
                  beta=beta,
                  ceta=ceta,
                  transform=T,
                  output_space=3)

def ConeMeshConverter(prim: Usd.Prim) -> Cone_T:
    scale = getScale(prim)
    r1 = np.min([scale[0], scale[1]])
    min_radius = 0
    max_radius = 0.5 * r1
    min_height = -0.5
    max_height = 0.5
    min_theta = 0
    max_theta = np.pi*2
    alpha = scale[0]/r1
    beta = scale[1]/r1
    T = getTransform(prim)
    return Cone_T(min_radius=min_radius,
                  max_radius=max_radius,
                  min_height=min_height,
                  max_height=max_height,
                  theta_min=min_theta,
                  theta_max=max_theta,
                  alpha=alpha,
                  beta=beta,
                  transform=T,
                  output_space=3)

def TorusMeshConverter(prim: Usd.Prim) -> Torus_T:
    scale = getScale(prim)
    r1 = np.min([scale[0], scale[1]])
    min_radius2 = 0
    max_radius2 = 0.5 * r1
    min_radius1 = 0
    max_radius1 = 0.25 * r1
    min_theta1 = 0
    max_theta1 = np.pi*2
    min_theta2 = 0
    max_theta2 = np.pi*2
    alpha = scale[0]/r1
    beta = scale[1]/r1
    ceta = scale[2]/r1
    T = getTransform(prim)
    return Torus_T(min_radius1=min_radius1,
                  max_radius1=max_radius1,
                  min_radius2=min_radius2,
                  max_radius2=max_radius2,
                  theta1_min=min_theta1,
                  theta1_max=max_theta1,
                  theta2_min=min_theta2,
                  theta2_max=max_theta2,
                  alpha1=alpha,
                  beta1=beta,
                  alpha2=alpha,
                  beta2=beta,
                  ceta2=ceta,
                  transform=T,
                  output_space=3)

def CubeConverter(prim: Usd.Prim) -> Cube_T:
    scale = getScale(prim)
    size = prim.GetAttribute("size").Get()
    xmin = (-size/2) * scale[0]
    xmax = (size/2) * scale[0]
    ymin = (-size/2) * scale[1]
    ymax = (size/2) * scale[1]
    zmin = (-size/2) * scale[2]
    zmax = (size/2) * scale[2]
    T = getTransform(prim)
    return Cube_T(xmin=xmin,
                   xmax=xmax,
                   ymin=ymin,
                   ymax=ymax,
                   zmin=zmin,
                   zmax=zmax,
                   transform=T,
                   output_space=3)

def CylinderConverter(prim) -> Cylinder_T:
    scale = getScale(prim)
    radius = prim.GetAttribute("radius").Get()
    height = prim.GetAttribute("height").Get()
    axis = prim.GetAttribute("axis").Get()

    r1 = np.min([scale[0], scale[1]])
    min_radius = 0
    max_radius = radius * r1
    min_height = (-height/2) * scale[2]
    max_height = (height/2) * scale[2]
    min_theta = 0
    max_theta = np.pi*2
    alpha = scale[0] / r1
    beta = scale[1] / r1
    T = getTransform(prim)

    if axis == "X": # add a rotation of 90degrees on Y
        pass

    elif axis == "Y": # add a rotation of 90 degrees on X
        pass

    return Cylinder_T(min_radius=min_radius,
                  max_radius=max_radius,
                  min_height=min_height,
                  max_height=max_height,
                  theta_min=min_theta,
                  theta_max=max_theta,
                  alpha=alpha,
                  beta=beta,
                  transform=T,
                  output_space=3)

def SphereConverter(prim) -> Sphere_T:
    scale = getScale(prim)
    radius = prim.GetAttribute("radius").Get()
    r1 = np.min([scale[0], scale[1], scale[2]])
    min_radius = 0
    max_radius = radius * r1
    min_theta = 0
    max_theta = np.pi*2
    min_phi = 0
    max_phi = np.pi*2

    alpha = scale[0]/r1
    beta = scale[1]/r1
    ceta = scale[2]/r1
    T = getTransform(prim)
    return Sphere_T(min_radius=min_radius,
                  max_radius=max_radius,
                  theta_min=min_theta,
                  theta_max=max_theta,
                  phi_min=min_phi,
                  phi_max=max_phi,
                  alpha=alpha,
                  beta=beta,
                  ceta=ceta,
                  transform=T,
                  output_space=3)

def ConeConverter(prim) -> Cone_T:
    scale = getScale(prim)
    radius = prim.GetAttribute("radius").Get()
    height = prim.GetAttribute("height").Get()
    axis = prim.GetAttribute("axis").Get()


    r1 = np.min([scale[0], scale[1]])
    r2 = np.max([scale[0], scale[1]])
    r = r2/r1
    min_radius = 0
    max_radius = radius * r1
    min_height = (-height/2) * scale[2]
    max_height = (height/2) * scale[2]
    min_theta = 0
    max_theta = np.pi*2
    if r1 == scale[0]:
        alpha = 1
        beta = r
    else:
        alpha = r
        beta = 1
    T = getTransform(prim)

    if axis == "X": # add a rotation of 90degrees on Y
        pass

    elif axis == "Y": # add a rotation of 90 degrees on X
        pass

    return Cone_T(min_radius=min_radius,
                  max_radius=max_radius,
                  min_height=min_height,
                  max_height=max_height,
                  theta_min=min_theta,
                  theta_max=max_theta,
                  alpha=alpha,
                  beta=beta,
                  transform=T,
                  output_space=3)
