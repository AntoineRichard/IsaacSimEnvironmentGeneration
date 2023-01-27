import numpy as np
import dataclasses

##################################
#                                #
#           TRANSFORMS           # 
#                                #
##################################

@dataclasses.dataclass
class Orientation_T:
    pass

@dataclasses.dataclass
class Rot2D_T(Orientation_T):
    theta: float = 0

@dataclasses.dataclass
class Rot3D_T(Orientation_T):
    pass

@dataclasses.dataclass
class Quaternion_T(Rot3D_T):
    x: float = 0
    y: float = 0
    z: float = 0
    w: float = 1

@dataclasses.dataclass
class Euler_T(Rot3D_T):
    x: float = 0
    y: float = 0
    z: float = 0

@dataclasses.dataclass
class Translation_T:
    pass

@dataclasses.dataclass
class Translation2D_T(Translation_T):
    x: float = 0
    y: float = 0

@dataclasses.dataclass
class Translation3D_T(Translation_T):
    x: float = 0
    y: float = 0
    z: float = 0

@dataclasses.dataclass
class Transformation_T:
    orientation: Orientation_T
    translation: Translation_T

@dataclasses.dataclass
class Transformation2D_T(Transformation_T):
    orientation: Rot2D_T
    translation: Translation2D_T

    def __post_init__(self):
        assert isinstance(self.orientation, Rot2D_T), "A 2D transform requires a 2D rotation."
        assert isinstance(self.translation, Translation2D_T), "A 2D transform requires a 2D translation."

@dataclasses.dataclass
class Transformation3D_T(Transformation_T):
    orientation: Orientation_T
    translation: Translation3D_T

    def __post_init__(self):
        assert isinstance(self.orientation, Rot3D_T), "A 3D transform requires a 3D orientation."
        assert isinstance(self.translation, Translation3D_T), "A 3D transform requires a 3D translation."

##################################
#                                #
#              LAYERS            # 
#                                #
##################################

@dataclasses.dataclass
class Layer_T:
    output_space: int = 0
    transform: Transformation_T = None

    def __post_init__(self):
        assert self.output_space > 0, "output_space must be larger than 0."
        assert type(self.output_space) is int, "output_space must be an int."

@dataclasses.dataclass
class Line_T(Layer_T):
    xmax: float = None
    xmin: float = None
    output_space: int = 1

    def __post_init__(self):
        super().__post_init__()
        assert self.xmin <= self.xmax, "The maximum value along x must be larger than the minimum value."

@dataclasses.dataclass
class Circle_T(Layer_T):
    radius: float = None
    center: tuple = None
    theta_min: tuple = 0
    theta_max: tuple = np.pi*2
    alpha: float = 1
    beta: float = 1
    output_space: int = 2

    def __post_init__(self):
        super().__post_init__()
        assert self.output_space >= 2, "output_space must be greater or equal to 2."
        assert self.alpha > 0, "The alpha value must be larger than 0."
        assert self.beta > 0, "The beta value must be larger than 0."
        assert self.radius >= 0, "The radius must be larger than 0."
        assert self.theta_min >= 0, "The minimum value of theta must be larger than 0."
        assert self.theta_max <= np.pi*2, "The maximum value of theta must be smaller than 2pi."
        assert self.theta_max >= self.theta_min, "The maximum value of theta must be larger than self.theta_max."

@dataclasses.dataclass
class Plane_T(Layer_T):
    xmax: float = None
    xmin: float = None
    ymax: float = None
    ymin: float = None
    output_space: int = 2

    def __post_init__(self):
        super().__post_init__()
        assert self.output_space >= 2, "output_space must be greater or equal to 2."
        assert self.xmin <= self.xmax, "The maximum value along x must be larger than the minimum value."
        assert self.ymin <= self.ymax, "The maximum value along y must be larger than the minimum value."

@dataclasses.dataclass
class Disk_T(Layer_T):
    center: tuple = None
    radius_min: float = None
    radius_max: float = None
    theta_min: tuple = 0
    theta_max: tuple = np.pi*2
    alpha: float = 1
    beta: float = 1
    output_space: int = 2

    def __post_init__(self):
        super().__post_init__()
        assert self.output_space >= 2, "output_space must be greater or equal to 2."
        assert self.alpha > 0, "The alpha value must be larger than 0."
        assert self.beta > 0, "The beta value must be larger than 0."
        assert self.theta_min >= 0, "The minimum value of theta must be larger than 0."
        assert self.theta_max <= np.pi*2, "The maximum value of theta must be smaller than 2pi."
        assert self.theta_max >= self.theta_min, "The maximum value of theta must be larger than self.theta_max."
        assert self.radius_min >= 0, "The minimal radius must be larger than 0."
        assert self.radius_max >= self.radius_min, "The maximum radius must be larger than the minimum radius."

@dataclasses.dataclass
class Cube_T(Layer_T):
    xmax: float = None
    xmin: float = None
    ymax: float = None
    ymin: float = None
    zmax: float = None
    zmin: float = None
    output_space: int = 3
    transform: Transformation3D_T = None

    def __post_init__(self):
        super().__post_init__()
        if self.transform is not None:
            assert isinstance(self.transform, Transformation3D_T), "A 3D object, like a Cube, requires a 3D transform."
        assert self.output_space >= 3, "output_space must be greater or equal to 3."
        assert self.xmin <= self.xmax, "The maximum value along x must be larger than the minimum value."
        assert self.ymin <= self.ymax, "The maximum value along y must be larger than the minimum value."
        assert self.zmin <= self.zmax, "The maximum value along z must be larger than the minimum value."

@dataclasses.dataclass
class Sphere_T(Layer_T):
    center: tuple = None
    radius_min: float = 0
    radius_max: float = None
    theta_min: tuple = 0
    theta_max: tuple = np.pi*2
    phi_min: tuple = 0
    phi_max: tuple = np.pi*2
    alpha: float = 1
    beta: float = 1
    ceta: float = 1
    output_space: int = 3
    transform: Transformation3D_T = None

    def __post_init__(self):
        super().__post_init__()
        if self.transform is not None:
            assert isinstance(self.transform, Transformation3D_T), "A 3D object, like a Shpere, requires a 3D transform."
        assert self.output_space >= 3, "output_space must be greater or equal to 3."
        assert self.alpha > 0, "The alpha value must be larger than 0."
        assert self.beta > 0, "The beta value must be larger than 0."
        assert self.ceta > 0, "The ceta value must be larger than 0."
        assert self.theta_min >= 0, "The minimum value of theta must be larger than 0."
        assert self.theta_max <= np.pi*2, "The maximum value of theta must be smaller than 2pi."
        assert self.theta_max >= self.theta_min, "The maximum value of theta must be larger than self.theta_max."
        assert self.phi_min >= 0, "The minimum value of phi must be larger than 0."
        assert self.phi_max <= np.pi*2, "The maximum value of phi must be smaller than 2pi."
        assert self.phi_max >= self.theta_min, "The maximum value of phi must be larger than self.theta_max."
        assert self.radius_min >= 0, "The minimal radius must be larger than 0."
        assert self.radius_max >= self.radius_min, "The maximum radius must be larger than the minimum radius."

@dataclasses.dataclass
class Cylinder_T(Layer_T):
    center: tuple = None
    radius_min: float = 0
    radius_max: float = None
    theta_min: tuple = 0
    theta_max: tuple = np.pi*2
    height_min: tuple = 0
    height_max: tuple = np.pi*2
    alpha: float = 1
    beta: float = 1
    output_space: int = 3
    transform: Transformation3D_T = None

    def __post_init__(self):
        super().__post_init__()
        if self.transform is not None:
            assert isinstance(self.transform, Transformation3D_T), "A 3D object, like a Cylinder, requires a 3D transform."
        assert self.output_space >= 3, "output_space must be greater or equal to 3."
        assert self.alpha > 0, "The alpha value must be larger than 0."
        assert self.beta > 0, "The beta value must be larger than 0."
        assert self.theta_min >= 0, "The minimum value of theta must be larger than 0."
        assert self.theta_max <= np.pi*2, "The maximum value of theta must be smaller than 2pi."
        assert self.theta_max >= self.theta_min, "The maximum value of theta must be larger than self.theta_max."
        assert self.radius_min > 0, "The minimal radius must be larger than 0."
        assert self.radius_max > self.radius_min, "The maximum radius must be larger than the minimum radius."
        assert self.height_max > self.height_min, "The maximum height must be larger than the minimum height."


##################################
#                                #
#             SAMPLER            # 
#                                #
##################################

@dataclasses.dataclass
class Sampler_T:
    randomization_space: int = 0
    use_rejection_sampling: bool = False
    use_image_sampling: bool = False
    seed: int = -1
    max_rejection_sampling_loop: int = 5

    def __post_init__(self):
        assert self.randomization_space > 0, "randomization_space must be larger than 0."
        assert type(self.randomization_space) is int, "randomization_space must be an int."

@dataclasses.dataclass
class UniformSampler_T(Sampler_T):
    min: tuple = ()
    max: tuple = ()

    def __post_init__(self):
        super().__post_init__()
        assert type(self.min) is tuple, "min must be a tuple."
        assert type(self.max) is tuple, "max must be a tuple."

@dataclasses.dataclass
class NormalSampler_T(Sampler_T):
    mean: tuple = ()
    std: tuple = ()
    use_rejection_sampling = False

    def __post_init__(self):
        super().__post_init__()
        assert type(self.mean) is tuple, "mean must be a tuple."
        assert type(self.std) is tuple, "std must be a tuple."

        if len(self.std) == 1:
            self.std = np.eye(self.randomization_space) * self.std
        elif len(self.std) == len(self.mean):
            self.std = np.eye(self.randomization_space) * np.array(self.std)
        else:
            self.std = np.array(self.std).reshape(self.randomization_space,self.randomization_space)

@dataclasses.dataclass
class MaternClusterPointSampler_T(Sampler_T):
    lambda_parent: int = 10  # density of parent Poisson point process
    lambda_daughter: int = 100  # mean number of points in each cluster
    cluster_radius: float = 0.1  # radius of cluster disk (for daughter points) 

    def __post_init__(self):
        super().__post_init__()
        assert type(self.lambda_parent) is int, "lambda_parent must be an int."
        assert type(self.lambda_daughter) is int, "lambda_daughter must be an int."

@dataclasses.dataclass
class HardCoreMaternClusterPointSampler_T(Sampler_T):
    lambda_parent: int = 10  # density of parent Poisson point process
    lambda_daughter: int = 100  # mean number of points in each cluster
    cluster_radius: float = 0.1  # radius of cluster disk (for daughter points) 
    core_radius: float = 0.02
    num_repeat: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert type(self.lambda_parent) is int, "lambda_parent must be an int."
        assert type(self.lambda_daughter) is int, "lambda_daughter must be an int."

@dataclasses.dataclass
class ThomasClusterSampler_T(Sampler_T):
    lambda_parent: int = 10  # density of parent Poisson point process
    lambda_daughter: int = 100  # mean number of points in each cluster
    sigma: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        assert type(self.lambda_parent) is int, "lambda_parent must be an int."
        assert type(self.lambda_daughter) is int, "lambda_daughter must be an int."

@dataclasses.dataclass
class HardCoreThomasClusterSampler_T(Sampler_T):
    lambda_parent: int = 10  # density of parent Poisson point process
    lambda_daughter: int = 100  # mean number of points in each cluster
    sigma: float = 0.05 
    core_radius: float = 0.02
    num_repeat: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert type(self.lambda_parent) is int, "lambda_parent must be an int."
        assert type(self.lambda_daughter) is int, "lambda_daughter must be an int."

@dataclasses.dataclass
class PoissonPointSampler_T(Sampler_T):
    lambda_poisson: int = 100  # density of parent Poisson point process

    def __post_init__(self):
        super().__post_init__()
        assert type(self.lambda_poisson) is int, "lambda_poisson must be an int."

@dataclasses.dataclass
class LinearInterpolationSampler_T(Sampler_T):
    min: tuple = ()
    max: tuple = ()

    def __post_init__(self):
        super().__post_init__()
        assert type(self.min) is tuple, "min must be a tuple."
        assert type(self.max) is tuple, "max must be a tuple."