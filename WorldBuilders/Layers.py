import numpy as np
from Types import *
from Samplers import *

class BaseLayer:
    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T, **kwarg) -> None:
        self._randomizer = None
        self._randomization_space = None

        self._sampler_cfg = sampler_cfg
        self._layer_cfg = layer_cfg

        if self._layer_cfg.output_space == self._sampler_cfg.randomization_space:
            self._skip_projection = True
        else:
            self._skip_projection = False

        if self._layer_cfg.transform is None:
            self._skip_transform = True
        else:
            self._skip_transform = False
            if isinstance(self._layer_cfg.transform, Transformation2D_T):
                assert self._layer_cfg.output_space == 2, "The output_shape must be equal to 2 to apply a 2D transform."
                self._T = self.buildTransform2D(self._layer_cfg.transform)
            else:
                assert self._layer_cfg.output_space == 3, "The output_shape must be equal to 3 to apply a 3D transform."
                self._T = self.buildTransform3D(self._layer_cfg.transform)
        
        self.getBounds()

    def initializeSampler(self) -> None:
        self._sampler = Sampler_Factory.get(self._sampler_cfg)

    def getBounds(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def buildRotMat3DfromQuat(xyzw: list) -> np.ndarray([3,3], dtype=float):
        q0 = xyzw[-1]
        q1 = xyzw[0]
        q2 = xyzw[1]
        q3 = xyzw[2]
        return 2*np.array([[q0*q0 + q1*q1, q1*q2 - q0*q3, q1*q3 + q0*q2],
                           [q1*q2 + q0*q3, q0*q0 + q2*q2, q2*q3 - q0*q1],
                           [q1*q3 - q0*q2, q2*q3 + q0*q1, q0*q0 + q3*q3]]) - np.eye(3)

    @staticmethod
    def buildRotMat2D(theta: float) -> np.ndarray([2,2], dtype=float):
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    @staticmethod
    def buildRotMat3DfromEuler(xyz: list) -> np.ndarray([3,3], dtype=float):
        cx = np.cos(xyz[0])
        sx = np.cos(xyz[0])
        cy = np.cos(xyz[1])
        sy = np.cos(xyz[1])
        cz = np.cos(xyz[2])
        sz = np.cos(xyz[2])
        return np.array([[cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sy],
                         [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
                         [-sy, sx*cy, cx*cy]])

    def buildTransform3D(self, trans: Transformation3D_T) -> np.ndarray([4,4], dtype=float):
        t = trans.translation
        T = np.zeros([4,4])
        if isinstance(self._layer_cfg.transform.orientation, Quaternion_T):    
            q = trans.orientation
            T[:3,:3] = self.buildRotMat3DfromQuat([q.x, q.y, q.z, q.w])
        else:
            euler = trans.orientation
            T[:3,:3] = self.buildRotMat3DfromEuler([euler.x, euler.y, euler.z])

        T[3,3] = 1
        T[:3,3] = np.array([t.x, t.y, t.z]) 
        return T
    
    def buildTransform2D(self, trans: Transformation2D_T) -> np.ndarray([3,3], dtype=float):
        t = trans.translation
        T = np.zeros([3,3])
        theta = trans.orientation.theta
        T[:2,:2] = self.buildRotMat2D(theta)
        T[2,2] = 1
        T[:2,2] = np.array([t.x, t.y]) 
        return T
    
    def project(self, points: np.ndarray([])) -> np.ndarray([]):
        if self._skip_projection:
            return points
        else:
            zeros = np.zeros([points.shape[0],self._layer_cfg.output_space - self._sampler_cfg.randomization_space])
            points = np.concatenate([points,zeros],axis=-1)
            return points

    def transform(self, points: np.ndarray([])) -> np.ndarray([]):
        if self._skip_transform:
            return points
        else:
            ones = np.ones([points.shape[0],1])
            points = np.concatenate([points,ones],axis=-1)
            proj = np.matmul(self._T,points.T).T[:,:-1]
            return proj

    def sample(self, num: int):
        raise NotImplementedError()

    def applyProjection(self, points:np.ndarray([])) -> np.ndarray([]):
        if self._skip_projection:
            return points
        else:
            return self.project(points)

    def applyTransform(self, points:np.ndarray([])) -> np.ndarray([]):
        if self._skip_transform:
            return points
        else:
            return self.transform(points)

    def __call__(self, num: int):
        points = self.sample(num)
        points = self.project(points)
        points = self.transform(points)
        return points

class Layer1D(BaseLayer):
    # Defines a 1D randomization space.
    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)

class Layer2D(BaseLayer):
    # Defines a 2D randomization space.
    def __init__(self, output_space: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(output_space, sampler_cfg)

class Layer3D(BaseLayer):
    # Defines a 3D randomization space.
    def __init__(self, output_space: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(output_space, sampler_cfg)

class Layer4D(BaseLayer):
    def __init__(self, output_space: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(output_space, sampler_cfg)

class LineLayer(Layer1D):
    def __init__(self, layer_cfg: Line_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 1
            self._sampler_cfg.min = [self._layer_cfg.xmin]
            self._sampler_cfg.max = [self._layer_cfg.xmax]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points):
        b1 = points[:,0] > self._layer_cfg.xmin
        b2 = points[:,0] < self._layer_cfg.xmax
        return b1*b2

    def getBounds(self):
        self._bounds = np.array([[self._layer_cfg.xmin, self._layer_cfg.xmax]])

    def createMask(self):
        pass

    def sample(self, num):
        return self._sampler(num=num, bounds=self._bounds)

class CircleLayer(Layer1D):
    def __init__(self, layer_cfg: Circle_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 1
            self._sampler_cfg.min = [self._layer_cfg.theta_min]
            self._sampler_cfg.max = [self._layer_cfg.theta_max]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

        if self._layer_cfg.output_space == (self._sampler_cfg.randomization_space + 1):
            self._skip_projection = True
        else:
            self._skip_projection = False

    def checkBoundaries(self, points):
        b1 = points[:,0] > self._layer_cfg.theta_min
        b2 = points[:,0] < self._layer_cfg.theta_max
        return b1*b2

    def getBounds(self):
        self._bounds = np.array([[self._layer_cfg.theta_min, self._layer_cfg.theta_max]])

    def createMask(self):
        pass

    def sample(self, num):
        theta = self._sampler(num=num, bounds=self._bounds)
        x = self._layer_cfg.center[0] + np.cos(theta)*self._layer_cfg.radius*self._layer_cfg.alpha
        y = self._layer_cfg.center[1] + np.sin(theta)*self._layer_cfg.radius*self._layer_cfg.beta
        return np.stack([x,y]).T[0]

    def project(self, points):
        if self._skip_projection:
            return points
        else:
            zeros = np.zeros([points.shape[0],self._layer_cfg.output_space - self._sampler_cfg.randomization_space -1])
            points = np.concatenate([points,zeros],axis=-1)
            return points

class PlaneLayer(Layer2D):
    def __init__(self, layer_cfg: Plane_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)


        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 2
            self._sampler_cfg.min = [self._layer_cfg.xmin, self._layer_cfg.ymin]
            self._sampler_cfg.max = [self._layer_cfg.xmax, self._layer_cfg.ymax]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries


    def checkBoundaries(self, points):
        b1 = points[:,0] > self._layer_cfg.xmin
        b2 = points[:,0] < self._layer_cfg.xmax
        b3 = points[:,1] > self._layer_cfg.ymin
        b4 = points[:,1] < self._layer_cfg.ymax
        return b1*b2*b3*b4

    def getBounds(self):
        self._bounds = np.array([[self._layer_cfg.xmin, self._layer_cfg.xmax],
                                [self._layer_cfg.ymin, self._layer_cfg.ymax]])

    def createMask(self):
        pass

    def sample(self, num):
        return self._sampler(num=num, bounds=self._bounds)

class DiskLayer(Layer2D):
    def __init__(self, layer_cfg: Disk_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 2
            self._sampler_cfg.min = [self._layer_cfg.radius_min**2, self._layer_cfg.theta_min]
            self._sampler_cfg.max = [self._layer_cfg.radius_max**2, self._layer_cfg.theta_max]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points):
        b1 = points[:,0] > self._layer_cfg.radius_min**2
        b2 = points[:,0] < self._layer_cfg.radius_max**2
        b3 = points[:,1] > self._layer_cfg.theta_min
        b4 = points[:,1] < self._layer_cfg.theta_max
        return b1*b2*b3*b4

    def getBounds(self):
        self._bounds = np.array([[self._layer_cfg.radius_min**2, self._layer_cfg.radius_max**2],
                                [self._layer_cfg.theta_min, self._layer_cfg.theta_max]])
        self._area = (self._layer_cfg.theta_max - self._layer_cfg.theta_min) * (self._layer_cfg.radius_max - self._layer_cfg.radius_min)**2

    def createMask(self):
        pass

    def sample(self, num):
        rand = self._sampler(num=num, bounds=self._bounds, area=self._area)
        x = self._layer_cfg.center[0] + np.cos(rand[:,1])*np.sqrt(rand[:,0])*self._layer_cfg.alpha
        y = self._layer_cfg.center[1] + np.sin(rand[:,1])*np.sqrt(rand[:,0])*self._layer_cfg.beta
        return np.stack([x,y]).T

class CubeLayer(Layer3D):
    def __init__(self, layer_cfg: Cube_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 3
            self._sampler_cfg.min = [self._layer_cfg.xmin, self._layer_cfg.ymin, self._layer_cfg.zmin]
            self._sampler_cfg.max = [self._layer_cfg.xmax, self._layer_cfg.ymax, self._layer_cfg.zmax]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points):
        b1 = points[:,0] > self._layer_cfg.xmin
        b2 = points[:,0] < self._layer_cfg.xmax
        b3 = points[:,1] > self._layer_cfg.ymin
        b4 = points[:,1] < self._layer_cfg.ymax
        b5 = points[:,2] > self._layer_cfg.zmin
        b6 = points[:,2] < self._layer_cfg.zmax
        return b1*b2*b3*b4*b5*b6

    def getBounds(self):
        self._bounds = np.array([[self._layer_cfg.xmin, self._layer_cfg.xmax],
                                [self._layer_cfg.ymin, self._layer_cfg.ymax],
                                [self._layer_cfg.zmin, self._layer_cfg.zmax]])

    def createMask(self):
        pass

    def sample(self, num):
        return self._sampler(num=num, bounds=self._bounds)

class SphereLayer(Layer3D):
    def __init__(self, layer_cfg: Sphere_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 3
            self._sampler_cfg.min = [self._layer_cfg.radius_min**2, self._layer_cfg.theta_min, self._layer_cfg.phi_min]
            self._sampler_cfg.max = [self._layer_cfg.radius_max**2, self._layer_cfg.theta_max, self._layer_cfg.phi_max]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points):
        b1 = points[:,0] > self._layer_cfg.radius_min**2
        b2 = points[:,0] < self._layer_cfg.radius_max**2
        b3 = points[:,1] > self._layer_cfg.theta_min
        b4 = points[:,1] < self._layer_cfg.theta_max
        b5 = points[:,2] > self._layer_cfg.phi_min
        b6 = points[:,2] < self._layer_cfg.phi_max
        return b1*b2*b3*b4*b5*b6

    def getBounds(self):
        self._bounds = np.array([[self._layer_cfg.radius_min**2, self._layer_cfg.radius_max**2],
                                [self._layer_cfg.theta_min, self._layer_cfg.theta_max],
                                [self._layer_cfg.phi_min, self._layer_cfg.phi_max]])
        self._area = (self._layer_cfg.theta_max - self._layer_cfg.theta_min) * (self._layer_cfg.phi_max - self._layer_cfg.phi_min) * (self._layer_cfg.radius_max - self._layer_cfg.radius_min)**2

    def createMask(self):
        pass

    def sample(self, num):
        rand = self._sampler(num=num, bounds=self._bounds, area=self._area)
        x = self._layer_cfg.center[0] + np.sin(rand[:,2])*np.cos(rand[:,1])*np.sqrt(rand[:,0])*self._layer_cfg.alpha
        y = self._layer_cfg.center[1] + np.sin(rand[:,2])*np.sin(rand[:,1])*np.sqrt(rand[:,0])*self._layer_cfg.beta
        z = self._layer_cfg.center[2] + np.cos(rand[:,2])*np.sqrt(rand[:,0])*self._layer_cfg.ceta
        return np.stack([x,y,z]).T

class CylinderLayer(Layer3D):
    def __init__(self, layer_cfg: Cylinder_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 3
            self._sampler_cfg.min = [self._layer_cfg.radius_min**2, self._layer_cfg.height_min, self._layer_cfg.theta_min]
            self._sampler_cfg.max = [self._layer_cfg.radius_max**2, self._layer_cfg.height_max, self._layer_cfg.theta_max]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points):
        b1 = points[:,0] > self._layer_cfg.radius_min**2
        b2 = points[:,0] < self._layer_cfg.radius_max**2
        b3 = points[:,1] > self._layer_cfg.height_min
        b4 = points[:,1] < self._layer_cfg.height_max
        b5 = points[:,2] > self._layer_cfg.theta_min
        b6 = points[:,2] < self._layer_cfg.theta_max
        return b1*b2*b3*b4*b5*b6

    def getBounds(self):
        self._bounds = np.array([[self._layer_cfg.radius_min**2, self._layer_cfg.radius_max**2],
                                [self._layer_cfg.height_min, self._layer_cfg.height_max],
                                [self._layer_cfg.theta_min, self._layer_cfg.theta_max]])
        self._area = (self._layer_cfg.theta_max - self._layer_cfg.theta_min) * (self._layer_cfg.height_max - self._layer_cfg.height_min) * (self._layer_cfg.radius_max - self._layer_cfg.radius_min)**2

    def createMask(self):
        pass

    def sample(self, num):
        rand = self._sampler(num=num, bounds=self._bounds, area=self._area)
        x = self._layer_cfg.center[0] + np.cos(rand[:,2])*np.sqrt(rand[:,0])*self._layer_cfg.alpha
        y = self._layer_cfg.center[1] + np.sin(rand[:,2])*np.sqrt(rand[:,0])*self._layer_cfg.beta
        z = rand[:,2]
        return np.stack([x,y,z]).T

class ImageLayer(Layer2D):
    def __init__(self, layer_cfg: Image_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)





class LayerFactory:
    def __init__(self):
        self.creators = {}
    
    def register(self, name: str, class_: BaseLayer) -> None:
        self.creators[name] = class_
        
    def get(self, cfg: Layer_T, **kwargs:dict) -> BaseLayer:
        if cfg.__class__.__name__ not in self.creators.keys():
            raise ValueError("Unknown layer requested.")
        return self.creators[cfg.__class__.__name__](cfg)

Layer_Factory = LayerFactory()
Layer_Factory.register("Line_T", LineLayer)
Layer_Factory.register("Circle_T", CircleLayer)
Layer_Factory.register("Plane_T", PlaneLayer)
Layer_Factory.register("Disk_T", DiskLayer)
Layer_Factory.register("Cube_T", CubeLayer)
Layer_Factory.register("Sphere_T", SphereLayer)
Layer_Factory.register("Cylinder_T", CylinderLayer)
Layer_Factory.register("Torus_T", ConeLayer)
Layer_Factory.register("Cone_T", TorusLayer)


class Spline(Layer1D):
    def __init__(self) -> None:
        super().__init__()
        # [[start, end]]
        # Rotation matrix

class SurfacePolygon(Layer2D):
    def __init__(self) -> None:
        super().__init__()

class SurfaceSphere(Layer2D):
    def __init__(self) -> None:
        super().__init__()

class SurfaceTorus(Plane_T):
    def __init__(self) -> None:
        super().__init__()

class Image(BaseLayer):
    def __init__(self) -> None:
        super().__init__()

class SemanticImage(Image):
    def __init__(self) -> None:
        super().__init__()

class FloatImage(Image):
    def __init__(self) -> None:
        super().__init__()


class Quaternion(Layer4D):
    def __init__(self) -> None:
        super().__init__()

