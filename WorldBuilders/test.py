import numpy as np
import dataclasses
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from Types import *
from Layers import *
from Samplers import *
"""
class BaseSampler:
    def __init__(self, use_rejection_sampling=False, use_image_sampling=False):
        self.rng = np.random.default_rng()

        self.use_rejection_sampling = use_rejection_sampling
        self.use_image_sampling = use_image_sampling

        self.image = None
        self.offset = (None, None)

    def __call__(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def sample_equation_based_rejection(self):
        raise NotImplementedError()

    def mask_based_sampling(self):
        raise NotImplementedError()

class UniformRandomizer(BaseSampler):
    # Uniformly samples points.
    def __init__(self, min=[], max=[], dim=0, use_rejection_sampling=False, use_image_sampling=False, check_fn = lambda x: np.ones_like(x[:,0],dtype=bool), mask = None):
        super().__init__(self)
        assert (len(max) == dim), "The length of max must be a list of same length as dim, the number of specified dimensions."
        assert (len(min) == dim), "The length of min must be a list of same length as dim, the number of specified dimensions."

        self.min = min
        self.max = max
        self.dim = dim

        self.check_fn = check_fn
        self.mask = mask

    def setMaskAndOffset(self, mask: np.ndarray, offset: tuple, resolution: float):
        self.mask = mask.copy()
        self.offset = np.array(offset)
        self.resolution = resolution

        pd = self.mask * 1.0

        masked_pd = mask * pd
        self.idx = np.arange(masked_pd.flatten().shape[0])
        self.p = masked_pd / np.sum(masked_pd)

    def __call__(self, num = 1):
        if self.use_rejection_sampling:
            return self.sample_equation_based_rejection(num)
        elif self.use_image_sampling:
            return self.sample_image(num)
        else:
            return self.sample(num)

    def sample(self, num):
        points = np.stack([self.rng.uniform(self.min[dim], self.max[dim], (num)) for dim in range(self.dim)]).T
        correct = self.check_fn(points)      
        return points[correct]
    
    def sample_equation_based_rejection(self, num):
        points = []
        for i in range(num):
            check_not_ok = True
            while check_not_ok:
                point = [self.rng.normal(self.mean[dim], self.std[dim]) for dim in range(self.dim)]
                check_not_ok = not self.check_fn([point])[0]
            points.append(point)
        return np.array(points)

    def sample_image(self, num):
        idx = self.rng.choice(self.idx, p = self.p, size=num)
        local = self.rng.uniform(0, self.resolution, size=(num,self.dim))
        if self.dim == 2:
            x = idx // self.mask.shape[1]
            y = idx % self.mask.shape[1]
            return np.stack([x,y]).T + local
        if self.dim == 3:
            x = idx // self.mask.shape[2] // self.mask.shape[1]
            y = idx // self.mask.shape[2] % self.mask[1]
            z = idx % self.mask.shape[2] % self.mask.shape[1]
            return np.stack([x,y,z]).T + local

class NormalRandomizer(BaseSampler):
    # Uniformly samples points.
    def __init__(self, mean, std, dim=0, use_rejection_sampling = False, use_image_sampling = False, check_fn = lambda x: np.ones_like(x[:,0],dtype=bool), mask = None):
        super().__init__(self)
        assert (len(mean) == dim), "The length of mean must be a list of same length as dim, the number of specified dimensions."
        if len(std) != 1:
            assert (len(std) == dim**2), "The length of std must be a list of same length as the sqaure of dim, the number of specified dimensions."

        if len(std) == 1:
            self.sigma = np.eye(dim) * std
        else:
            self.sigma = np.array(std).reshape(dim,dim)

        self.mean = mean
        self.std = std
        self.dim = dim

        self.check_fn = check_fn
        self.mask = mask

    def multivariateGaussian(self, sigma, mu, pos):
        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        N = np.sqrt(((2*np.pi)**self.dim) * sigma_det)
        fac = np.einsum('...k,k1,...1->...', pos - mu, sigma_inv, pos - mu)
        return np.exp(-fac/2) / N

    def make2DGaussian(self, mask_shape, resolution):
        x = np.linspace(0, mask_shape[0] * resolution, mask_shape[0])
        y = np.linspace(0, mask_shape[1] * resolution, mask_shape[1])

        x,y = np.meshgrid(x,y)

        mu = np.array(self.mean)

        pos = np.empty(x.shape + (2,))
        pos[:,:,0] = x
        pos[:,:,1] = y
        return self.multivariateGaussian(self.sigma, mu, pos)

    def make3DGaussian(self, mask_shape, resolution):
        x = np.linspace(0, mask_shape[0] * resolution, mask_shape[0])
        y = np.linspace(0, mask_shape[1] * resolution, mask_shape[1])
        z = np.linspace(0, mask_shape[2] * resolution, mask_shape[2])

        x,y,z = np.meshgrid(x,y,z)

        mu = np.array(self.mean) + self.offset

        pos = np.empty(x.shape + (3,))
        pos[:,:,:,0] = x
        pos[:,:,:,1] = y
        pos[:,:,:,2] = z
        return self.multivariateGaussian(self.sigma, mu, pos)

    def setMaskAndOffset(self, mask: np.ndarray, offset: tuple, resolution: float):
        self.mask = mask.copy()
        self.offset = np.array(offset)
        self.resolution = resolution

        if self.dim == 2:
            pd = self.make2DGaussian(mask.shape, resolution)
        elif self.dim == 3:
            pd = self.make3DGaussian(mask.shape, resolution)
        else:
            raise ValueError("Image/Voxel processing not supported for inputs of this size")

        masked_pd = mask * pd
        self.idx = np.arange(masked_pd.flatten().shape[0])
        self.p = masked_pd / np.sum(masked_pd)

    def __call__(self, num = 1):
        if self.use_rejection_sampling:
            return self.sample_equation_based_rejection(num)
        elif self.use_image_sampling:
            return self.sample_image(num)
        else:
            return self.sample(num)
        
    def sample(self, num):
        if self.use_rejection_sampling:
            return self.sample_equation_based_rejection(num)
        else:
            points = self.rng.multivariate_normal(self.mean, self.sigma, (num))
            correct = self.check_fn(points)
            return points[correct]

    def sample_equation_based_rejection(self, num):
        points = []
        num_points = 0
        while num_points < num:
            pts = self.rng.multivariate_normal(self.mean, self.sigma, (num))
            correct = self.check_fn(pts)
            if np.sum(correct) > 0:
                print(np.sum(correct))
                print(pts[:5])
                print(correct[:5])
                num_points += np.sum(correct)
                points.append(pts[correct])
        points = np.concatenate(points)[:num]
        return np.array(points)

    def sample_image(self, num):
        idx = self.rng.choice(self.idx, p = self.p, size=num)
        local = self.rng.uniform(0, self.resolution, size=(num,self.dim))
        if self.dim == 2:
            x = idx // self.mask.shape[1]
            y = idx % self.mask.shape[1]
            return np.stack([x,y]).T + local
        if self.dim == 3:
            x = idx // self.mask.shape[2] // self.mask.shape[1]
            y = idx // self.mask.shape[2] % self.mask[1]
            z = idx % self.mask.shape[2] % self.mask.shape[1]
            return np.stack([x,y,z]).T + local
"""
class Mixer:
    def __init__(self, object_path, attribute_name, action_config, layer_config):
        pass

    def getPrim():
        pass

    def getPrimAttribute():
        pass

    def getPrimAttributeType():
        pass

    def Build():
        pass




trans = Translation3D_T(z=0.25)
quat =Quaternion_T(x=0.383,y=0.0,z=0.0,w=0.924)
quat2 =Quaternion_T(x=0.,y=0.383,z=0.0,w=0.924)
T = Transformation3D_T(translation=trans, orientation=quat)
T2 = Transformation3D_T(translation=trans, orientation=quat2)

uni1 = UniformSampler_T(randomization_space=1)
uni2 = UniformSampler_T(randomization_space=2)
uni3 = UniformSampler_T(randomization_space=3)

norm1 = NormalSampler_T(mean=(0.5,), std=(0.05,), randomization_space=1, use_rejection_sampling=True)
norm2 = NormalSampler_T(mean=(0.5,0.5), std=(0.05, 0.1, 0.1, 0.05), randomization_space=2, use_rejection_sampling=True)
norm3 = NormalSampler_T(mean=(0.5,0.5,0.5), std=(0.05, 0., 0., 0., 0.1, 0, 0., 0., 0.05), randomization_space=3, use_rejection_sampling=True)

norm1c = NormalSampler_T(mean=(np.pi,), std=(np.pi/8,), randomization_space=1, use_rejection_sampling=True)
norm2c = NormalSampler_T(mean=(np.pi, 0.25), std=(np.pi/8,0,0,0.02), randomization_space=2, use_rejection_sampling=True)
norm3c = NormalSampler_T(mean=(np.pi, np.pi, 0.25), std=(np.pi/8,0,0,0,np.pi/8,0,0,0,0.02), randomization_space=3, use_rejection_sampling=True)

matern2 = MaternClusterPointSampler_T(lambda_parent=10, lambda_daughter=100, cluster_radius=0.1, randomization_space=2, use_rejection_sampling=False)
matern3 = MaternClusterPointSampler_T(lambda_parent=10, lambda_daughter=100, cluster_radius=0.1, randomization_space=3, use_rejection_sampling=False)
hcmatern2 = HardCoreMaternClusterPointSampler_T(lambda_parent=10, lambda_daughter=100, cluster_radius=0.1, randomization_space=2, use_rejection_sampling=False)
hcmatern3 = HardCoreMaternClusterPointSampler_T(lambda_parent=10, lambda_daughter=100, cluster_radius=0.1, randomization_space=3, use_rejection_sampling=False)
#thomas2 = ThomasClusterSampler_T(lambda_parent=10, lambda_daughter=100, sigma=0.05, randomization_space=2, use_rejection_sampling=False)
#thomas3 = ThomasClusterSampler_T(lambda_parent=10, lambda_daughter=100, sigma=0.05, randomization_space=3, use_rejection_sampling=False)

line = Line_T(xmin=-0.5, xmax=0.5, transform=T2, output_space=3)
plane = Plane_T(xmin=-0.5,xmax=0.5,ymin=-0.5,ymax=0.5, transform=T, output_space=3)
cube = Cube_T(xmin=-0.5,xmax=0.5,ymin=-0.5,ymax=0.5,zmin=-0.5,zmax=0.5, transform=T2, output_space=3)
circle = Circle_T(center=(0,0), radius=0.5, theta_min = 1/8*np.pi, theta_max=7/4*np.pi, transform=T, alpha=1.5, output_space=3)
disk = Disk_T(center=(0,0), radius_min=0.1, radius_max=0.5, theta_min = 1/8*np.pi,theta_max=7/4*np.pi, transform=T2, beta=1.5, output_space=3)
sphere = Sphere_T(center=(0,0,0), radius_min = 0.1, radius_max=0.5, theta_min = 1/8*np.pi,theta_max=7/4*np.pi, phi_min=1/4*np.pi, phi_max=15/8*np.pi, output_space=3, transform=T2)


fig = plt.figure()

plane2 = Plane_T(xmin=-0.5,xmax=0.5,ymin=-0.5,ymax=0.5, transform=T, output_space=3)
plane_layer = PlaneLayer(plane2, matern2)
points = plane_layer(1000)
ax = fig.add_subplot(2,4,1, projection='3d')
ax.scatter(points[:,0],points[:,1], points[:,2], "o")
ax.set_title("MaternPointProcess on a Plane")

cube2 = Cube_T(xmin=-0.5,xmax=0.5,ymin=-0.5,ymax=0.5, zmin=-0.5, zmax=0.5, transform=T, output_space=3)
cube_layer = CubeLayer(cube2, matern3)
points = cube_layer(1000)
ax = fig.add_subplot(2,4,2, projection='3d')
ax.scatter(points[:,0],points[:,1], points[:,2], "o")
ax.set_title("MaternPointProcess in a Cube")

disk2 = Disk_T(center=(0,0), radius_min=0.0, radius_max=0.5, theta_min = 0,theta_max=2*np.pi, transform=T2, beta=1, output_space=3)
disk_layer = DiskLayer(disk2, matern2)
points = disk_layer(1000)
ax = fig.add_subplot(2,4,3, projection='3d')
ax.scatter(points[:,0],points[:,1], points[:,2], "o")
ax.set_title("MaternPointProcess on a Disk")

sphere2 = Sphere_T(center=(0,0,0), radius_min = 0.0, radius_max=0.5, theta_min = 0, theta_max=np.pi*2, phi_min=0, phi_max=2*np.pi, output_space=3, transform=T2)
sphere_layer = SphereLayer(sphere2, matern3)
points = sphere_layer(1000)
ax = fig.add_subplot(2,4,4, projection='3d')
ax.scatter(points[:,0],points[:,1], points[:,2], "o")
ax.set_title("MaternPointProcess in a Sphere")

plane2 = Plane_T(xmin=-0.5,xmax=0.5,ymin=-0.5,ymax=0.5, transform=T, output_space=3)
plane_layer = PlaneLayer(plane2, hcmatern2)
points = plane_layer(1000)
ax = fig.add_subplot(2,4,5, projection='3d')
ax.scatter(points[:,0],points[:,1], points[:,2], "o")
ax.set_title("HardCoreMaternPointProcess on a Plane")

cube2 = Cube_T(xmin=-0.5,xmax=0.5,ymin=-0.5,ymax=0.5, zmin=-0.5, zmax=0.5, transform=None, output_space=3)
plane_layer = CubeLayer(cube2, hcmatern3)
points = plane_layer(1000)
ax = fig.add_subplot(2,4,6, projection='3d')
ax.scatter(points[:,0],points[:,1], points[:,2], "o")
ax.set_title("HardCoreMaternPointProcess in a Cube")
disk2 = Disk_T(center=(0,0), radius_min=0.0, radius_max=0.5, theta_min = 0,theta_max=2*np.pi, transform=T2, beta=1, output_space=3)
disk_layer = DiskLayer(disk2, hcmatern2)
points = disk_layer(1000)
ax = fig.add_subplot(2,4,7, projection='3d')
ax.scatter(points[:,0],points[:,1], points[:,2], "o")
ax.set_title("HardCoreMaternPointProcess on a Disk")
sphere2 = Sphere_T(center=(0,0,0), radius_min = 0.0, radius_max=0.5, theta_min = 0, theta_max=np.pi*2, phi_min=0, phi_max=2*np.pi, output_space=3, transform=T2)
sphere_layer = SphereLayer(sphere2, hcmatern3)
points = sphere_layer(1000)
ax = fig.add_subplot(2,4,8, projection='3d')
ax.scatter(points[:,0],points[:,1], points[:,2], "o")
ax.set_title("HardCoreMaternPointProcess in a Sphere")

plt.show()

fig = plt.figure()
cube_layer = LineLayer(line, uni1)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,1, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Uniform on a Line")

cube_layer = LineLayer(line, norm1)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,2, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Multivariate Normal on a Line")

cube_layer = PlaneLayer(plane, uni2)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,3, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Uniform on a Plane")

cube_layer = PlaneLayer(plane, norm2)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,4, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Multivariate Normal on a Plane")

cube_layer = CubeLayer(cube, uni3)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,5, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Uniform on a Cube")

cube_layer = CubeLayer(cube, norm3)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,6, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Multivariate Normal on a Cube")

cube_layer = CircleLayer(circle, uni1)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,7, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Uniform on a Circle")

cube_layer = CircleLayer(circle, norm1c)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,8, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Multivariate Normal on a Circle")

cube_layer = DiskLayer(disk, uni2)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,9, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Uniform on a Disk")

cube_layer = DiskLayer(disk, norm2c)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,10, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Multivariate Normal on a Disk")

cube_layer = SphereLayer(sphere, uni3)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,11, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Uniform on a Sphere")

cube_layer = SphereLayer(sphere, norm3c)
points = cube_layer(1000)
ax = fig.add_subplot(2,6,12, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Multivariate Normal on a Sphere")
plt.show()




"""
import omni
from pxr import Gf
stage = omni.usd.get_context().get_stage()
cube = stage.GetPrimAtPath("/Environment/Cube")
attr = cube.GetAttributes()
print(attr)
extent= cube.GetAttribute("extent").Get()
extentType = cube.GetAttribute("extent").GetTypeName()
print(extent, extentType)

extent= cube.GetAttribute("xformOp:scale").Get()
extentS = cube.GetAttribute("xformOp:scale").Set
extentType = cube.GetAttribute("xformOp:scale").GetTypeName()
print(extent, extentType)
value = Gf.Vec3d(0.5,0.5,0.5)
extentS(value)
"""