import numpy as np
from Types import *


class BaseSampler:
    def __init__(self, sampler_cfg: Sampler_T):
        self._sampler_cfg = sampler_cfg
        self._rng = np.random.default_rng()
        self.image = None
        self.offset = (None, None)
        self._check_fn = lambda x: np.ones_like(x[:,0],dtype=bool)

    def __call__(self, **kwargs):
        if self._sampler_cfg.use_rejection_sampling:
            return self.sample_equation_based_rejection(**kwargs)
        elif self._sampler_cfg.use_image_sampling:
            return self.sample_image(**kwargs)
        else:
            return self.sample(**kwargs)

    def sample(self, **kwargs):
        raise NotImplementedError()

    def sample_equation_based_rejection(self, **kwargs):
        raise NotImplementedError()

    def mask_based_sampling(self, **kwargs):
        raise NotImplementedError()


class UniformSampler(BaseSampler):
    # Uniformly samples points.
    def __init__(self, sampler_cfg: UniformSampler_T):
        super().__init__(sampler_cfg)

    def setMaskAndOffset(self, mask: np.ndarray, offset: tuple, resolution: float):
        self.mask = mask.copy()
        self.offset = np.array(offset)
        self.resolution = resolution

        pd = self.mask * 1.0

        masked_pd = mask * pd
        self.idx = np.arange(masked_pd.flatten().shape[0])
        self.p = masked_pd / np.sum(masked_pd)

    def sample(self, num=1, **kwargs):
        points = np.stack([self._rng.uniform(self._sampler_cfg.min[dim], self._sampler_cfg.max[dim], (num)) for dim in range(self._sampler_cfg.randomization_space)]).T
        correct = self._check_fn(points)      
        return points[correct]
    
    def sample_equation_based_rejection(self, num=1, **kwargs):
        points = []
        for i in range(num):
            check_not_ok = True
            while check_not_ok:
                point = [self._rng.uniform(self._sampler_cfg.min[dim], self._sampler_cfg.max[dim]) for dim in range(self._sampler_cfg.randomization_space)]
                check_not_ok = not self._check_fn([point])[0]
            points.append(point)
        return np.array(points)

    def sample_image(self, num=1, **kwargs):
        idx = self._rng.choice(self.idx, p = self.p, size=num)
        local = self._rng.uniform(0, self.resolution, size=(num,self._sampler_cfg.randomization_space))
        if self._sampler_cfg.randomization_space == 2:
            x = idx // self.mask.shape[1]
            y = idx % self.mask.shape[1]
            return np.stack([x,y]).T + local
        if self._sampler_cfg.randomization_space == 3:
            x = idx // self.mask.shape[2] // self.mask.shape[1]
            y = idx // self.mask.shape[2] % self.mask[1]
            z = idx % self.mask.shape[2] % self.mask.shape[1]
            return np.stack([x,y,z]).T + local


class NormalSampler(BaseSampler):
    # Uniformly samples points.
    def __init__(self, sampler_cfg: NormalSampler_T):
        super().__init__(sampler_cfg)

    def multivariateGaussian(self, sigma, mu, pos):
        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        N = np.sqrt(((2*np.pi)**self._sampler_cfg.randomization_space) * sigma_det)
        fac = np.einsum('...k,k1,...1->...', pos - mu, sigma_inv, pos - mu)
        return np.exp(-fac/2) / N

    def make2DGaussian(self, mask_shape, resolution):
        x = np.linspace(0, mask_shape[0] * resolution, mask_shape[0])
        y = np.linspace(0, mask_shape[1] * resolution, mask_shape[1])

        x,y = np.meshgrid(x,y)

        mu = np.array(self._sampler_cfg.mean)

        pos = np.empty(x.shape + (2,))
        pos[:,:,0] = x
        pos[:,:,1] = y
        return self.multivariateGaussian(self._sampler_cfg.std, mu, pos)

    def make3DGaussian(self, mask_shape, resolution):
        x = np.linspace(0, mask_shape[0] * resolution, mask_shape[0])
        y = np.linspace(0, mask_shape[1] * resolution, mask_shape[1])
        z = np.linspace(0, mask_shape[2] * resolution, mask_shape[2])

        x,y,z = np.meshgrid(x,y,z)

        mu = np.array(self._sampler_cfg.mean) + self.offset

        pos = np.empty(x.shape + (3,))
        pos[:,:,:,0] = x
        pos[:,:,:,1] = y
        pos[:,:,:,2] = z
        return self.multivariateGaussian(self._sampler_cfg.std, mu, pos)

    def setMaskAndOffset(self, mask: np.ndarray, offset: tuple, resolution: float):
        self.mask = mask.copy()
        self.offset = np.array(offset)
        self.resolution = resolution

        if self._sampler_cfg.randomization_space == 2:
            pd = self.make2DGaussian(mask.shape, resolution)
        elif self._sampler_cfg.randomization_space == 3:
            pd = self.make3DGaussian(mask.shape, resolution)
        else:
            raise ValueError("Image/Voxel processing not supported for inputs of this size")

        masked_pd = mask * pd
        self.idx = np.arange(masked_pd.flatten().shape[0])
        self.p = masked_pd / np.sum(masked_pd)
        
    def sample(self, num=0, **kwargs):
        points = self._rng.multivariate_normal(self._sampler_cfg.mean, self._sampler_cfg.std, (num))
        correct = self._check_fn(points)
        return points[correct]

    def sample_equation_based_rejection(self, num=0, **kwargs):
        points = []
        num_points = 0
        while num_points < num:
            pts = self._rng.multivariate_normal(self._sampler_cfg.mean, self._sampler_cfg.std, (num))
            correct = self._check_fn(pts)
            if np.sum(correct) > 0:
                num_points += np.sum(correct)
                points.append(pts[correct])
        points = np.concatenate(points)[:num]
        return np.array(points)

    def sample_image(self, num=0, **kwargs):
        idx = self._rng.choice(self.idx, p = self.p, size=num)
        local = self._rng.uniform(0, self.resolution, size=(num,self.dim))
        if self._sampler_cfg.randomization_space == 2:
            x = idx // self.mask.shape[1]
            y = idx % self.mask.shape[1]
            return np.stack([x,y]).T + local
        if self._sampler_cfg.randomization_space == 3:
            x = idx // self.mask.shape[2] // self.mask.shape[1]
            y = idx // self.mask.shape[2] % self.mask[1]
            z = idx % self.mask.shape[2] % self.mask.shape[1]
            return np.stack([x,y,z]).T + local


class MaternClusterPointSampler(BaseSampler):
    # Samples points in a layer defined space using a Matern cluser point process. 
    def __init__(self, sampler_cfg: MaternClusterPointSampler_T):
        super().__init__(sampler_cfg)

    def setMaskAndOffset(self, mask: np.ndarray, offset: tuple, resolution: float):
        self.mask = mask.copy()
        self.offset = np.array(offset)
        self.resolution = resolution

        pd = self.mask * 1.0

        masked_pd = mask * pd
        self.idx = np.arange(masked_pd.flatten().shape[0])
        self.p = masked_pd / np.sum(masked_pd)

    def getParents(self, bounds):
        bounds_ext = np.array(bounds)
        bounds_ext[:,0] -= self._sampler_cfg.cluster_radius
        bounds_ext[:,1] += self._sampler_cfg.cluster_radius
        print(bounds_ext)
        area_ext = np.prod(bounds_ext[:,1] - bounds_ext[:,0])
        num_points_parent = self._rng.poisson(area_ext * self._sampler_cfg.lambda_parent)
        coords = []
        for i in range(bounds_ext.shape[0]):
            coords.append(bounds_ext[i,0] + (bounds_ext[i,1] - bounds_ext[i,0]) * self._rng.uniform(0, 1, num_points_parent))
        print(coords)
        return np.stack(coords).T

    def getDaughters(self, parents_coords):
        num_points_daughter = self._rng.poisson(self._sampler_cfg.lambda_daughter, parents_coords.shape[0])
        num_points = sum(num_points_daughter)
        # simulating independent variables.
        theta = 2 * np.pi * self._rng.uniform(0, 1, num_points);  # angular coordinates
        rho = self._sampler_cfg.cluster_radius * np.sqrt(np.random.uniform(0, 1, num_points));  # radial coordinates
        if parents_coords.shape[1] == 3:
            # Convert from spherical to Cartesian coordinates
            phi = 2 * np.pi * self._rng.uniform(0, 1, num_points);  # angular coordinates
            x = np.sin(phi)*np.cos(theta)*rho
            y = np.sin(phi)*np.sin(theta)*rho
            z = np.cos(phi)*rho
            daughter_coords = np.stack([x,y,z]).T
        else:
            # Convert from polar to Cartesian coordinates
            x = rho*np.cos(theta)
            y = rho*np.sin(theta)
            daughter_coords = np.stack([x,y]).T
        parents_coords = np.repeat(parents_coords.T, num_points_daughter,axis=-1).T
        daughter_coords = daughter_coords + parents_coords
        correct = self._check_fn(daughter_coords)
        print(daughter_coords.shape)
        print(correct.shape)
        print(np.sum(correct))
        return daughter_coords[correct]

    def sample(self, bounds=[], **kwargs):
        parents_coords = self.getParents(bounds)
        points = self.getDaughters(parents_coords)
        return points
    
    def sample_equation_based_rejection(self, num):
        points = []
        for i in range(num):
            check_not_ok = True
            while check_not_ok:
                point = [self._rng.uniform(self._sampler_cfg.min[dim], self._sampler_cfg.max[dim]) for dim in range(self._sampler_cfg.randomization_space)]
                check_not_ok = not self._check_fn([point])[0]
            points.append(point)
        return np.array(points)

    def sample_image(self, num):
        idx = self._rng.choice(self.idx, p = self.p, size=num)
        local = self._rng.uniform(0, self.resolution, size=(num,self._sampler_cfg.randomization_space))
        if self._sampler_cfg.randomization_space == 2:
            x = idx // self.mask.shape[1]
            y = idx % self.mask.shape[1]
            return np.stack([x,y]).T + local
        if self._sampler_cfg.randomization_space == 3:
            x = idx // self.mask.shape[2] // self.mask.shape[1]
            y = idx // self.mask.shape[2] % self.mask[1]
            z = idx % self.mask.shape[2] % self.mask.shape[1]
            return np.stack([x,y,z]).T + local

class PoissonClusterPointSampler:
    # Samples points in a layer defined space using a Poisson cluser point process.
    pass

class LinearInterpolationSampler:
    pass

class SamplerFactory:
    def __init__(self):
        self.creators = {}
    
    def register(self, name: str, class_: BaseSampler) -> None:
        self.creators[name] = class_
        
    def get(self, cfg: Sampler_T, **kwargs:dict) -> BaseSampler:
        if cfg.__class__.__name__ not in self.creators.keys():
            raise ValueError("Unknown randomizer requested.")
        return self.creators[cfg.__class__.__name__](cfg)

Sampler_Factory = SamplerFactory()
Sampler_Factory.register("UniformSampler_T", UniformSampler)
Sampler_Factory.register("NormalSampler_T", NormalSampler)
Sampler_Factory.register("MaternClusterPointSampler_T", MaternClusterPointSampler)