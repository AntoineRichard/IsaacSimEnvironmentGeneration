import numpy as np
import quaternion
from Types import *

# I was in troublem deciding whether I should wrap BaseSampler or create new class and wrap it. 
# Since Z-value clip function is deterministic, meaning z value is always determined based on heightmap. 
# Therefore, there is no point in creating random distribution as done in BaseSampler and other wrapped class. 
# So, I made BaseClipper method and new class called HeightClipper wraps BaseClipper. 

class BaseClipper:
    def __init__(self, sampler_cfg: Sampler_T):
        self._sampler_cfg = sampler_cfg
        self.image = self._sampler_cfg.data
        self.resolution = self._sampler_cfg.resolution
        self.mpp_resolution = self._sampler_cfg.mpp_resolution

        assert len(self.image.shape) == 2, f"image need to be 1 channel image, not {image.shape}"
    
    def __call__(self, **kwargs):
        return self.sample(**kwargs)

    def sample(self, **kwargs):
        raise NotImplementedError()

class HeightClipper(BaseClipper):
    def __init__(self, sampler_cfg: Sampler_T):
        super().__init__(sampler_cfg)

    def sample(self, query_point:np.ndarray, **kwargs):
        """
        query point is (x, y) point generated from 2D sampler. 
        """
        x = query_point[:, 0]
        y = query_point[:, 1]
        H, W = self.resolution
        # cordinate transformation from xy to image plane
        us = x // self.mpp_resolution #horizontal
        vs = H * np.ones_like(y) - y // self.mpp_resolution #vertical
        ##
        images = []
        for u, v in zip(us, vs):
            u = int(u)
            v = int(v)
            images.append(self.image[v-1, u-1])
        return np.stack(images)[:, np.newaxis]

class NormalMapClipper(BaseClipper):
    def __init__(self, sampler_cfg: Sampler_T):
        super().__init__(sampler_cfg)

    def compute_slopes(self)->None:
        nx,ny = np.gradient(self.image)
        slope_x = np.arctan2(nx,1) #theta_x = tan^-1(nx)
        slope_y = np.arctan2(ny,1) #theta_y = tan^-1(ny)
        # magnitude = np.hypot(nx,ny)
        # slope_xy = np.arctan2(magnitude,1)
        self.slope_x = slope_x
        self.slope_y = slope_y
        # self.slope_xy = slope_xy
        # self.magnitude = magnitude

    def sample(self, query_point:np.ndarray, **kwargs):
        """
        query point is (x, y) point generated from sampler
        """
        self.compute_slopes()
        x = query_point[:, 0]
        y = query_point[:, 1]
        H, W = self.resolution
        us = x // self.mpp_resolution #horizontal
        vs = H * np.ones_like(y) - y // self.mpp_resolution #vertical
        quat = []
        for u, v in zip(us, vs):
            u = int(u)
            v = int(v)
            roll = self.slope_y[v-1, u-1]
            pitch = self.slope_x[v-1, u-1]
            yaw = 0.0
            q = quaternion.as_float_array(quaternion.from_euler_angles([roll, pitch, yaw]))
            quat.append(q)

        return np.stack(quat)

class ClipperFactory:
    def __init__(self):
        self.creators = {}
    
    def register(self, name: str, class_: BaseClipper) -> None:
        self.creators[name] = class_
        
    def get(self, cfg: Sampler_T, **kwargs:dict) -> BaseClipper:
        if cfg.__class__.__name__ not in self.creators.keys():
            raise ValueError("Unknown sampler requested.")
        return self.creators[cfg.__class__.__name__](cfg)

Clipper_Factory = ClipperFactory()
Clipper_Factory.register("ImageClipper_T", HeightClipper)
Clipper_Factory.register("NormalMapClipper_T", NormalMapClipper)