import numpy as np
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

    def sample(self, query_point, **kwargs):
        """
        query point is (x, y) point generated from sampler
        """
        x = query_point[:, 0]
        y = query_point[:, 1]
        H, W = self.resolution
        us = x // self.mpp_resolution + (W // 2) * np.ones_like(x)
        vs = (H // 2) * np.ones_like(y) - y // self.mpp_resolution
        images = []
        for u, v in zip(us, vs):
            u = int(u)
            v = int(v)
            images.append(self.image[v, u])
        return np.stack(images)[:, np.newaxis]

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