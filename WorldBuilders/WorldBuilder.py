from abc import ABC

#class BaseWorld:
#    def __init__(self, config):
#    
#    def load():
#        return None
#
#    def reset():
#        return None


#class SolidGroundEnvironment:
#    def __init__(self, config):


# How would it work?
# An instance object would load the layer and randomizer + the configurations and generate assets.
# So the question is who's boss? Layer or Randomizer?
# I feel like randomizer should eat a layer.
# Then each randomizer must implement 2D and 3D generation methods.
# Ideally we don't want to have the randomizer to store the Layer as a parameter @FuckMyRAM, but instead access the values at runtime.
# So... We build a graph? Sounds complicated. But we need to tell the randomizer where to find what.
# A dictionary built recursively could work. Like when we build the world, we iterate through each object,
# And create the layers and randomizer at the world level. It makes sense. Since the world has access to the whole scene.
# Or does it? We have to be mindfull about references, we cannot mess with what's inside a reference can we? To check.
# Apparently is possible. So that's settled? The world object is getting a configuration file, and loading everything.
# Upon loading, it collects all randomizable actions, as well as interfaces, and sets them up.
# Then, when called, it can trigger the proper actions. Randomize or apply the commands through the requested interfaces.
# I feel like we are going to need an interface object.

# This is the chief! The master organizer? It looks a lot like an object though...
class World:
    def __init__(self, cfg):
        pass
    def load():
        pass
    def parse_configuration():
        pass
    def reset():
        pass
    def randomize():
        pass

# This stuff is an object it contains stuff.
class BaseObject:
    def __init__(self, cfg):
        self.cfg = cfg
    def parse_configuration():
        pass        
    def load():
        pass
    def reset():
        pass
    def randomize():
        pass
    def exposeInterfaces():
        pass

class SingleAsset(BaseObject):
    pass

class MultiAsset(SingleAsset):
    pass

class Instancer(BaseObject):
    pass

class BaseLight(BaseObject):
    pass

class GlobalLight(BaseLight):
    pass

class DomeLight(BaseLight):
    pass

class Skybox(BaseObject):
    pass


# This stuff randomizes shit given a set of constraints
class BaseRandomizer: # Must be able to randomize in 2D and 3D
    pass

class UniformRandomizer:
    # Uniformly samples points in a Layer defined space.
    pass

class NormalRandomizer:
    # Samples points in a Layer defined space using a Normal distribution.
    pass

class PoissonClusterPointProcess:
    # Samples points in a layer defined space using a Poisson cluser point process.
    pass

class MaternClusterPointRandomizer:
    # Samples points in a layer defined space using a Matern cluser point process. 
    pass


# This stuff provides constraints and informations to randomizers
class BaseLayer:
    pass

class VolumeLayer(BaseLayer): # Cube/Rectangle or Sphere/3DElipsoid
    # Defines a 3D space.
    pass

class PlaneLayer(BaseLayer): # Square/Rectangle, circle/elipse or Polygon.
    # Defines a 2D space.
    pass

class SemanticLayer(BaseLayer): # Square/Rectangle
    # Defines a 2D map with semantic information.
    pass

class NormalLayer(BaseLayer): # Square/Rectangle
    # Defines a 2D map with normal information.
    pass

class FloatLayer(BaseLayer): # Square/Rectangle
    # Defines a 2D map with Floating values information.
    pass


