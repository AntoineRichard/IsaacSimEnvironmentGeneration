# Essentially 2 types of wrappers
# Instancer
# PrimView and RigidPrimView
import numpy as np

from Mixer import RequestMixer
from Types import UserRequest_T

import omni
from pxr_utils import createInstancerAndCache, setInstancerParameters

class RequestInstancer: #TODO add Collisions and rigid bodies.
    def __init__(self, instancer_path: str, requests: list, asset_list: list = None, enable_collisions: bool = False, make_rigid: bool = False) -> None:
        self.instancer_path = instancer_path
        self.asset_list = asset_list

        self.enable_collisions = enable_collisions
        self.make_rigid = make_rigid
        self.mixers = []

        self.buildInstancer()

        # We expect a list of list of requests
        new_requests = []
        if requests is not list:
            new_requests = [[requests]]
        else:
            for req_list in requests:
                if req_list is not list:
                    assert isinstance(req_list, UserRequest_T), "requests must be of type list(list(UserRequest_T))."
                    new_requests.append([req_list])
                else:
                    for req in req_list:
                        assert isinstance(req, UserRequest_T), "requests must be of type list(list(UserRequest_T))."
                    new_requests.append(req)

        for req_list in new_requests:
            self.mixers.append(RequestMixer(req_list))
        self.requests = new_requests

    def buildInstancer(self):
        # Check if the instancer already exists. If it does act as if it was complete.
        # Else create the instancer and its associated cache.
        stage = omni.usd.get_context().get_stage()
        if self.asset_list is None:
            raise ValueError("No assets provided. Some asset must be passed as argument.")
        createInstancerAndCache(stage, self.instancer_path, self.asset_list)

    def sample(self, num):
        attribute_list = {"xformOp:translation":[],"xformOp:orientation":[],"xformOp:scale":[]}
        for mixer in self.mixers:
            attributes = mixer.execute_graph(num)
            for attr_key in attributes.keys():
                attribute_list[attr_key].append(attributes[attr_key])

        attributes = {}
        for attr_key in attribute_list.keys():
            attributes[attr_key] = np.concatenate(attribute_list[attr_key],axis=-1)
        return attributes

    def __call__(self, num):
        attributes = self.sample(num)
        stage = omni.get_context().get_stage()
        setInstancerParameters(stage, self.instancer_path, attributes["xformOp:translate"], scale=attributes['xformOp:scale'],quat=attributes['xformOp:orient'])
            


# This one will be given a list of prims.
class RequestPrimView:
    def __init__(self, request: list, enable_collisions: bool = False, make_rigid: bool = False) -> None:
        pass

    def __call__(self):
        attributes = self.sample(self.num)
        stage = omni.get_context().get_stage()
        pass

