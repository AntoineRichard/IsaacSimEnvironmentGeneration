from Types import *
from Layers import *
from Samplers import *

import copy

class MetaLayer:
    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T) -> None:
       self.layer = Layer_Factory.get(layer_cfg, sampler_cfg)

    def __call__(self, num=1, **kwargs) -> np.ndarray([]):
        return self.layer(num, **kwargs)

class RequestMixer:
    def __init__(self, requests: tuple()) -> None:
        self.requests = requests
        self.parseRequests()
        self.buildExecutionGraph()
        self.num = 1000

    def __call__(self, num) -> None:
        self.executeGraph(num)


    def parseRequests(self):
        # Take all the requests and sort them by parameters
        requests_per_type = {}
        for req in self.requests:
            if req.p_type.name in requests_per_type.keys():
                requests_per_type[req.p_type.name].append(req)
            else:
                requests_per_type[req.p_type.name] = [req]
        # For each requested parameter type, check for axes errors
        for reqs_key in requests_per_type.keys():
            axes = []
            for req in requests_per_type[reqs_key]:
                for axis in "".join(req.axes):
                    axes.append(axis)
                    print(axis)
                # Check that the dimension of the layer matches the one of the axes.
                assert len(req.axes) == req.layer.output_space, "An error occured while parsing "+reqs_key+". Layer dimension do not match the number axes."
            assert len(np.unique(axes)) == len(axes), "An error occured while parsing parameter "+reqs_key+". Duplicate axes found."
        self.requests_per_type = requests_per_type

    def buildExecutionGraph(self):
        self.execution_graph = {}
        for req_type in self.requests_per_type.keys():
            attribute_name = self.requests_per_type[req_type][0].p_type.attribute_name
            to_exec = {}
            to_exec["meta_layer"] = []
            to_exec["replicate"] = []
            to_exec["order"] = []
            to_exec["axes"] = []
            specified_axes = []
            for req in self.requests_per_type[req_type]:
                to_exec["meta_layer"].append(MetaLayer(req.layer, req.sampler))
                to_exec["replicate"].append(np.repeat(list(range(len(req.axes))), [len(i) for i in req.axes]))
                to_exec["order"].append([req.p_type.index_mapping[axis] for axis in "".join(req.axes)])
                to_exec["axes"].append(list(range(len(to_exec["replicate"][-1]))))
                specified_axes.append(req.axes)
            specified_axes = [item for sublist in specified_axes for item in sublist]
            for axis in req.p_type.components:
                if axis not in "".join(specified_axes):
                    idx = req.p_type.index_mapping[axis]
                    value = req.p_type.default_value[idx]
                    to_exec["meta_layer"].append(lambda x : np.ones((x,1))*value)
                    to_exec["replicate"].append([0])
                    to_exec["order"].append([idx])
                    to_exec["axes"].append([0])
            self.execution_graph[attribute_name] = to_exec

    def executeGraph(self, num):
        print(self.execution_graph)
        attributes = {}
        for attribute in self.execution_graph:
            current_order = []
            print(attribute)
            to_exec = self.execution_graph[attribute]
            print(to_exec)
            p_list = []
            for j in range(len(to_exec["meta_layer"])):
                points = to_exec["meta_layer"][j](num)
                print(to_exec["axes"][j], to_exec["replicate"][j])
                print(np.repeat(to_exec["axes"][j], to_exec["replicate"][j]))
                print(points.shape)
                points = np.stack([points[:,i] for i in to_exec["replicate"][j]]).T
                print(points.shape)
                current_order += to_exec["order"][j]
                p_list.append(points)
            points = np.concatenate(p_list,axis=-1)
            remapped = [current_order.index(i) for i in range(len(current_order))]
            points = np.stack([points[:,i] for i in remapped]).T
            attributes[attribute] = points
            print(points)
        return attributes