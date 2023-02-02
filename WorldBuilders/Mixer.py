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
        self.has_point_process = False
        self.point_process_attr = None
        self.parseRequests()
        self.buildExecutionGraph()

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
        point_processes = 0
        for reqs_key in requests_per_type.keys():
            axes = []
            for i, req in enumerate(requests_per_type[reqs_key]):
                if isinstance(req.sampler, PointProcess_T):
                    point_processes += 1
                    assert point_processes <= 1, "An error occured while parsing the requests. There can only be one point process."
                    self.has_point_process = True
                    self.point_process_attr = req.p_type.attribute_name
                    point_process_idx = i
                for axis in "".join(req.axes):
                    axes.append(axis)
                # Check that the dimension of the layer matches the one of the axes.
                assert len(req.axes) == req.layer.output_space, "An error occured while parsing "+reqs_key+". Layer dimension do not match the number axes."
            if point_processes > 0:
                tmp_list = [requests_per_type[reqs_key][point_process_idx]]
                for i,req in enumerate(requests_per_type):
                    if i != point_process_idx:
                        tmp_list.append(req)
            assert len(np.unique(axes)) == len(axes), "An error occured while parsing "+reqs_key+". Duplicate axes found."
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
            for j, req in enumerate(self.requests_per_type[req_type]):
                to_exec["meta_layer"].append(MetaLayer(req.layer, req.sampler))
                to_exec["replicate"].append(np.repeat(list(range(len(req.axes))), [len(i) for i in req.axes]))
                to_exec["order"].append([req.p_type.index_mapping[axis] for axis in "".join(req.axes)])
                to_exec["axes"].append(list(range(len(to_exec["replicate"][-1]))))
                specified_axes.append(req.axes)
            specified_axes = [item for sublist in specified_axes for item in sublist]

            # If an axis is not provided by the user, fill this axies of the generated point using the default value for this attribute.
            for axis in req.p_type.components:
                if axis not in "".join(specified_axes): # If an axis is missing
                    idx = req.p_type.index_mapping[axis] # Get mapping (the index of that value)
                    value = req.p_type.default_value[idx] # Get the default value
                    # Generate a lambda function that will behave like a meta layer.
                    to_exec["meta_layer"].append(lambda x, value=value : np.ones((x,1))*value)
                    # Add the proper hyper parameters to enable merging.
                    to_exec["replicate"].append([0])
                    to_exec["order"].append([idx])
                    to_exec["axes"].append([0])
            self.execution_graph[attribute_name] = to_exec

    def executeGraph(self, num):
        output = {}

        attributes = self.execution_graph
        if self.point_process_attr is not None:
            tmp = [self.point_process_attr]
            for attr in attributes:
                if attr != self.point_process_attr:
                    tmp.append(attr)
            attributes = tmp

        is_first = True
        for attribute in attributes:
            current_order = []
            to_exec = self.execution_graph[attribute]
            p_list = []
            for j in range(len(to_exec["meta_layer"])):
                points = to_exec["meta_layer"][j](num)
                points = np.stack([points[:,i] for i in to_exec["replicate"][j]]).T
                current_order += to_exec["order"][j]
                p_list.append(points)
                if self.has_point_process and is_first:
                    num = points.shape[0]
                    is_first = False
            points = np.concatenate(p_list,axis=-1)
            remapped = [current_order.index(i) for i in range(len(current_order))]
            points = np.stack([points[:,i] for i in remapped]).T
            output[attribute] = points
        return output