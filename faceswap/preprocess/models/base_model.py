import copy

import torch

from .predictor import get_predictor


class BaseModel:
    def __init__(self, **kwargs):
        self.kwargs = copy.deepcopy(kwargs)
        self.predictor = get_predictor(**self.kwargs)
        self.device = torch.cuda.current_device()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.predict_type = kwargs.get("predict_type", "trt")

        if self.predictor is not None:
            self.input_shapes = self.predictor.input_spec()
            self.output_shapes = self.predictor.output_spec()

    def input_process(self, *data):
        pass

    def output_process(self, *data):
        pass

    def predict(self, *data):
        pass

    def __del__(self):
        if self.predictor is not None:
            del self.predictor
