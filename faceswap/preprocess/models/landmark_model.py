import cv2
import numpy as np

from faceswap.preprocess.utils.crop import _transform_pts, crop_image

from .base_model import BaseModel


class LandmarkModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dsize = 224

    def input_process(self, *data):
        img_rgb, lmk = data
        crop_dct = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
        img_crop_rgb = crop_dct["img_crop"]
        inp = (img_crop_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)
        return inp, crop_dct

    def output_process(self, *data):
        out_pts, crop_dct = data
        lmk = out_pts[2].reshape(-1, 2) * self.dsize  # scale to 0-224
        lmk = _transform_pts(lmk, M=crop_dct["M_c2o"])
        return lmk

    def predict(self, *data):
        input_image, crop_dct = self.input_process(*data)
        preds = self.predictor.predict(input_image)
        outputs = self.output_process(preds, crop_dct)
        return outputs
