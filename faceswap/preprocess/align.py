import glob
import logging

import cv2
import numpy as np
from scipy.spatial import ConvexHull

import faceswap.preprocess.models as models
from faceswap.preprocess.utils.crop import crop_image
from faceswap.preprocess.utils.utils import resize_to_limit
from faceswap.preprocess.config import cfg as config

logger = logging.getLogger(__name__)


class FaceAlignerPipeline:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.init(**kwargs)

    def init(self, **kwargs):
        self.init_models(**kwargs)

    def update_cfg(self, args_user):
        update_ret = False
        for key in args_user:
            if key in self.cfg.infer_params:
                if self.cfg.infer_params[key] != args_user[key]:
                    update_ret = True
                print(f"update infer cfg {key} from {self.cfg.infer_params[key]} to {args_user[key]}")
                self.cfg.infer_params[key] = args_user[key]
            elif key in self.cfg.crop_params:
                if self.cfg.crop_params[key] != args_user[key]:
                    update_ret = True
                print(f"update crop cfg {key} from {self.cfg.crop_params[key]} to {args_user[key]}")
                self.cfg.crop_params[key] = args_user[key]
            else:
                if key in self.cfg.infer_params and self.cfg.infer_params[key] != args_user[key]:
                    update_ret = True
                print(f"add {key}:{args_user[key]} to infer cfg")
                self.cfg.infer_params[key] = args_user[key]
        return update_ret

    def clean_models(self):
        for key in list(self.model_dict.keys()):
            del self.model_dict[key]
        self.model_dict = {}

    def init_models(self, **kwargs):
        self.model_dict = {}
        for model_name in self.cfg["models"]:
            print(f"loading model: {model_name}")
            print(self.cfg["models"][model_name])
            self.model_dict[model_name] = getattr(models, self.cfg["models"][model_name]["name"])(
                **self.cfg["models"][model_name]
            )

    def prepare_source(self, source_image):
        try:
            crop_infos = []
            img_bgr = resize_to_limit(
                source_image, self.cfg["infer_params"]["source_max_dim"], self.cfg["infer_params"]["source_division"]
            )
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            src_faces = self.model_dict["face_analysis"].predict(img_bgr)
            if len(src_faces) == 0:
                print("No face detected in the this image.")
                return crop_infos

            for i in range(len(src_faces)):
                lmk = src_faces[i]

                ret_dct = crop_image(
                    img_rgb,  # ndarray
                    lmk,  # 106x2 or Nx2
                    dsize=self.cfg["crop_params"]["src_dsize"],
                    scale=self.cfg["crop_params"]["src_scale"],
                    vx_ratio=self.cfg["crop_params"]["src_vx_ratio"],
                    vy_ratio=self.cfg["crop_params"]["src_vy_ratio"],
                )

                lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                ret_dct["lmk_crop"] = lmk
                crop_infos.append(ret_dct)
                return crop_infos[0]["img_crop"]
        except Exception:
            logger.exception("Error in prepare_source")
            return False

    def create_masks(self, target_image):
        try:
            crop_infos = []
            img_bgr = resize_to_limit(
                target_image, self.cfg["infer_params"]["source_max_dim"], self.cfg["infer_params"]["source_division"]
            )
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            src_faces = self.model_dict["face_analysis"].predict(img_bgr)
            if len(src_faces) == 0:
                print("No face detected in the this image.")
                return crop_infos

            for i in range(len(src_faces)):
                lmk = src_faces[i]

                ret_dct = crop_image(
                    img_rgb,  # ndarray
                    lmk,  # 106x2 or Nx2
                    dsize=self.cfg["crop_params"]["src_dsize"],
                    scale=self.cfg["crop_params"]["src_scale"],
                    vx_ratio=self.cfg["crop_params"]["src_vx_ratio"],
                    vy_ratio=self.cfg["crop_params"]["src_vy_ratio"],
                )

                lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                ret_dct["lmk_crop"] = lmk

                mask = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

                binary_mask = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255

                points = np.array(lmk, dtype=np.int32)
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                cv2.fillConvexPoly(binary_mask, hull_points, 0)  # 0 = area to be transparent

                binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)

                mask[:, :, 3] = binary_mask

                ret_dct["mask"] = mask
                crop_infos.append(ret_dct)

            return mask
        except Exception:
            logger.exception("Error in prepare_source")
            return False

    def __del__(self):
        self.clean_models()


def align(source_image):
    aligner = FaceAlignerPipeline(cfg=config)
    result = aligner.prepare_source(source_image)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def create_mask(target_image):
    aligner = FaceAlignerPipeline(cfg=config)
    result = aligner.create_masks(target_image)
    return result


if __name__ == "__main__":
    # for file in glob.glob("/home/arkadii/personal/faceswap/faceswap/data/raw/*.jpg"):
    #     stem = file.split("/")[-1].split(".")[0]
    #     image = cv2.imread(file)
    #     aligned_image = align(image)
    #     cv2.imwrite(f"/home/arkadii/personal/faceswap/faceswap/data/aligned/{stem}.png", aligned_image)
    #
    destination = cv2.imread("/home/arkadii/personal/faceswap/faceswap/data/destination.jpg")
    masked = create_mask(destination)
    cv2.imwrite(f"/home/arkadii/personal/faceswap/faceswap/data/destination_masked.png", masked)
