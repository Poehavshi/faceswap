import os

from faceswap.config import settings

cfg = {
    "models": {
        "face_analysis": {
            "name": "FaceAnalysisModel",
            "predict_type": "ort",
            "model_path": [
                os.path.join(settings.models_dir, "retinaface_det_static.onnx"),
                os.path.join(settings.models_dir, "face_2dpose_106_static.onnx"),
            ],
        },
        # "face_analysis": {
        #     "name": "MediaPipeFaceModel",
        #     "predict_type": "mp"
        # },
        "landmark": {
            "name": "LandmarkModel",
            "predict_type": "ort",
            "model_path": os.path.join(settings.models_dir, "landmark.onnx"),
        },
    },
    "infer_params": {
        "source_max_dim": 1280,
        "source_division": 2,
    },
    "crop_params": {
        "src_dsize": 512,
        "src_scale": 2.3,
        "src_vx_ratio": 0.0,
        "src_vy_ratio": -0.125,
    },
}
