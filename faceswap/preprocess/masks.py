import cv2
import numpy as np
from scipy.spatial import ConvexHull

from preprocess.config import cfg as config
from preprocess.align import align, FaceAlignerPipeline

aligner = FaceAlignerPipeline(cfg=config)


def create_mask(target_image):
    """
    Creates a binary mask for a face in the target image.

    Args:
        target_image: Input BGR image containing a face

    Returns:
        A binary mask image where the face area is white (255) and the rest is black (0)
    """
    aligner = FaceAlignerPipeline(cfg=config)
    crop_info = aligner.prepare_source(target_image)

    if crop_info is False or crop_info is None:
        print("No face detected in the image.")
        return None

    img_crop = crop_info
    mask = np.zeros(img_crop.shape[:2], dtype=np.uint8)

    crop_infos = []
    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

    lmk = aligner.model_dict["landmark"].predict(img_rgb, None)

    if lmk is None or len(lmk) == 0:
        print("No landmarks detected in the face.")
        return None

    points = np.array(lmk, dtype=np.int32)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    cv2.fillConvexPoly(mask, hull_points, 255)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    del aligner
    return mask


def apply_mask(image, mask):
    """
    Applies a mask to an image.

    Args:
        image: Input image (BGR)
        mask: Binary mask where white (255) represents the area to keep

    Returns:
        Image with mask applied (background is black)
    """
    # Ensure mask has the right number of channels
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Normalize mask to values between 0 and 1
    mask = mask.astype(float) / 255.0

    # Apply mask
    result = (image.astype(float) * mask).astype(np.uint8)

    return result


if __name__ == "__main__":
    import glob

    for file in glob.glob("/home/arkadii/personal/faceswap/faceswap/data/raw/*.jpg"):
        stem = file.split("/")[-1].split(".")[0]
        image = cv2.imread(file)

        face_mask = create_mask(image)

        if face_mask is not None:
            cv2.imwrite(f"/home/arkadii/personal/faceswap/faceswap/data/masks/{stem}_mask.png", face_mask)
            aligned_image = align(image)
            masked_face = apply_mask(aligned_image, face_mask)
            cv2.imwrite(f"/home/arkadii/personal/faceswap/faceswap/data/masked/{stem}_masked.png", masked_face)
