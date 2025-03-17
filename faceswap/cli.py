from pathlib import Path

import cv2

from faceswap.inpaint.inpaint import main


def faceswap(source: Path, target: Path, output: Path):
    # Load the source and target images
    cv2.imread(str(source))
    cv2.imread(str(target))
    # result = align(source_img)
    face_swap_result = main()
    cv2.imwrite(str(output), face_swap_result)


if __name__ == "__main__":
    faceswap(
        "/home/arkadii/personal/faceswap/faceswap/data/raw/IMG_0653.jpg",
        "/home/arkadii/personal/faceswap/faceswap/data/raw/IMG_0653.jpg",
        "/home/arkadii/personal/faceswap/output.jpg",
    )
