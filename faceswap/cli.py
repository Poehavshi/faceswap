from pathlib import Path

import click
import cv2

from faceswap.inpaint.inpaint import main
from faceswap.preprocess.align import align
from faceswap.preprocess.align import create_mask


@click.command()
@click.option("--source", prompt="Path to the source face.", help="The person to swap to.")
@click.option("--target", prompt="Path to the target image.", help="The image to apply faceswap.")
@click.option("--output", prompt="Path to the output image.", help="The output image.", default="output.jpg")
def faceswap(source: Path, target: Path, output: Path):
    # Load the source and target images
    source_image = cv2.imread(str(source))
    target_image = cv2.imread(str(target))
    masked_target = create_mask(target_image)
    aligned_source_face = align(source_image)
    face_swap_result = main(aligned_source_face, masked_target)
    cv2.imwrite(str(output), face_swap_result)
