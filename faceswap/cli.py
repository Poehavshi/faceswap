from pathlib import Path

import click
import cv2

from faceswap.preprocess.align import align


@click.command()
@click.option("--source", prompt="Path to the source face.", help="The person to swap to.")
@click.option("--target", prompt="Path to the target image.", help="The image to apply faceswap.")
@click.option("--output", prompt="Path to the output image.", help="The output image.", default="output.jpg")
def faceswap(source: Path, target: Path, output: Path):
    # Load the source and target images
    source_img = cv2.imread(str(source))
    cv2.imread(str(target))

    result = align(source_img)

    # Perform the face swap
    # (This is where the magic happens!)
    # result = face_swap(source_img, target_img)

    # Save the output image
    cv2.imwrite(str(output), result)
