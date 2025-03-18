import os
import random
import sys
from typing import Sequence, Mapping, Any, Union

import cv2
import torch

from folder_paths import get_input_directory
from folder_paths import get_output_directory


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")



add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS

def main(source_image, target_image):
    # masked_target_image_path = "clipspace/clipspace-mask-26662.png [input]"
    masked_target_image_name = "tmp_target.jpg"
    masked_target_image_path = os.path.join(get_input_directory(), masked_target_image_name)
    cv2.imwrite(masked_target_image_path, target_image)

    source_image_name = "tmp.jpg"
    source_image_path = os.path.join(get_input_directory(), source_image_name)
    cv2.imwrite(source_image_path, source_image)
    import_custom_nodes()
    with torch.inference_mode():
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_341 = dualcliploader.load_clip(clip_name1="clip_l.safetensors", clip_name2="t5xxl_fp16.safetensors", type="flux", device="default")

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_343 = cliptextencode.encode(text="Retain and beautify face", clip=get_value_at_index(dualcliploader_341, 0))

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_345 = fluxguidance.append(guidance=50, conditioning=get_value_at_index(cliptextencode_343, 0))

        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        conditioningzeroout_404 = conditioningzeroout.zero_out(conditioning=get_value_at_index(cliptextencode_343, 0))

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_338 = vaeloader.load_vae(vae_name="ae.safetensors")

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_239 = loadimage.load_image(image=masked_target_image_path)

        inpaintcrop = NODE_CLASS_MAPPINGS["InpaintCrop"]()
        inpaintcrop_411 = inpaintcrop.inpaint_crop(context_expand_pixels=0, context_expand_factor=1, fill_mask_holes=True, blur_mask_pixels=16, invert_mask=False, blend_pixels=16, rescale_algorithm="bicubic", mode="forced size", force_width=1024, force_height=1024, rescale_factor=1, min_width=1024, min_height=1024, max_width=768, max_height=768, padding=32, image=get_value_at_index(loadimage_239, 0), mask=get_value_at_index(loadimage_239, 1))

        imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
        imageresize_399 = imageresize.execute(width=1024, height=1024, interpolation="lanczos", method="keep proportion", condition="downscale if bigger", multiple_of=0, image=get_value_at_index(inpaintcrop_411, 1))

        loadimage_240 = loadimage.load_image(image=source_image_path)

        imageresize_175 = imageresize.execute(width=0, height=get_value_at_index(imageresize_399, 2), interpolation="lanczos", method="keep proportion", condition="always", multiple_of=0, image=get_value_at_index(loadimage_240, 0))

        imageconcanate = NODE_CLASS_MAPPINGS["ImageConcanate"]()
        imageconcanate_323 = imageconcanate.concatenate(direction="right", match_image_size=True, image1=get_value_at_index(imageresize_399, 0), image2=get_value_at_index(imageresize_175, 0))

        resizemask = NODE_CLASS_MAPPINGS["ResizeMask"]()
        resizemask_402 = resizemask.resize(width=get_value_at_index(imageresize_399, 1), height=get_value_at_index(imageresize_399, 2), keep_proportions=True, upscale_method="nearest-exact", crop="disabled", mask=get_value_at_index(inpaintcrop_411, 2))

        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        masktoimage_182 = masktoimage.mask_to_image(mask=get_value_at_index(resizemask_402, 0))

        emptyimage = NODE_CLASS_MAPPINGS["EmptyImage"]()
        emptyimage_184 = emptyimage.generate(width=get_value_at_index(imageresize_175, 1), height=get_value_at_index(imageresize_175, 2), batch_size=1, color=0)

        imageconcanate_181 = imageconcanate.concatenate(direction="right", match_image_size=True, image1=get_value_at_index(masktoimage_182, 0), image2=get_value_at_index(emptyimage_184, 0))

        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        imagetomask_185 = imagetomask.image_to_mask(channel="red", image=get_value_at_index(imageconcanate_181, 0))

        impactgaussianblurmask = NODE_CLASS_MAPPINGS["ImpactGaussianBlurMask"]()
        impactgaussianblurmask_403 = impactgaussianblurmask.doit(kernel_size=30, sigma=10, mask=get_value_at_index(imagetomask_185, 0))

        inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
        inpaintmodelconditioning_221 = inpaintmodelconditioning.encode(noise_mask=True, positive=get_value_at_index(fluxguidance_345, 0), negative=get_value_at_index(conditioningzeroout_404, 0), vae=get_value_at_index(vaeloader_338, 0), pixels=get_value_at_index(imageconcanate_323, 0), mask=get_value_at_index(impactgaussianblurmask_403, 0))

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_340 = unetloader.load_unet(unet_name="flux1FillDevFp8_v10.safetensors", weight_dtype="default")

        power_lora_loader_rgthree = NODE_CLASS_MAPPINGS["Power Lora Loader (rgthree)"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        imagecrop = NODE_CLASS_MAPPINGS["ImageCrop"]()
        imageandmaskpreview = NODE_CLASS_MAPPINGS["ImageAndMaskPreview"]()
        inpaintstitch = NODE_CLASS_MAPPINGS["InpaintStitch"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            power_lora_loader_rgthree_337 = power_lora_loader_rgthree.load_loras(PowerLoraLoaderHeaderWidget={'type': 'PowerLoraLoaderHeaderWidget'}, lora_1={'on': True, 'lora': 'comfyui_portrait_lora64.safetensors', 'strength': 1}, lora_2={'on': True, 'lora': 'FLUX.1-Turbo-Alpha.safetensors', 'strength': 1}, model=get_value_at_index(unetloader_340, 0), clip=get_value_at_index(dualcliploader_341, 0))

            ksampler_346 = ksampler.sample(seed=random.randint(1, 2**64), steps=12, cfg=1, sampler_name="euler", scheduler="normal", denoise=1, model=get_value_at_index(power_lora_loader_rgthree_337, 0), positive=get_value_at_index(inpaintmodelconditioning_221, 0), negative=get_value_at_index(inpaintmodelconditioning_221, 1), latent_image=get_value_at_index(inpaintmodelconditioning_221, 2))

            vaedecode_214 = vaedecode.decode(samples=get_value_at_index(ksampler_346, 0), vae=get_value_at_index(vaeloader_338, 0))

            imagecrop_228 = imagecrop.crop(width=get_value_at_index(imageresize_399, 1), height=get_value_at_index(imageresize_399, 2), x=0, y=0, image=get_value_at_index(vaedecode_214, 0))

            imageandmaskpreview_385 = imageandmaskpreview.execute(mask_opacity=0.5, mask_color="255, 0, 255", pass_through=False, image=get_value_at_index(imageconcanate_323, 0), mask=get_value_at_index(impactgaussianblurmask_403, 0))

            inpaintstitch_412 = inpaintstitch.inpaint_stitch(rescale_algorithm="bislerp", stitch=get_value_at_index(inpaintcrop_411, 0), inpainted_image=get_value_at_index(imagecrop_228, 0))

            saveimage_413 = saveimage.save_images(filename_prefix="AceFaceSwap/Faceswap", images=get_value_at_index(inpaintstitch_412, 0))['ui']['images'][0]
    image_name = saveimage_413['filename']
    folder = saveimage_413['subfolder']
    result_path = os.path.join(get_output_directory(), folder, image_name)
    return cv2.imread(result_path)
