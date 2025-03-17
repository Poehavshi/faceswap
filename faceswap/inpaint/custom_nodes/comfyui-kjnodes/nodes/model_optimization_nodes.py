from comfy.ldm.modules import attention as comfy_attention
import logging
import comfy.model_patcher
import comfy.utils
import comfy.sd
import torch
import folder_paths
import comfy.model_management as mm
from comfy.cli_args import args

orig_attention = comfy_attention.optimized_attention
original_patch_model = comfy.model_patcher.ModelPatcher.patch_model
original_load_lora_for_models = comfy.sd.load_lora_for_models

class BaseLoaderKJ:
    original_linear = None
    cublas_patched = False

    def _patch_modules(self, patch_cublaslinear, sage_attention):
        from comfy.ops import disable_weight_init, CastWeightBiasOp, cast_bias_weight

        if sage_attention != "disabled":
            print("Patching comfy attention to use sageattn")
            from sageattention import sageattn
            def set_sage_func(sage_attention):
                if sage_attention == "auto":
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
                    from sageattention import sageattn_qk_int8_pv_fp16_cuda
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
                    from sageattention import sageattn_qk_int8_pv_fp16_triton
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                    return func
                elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
                    from sageattention import sageattn_qk_int8_pv_fp8_cuda
                    def func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
                    return func

            sage_func = set_sage_func(sage_attention)

            @torch.compiler.disable()
            def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
                if skip_reshape:
                    b, _, _, dim_head = q.shape
                    tensor_layout="HND"
                else:
                    b, _, dim_head = q.shape
                    dim_head //= heads
                    q, k, v = map(
                        lambda t: t.view(b, -1, heads, dim_head),
                        (q, k, v),
                    )
                    tensor_layout="NHD"
                if mask is not None:
                    # add a batch dimension if there isn't already one
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    # add a heads dimension if there isn't already one
                    if mask.ndim == 3:
                        mask = mask.unsqueeze(1)
                out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
                if tensor_layout == "HND":
                    if not skip_output_reshape:
                        out = (
                            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
                        )
                else:
                    if skip_output_reshape:
                        out = out.transpose(1, 2)
                    else:
                        out = out.reshape(b, -1, heads * dim_head)
                return out

            comfy_attention.optimized_attention = attention_sage
            comfy.ldm.hunyuan_video.model.optimized_attention = attention_sage
            comfy.ldm.flux.math.optimized_attention = attention_sage
            comfy.ldm.genmo.joint_model.asymm_models_joint.optimized_attention = attention_sage
            comfy.ldm.cosmos.blocks.optimized_attention = attention_sage
            comfy.ldm.wan.model.optimized_attention = attention_sage

        else:
            comfy_attention.optimized_attention = orig_attention
            comfy.ldm.hunyuan_video.model.optimized_attention = orig_attention
            comfy.ldm.flux.math.optimized_attention = orig_attention
            comfy.ldm.genmo.joint_model.asymm_models_joint.optimized_attention = orig_attention
            comfy.ldm.cosmos.blocks.optimized_attention = orig_attention
            comfy.ldm.wan.model.optimized_attention = orig_attention

        if patch_cublaslinear:
            if not BaseLoaderKJ.cublas_patched:
                BaseLoaderKJ.original_linear = disable_weight_init.Linear
                try:
                    from cublas_ops import CublasLinear
                except ImportError:
                    raise Exception("Can't import 'torch-cublas-hgemm', install it from here https://github.com/aredden/torch-cublas-hgemm")

                class PatchedLinear(CublasLinear, CastWeightBiasOp):
                    def reset_parameters(self):
                        pass

                    def forward_comfy_cast_weights(self, input):
                        weight, bias = cast_bias_weight(self, input)
                        return torch.nn.functional.linear(input, weight, bias)

                    def forward(self, *args, **kwargs):
                        if self.comfy_cast_weights:
                            return self.forward_comfy_cast_weights(*args, **kwargs)
                        else:
                            return super().forward(*args, **kwargs)

                disable_weight_init.Linear = PatchedLinear
                BaseLoaderKJ.cublas_patched = True
        else:
            if BaseLoaderKJ.cublas_patched:
                disable_weight_init.Linear = BaseLoaderKJ.original_linear
                BaseLoaderKJ.cublas_patched = False

class PathchSageAttentionKJ(BaseLoaderKJ):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "sage_attention": (["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"], {"default": False, "tooltip": "Global patch comfy attention to use sageattn, once patched to revert back to normal you would need to run this node again with disabled option."}),
        }}

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"
    DESCRIPTION = "Experimental node for patching attention mode. This doesn't use the model patching system and thus can't be disabled without running the node again with 'disabled' option."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch(self, model, sage_attention):
        self._patch_modules(False, sage_attention)
        return model,

class CheckpointLoaderKJ(BaseLoaderKJ):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            "patch_cublaslinear": ("BOOLEAN", {"default": False, "tooltip": "Enable or disable the patching, won't take effect on already loaded models!"}),
            "sage_attention": (["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"], {"default": False, "tooltip": "Patch comfy attention to use sageattn."}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "patch"
    OUTPUT_NODE = True
    DESCRIPTION = "Experimental node for patching torch.nn.Linear with CublasLinear."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch(self, ckpt_name, patch_cublaslinear, sage_attention):
        self._patch_modules(patch_cublaslinear, sage_attention)
        from nodes import CheckpointLoaderSimple
        model, clip, vae = CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)
        return model, clip, vae

class DiffusionModelLoaderKJ(BaseLoaderKJ):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The name of the checkpoint (model) to load."}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"],),
            "compute_dtype": (["default", "fp16", "bf16", "fp32"], {"default": "fp16", "tooltip": "The compute dtype to use for the model."}),
            "patch_cublaslinear": ("BOOLEAN", {"default": False, "tooltip": "Enable or disable the patching, won't take effect on already loaded models!"}),
            "sage_attention": (["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"], {"default": False, "tooltip": "Patch comfy attention to use sageattn."}),
            "enable_fp16_accumulation": ("BOOLEAN", {"default": False, "tooltip": "Enable torch.backends.cuda.matmul.allow_fp16_accumulation, requires pytorch 2.7.0 nightly."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_and_load"
    OUTPUT_NODE = True
    DESCRIPTION = "Node for patching torch.nn.Linear with CublasLinear."
    EXPERIMENTAL = True
    CATEGORY = "KJNodes/experimental"

    def patch_and_load(self, model_name, weight_dtype, compute_dtype, patch_cublaslinear, sage_attention, enable_fp16_accumulation):
        DTYPE_MAP = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        model_options = {}
        if dtype := DTYPE_MAP.get(weight_dtype):
            model_options["dtype"] = dtype
            print(f"Setting {model_name} weight dtype to {dtype}")

        if weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True

        if enable_fp16_accumulation:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                raise RuntimeError("Failed to set fp16 accumulation, this requires pytorch 2.7.0 nightly currently")
        else:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = False

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        if dtype := DTYPE_MAP.get(compute_dtype):
            model.set_model_compute_dtype(dtype)
            model.force_cast_weights = False
            print(f"Setting {model_name} compute dtype to {dtype}")
        self._patch_modules(patch_cublaslinear, sage_attention)

        return (model,)

def patched_patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):

    if lowvram_model_memory == 0:
        full_load = True
    else:
        full_load = False

    device_to = mm.get_torch_device()
    load_weights = True
    if load_weights:
        self.load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
    for k in self.object_patches:
        old = comfy.utils.set_attr(self.model, k, self.object_patches[k])
        if k not in self.object_patches_backup:
            self.object_patches_backup[k] = old

    return self.model

def patched_load_lora_for_models(model, clip, lora, strength_model, strength_clip):

    patch_keys = list(model.object_patches_backup.keys())
    for k in patch_keys:
        #print("backing up object patch: ", k)
        comfy.utils.set_attr(model.model, k, model.object_patches_backup[k])

    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    loaded = comfy.lora.load_lora(lora, key_map)
    #print(temp_object_patches_backup)

    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_patches(loaded, strength_model)
    else:
        k = ()
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()
        k1 = new_clip.add_patches(loaded, strength_clip)
    else:
        k1 = ()
        new_clip = None
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("NOT LOADED {}".format(x))

    if patch_keys:
        if hasattr(model.model, "compile_settings"):
            compile_settings = getattr(model.model, "compile_settings")
            print("compile_settings: ", compile_settings)
            for k in patch_keys:
                if "diffusion_model." in k:
                    # Remove the prefix to get the attribute path
                    key = k.replace('diffusion_model.', '')
                    attributes = key.split('.')
                    # Start with the diffusion_model object
                    block = model.get_model_object("diffusion_model")
                    # Navigate through the attributes to get to the block
                    for attr in attributes:
                        if attr.isdigit():
                            block = block[int(attr)]
                        else:
                            block = getattr(block, attr)
                    # Compile the block
                    compiled_block = torch.compile(block, mode=compile_settings["mode"], dynamic=compile_settings["dynamic"], fullgraph=compile_settings["fullgraph"], backend=compile_settings["backend"])
                    # Add the compiled block back as an object patch
                    model.add_object_patch(k, compiled_block)
    return (new_modelpatcher, new_clip)

class PatchModelPatcherOrder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "patch_order": (["object_patch_first", "weight_patch_first"], {"default": "weight_patch_first", "tooltip": "Patch the comfy patch_model function to load weight patches (LoRAs) before compiling the model"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = "Patch the comfy patch_model function patching order, useful for torch.compile (used as object_patch) as it should come last if you want to use LoRAs with compile"
    EXPERIMENTAL = True

    def patch(self, model, patch_order):
        comfy.model_patcher.ModelPatcher.temp_object_patches_backup = {}
        if patch_order == "weight_patch_first":
            comfy.model_patcher.ModelPatcher.patch_model = patched_patch_model
            comfy.sd.load_lora_for_models = patched_load_lora_for_models
        else:
            comfy.model_patcher.ModelPatcher.patch_model = original_patch_model
            comfy.sd.load_lora_for_models = original_load_lora_for_models

        return model,

class TorchCompileModelFluxAdvanced:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "double_blocks": ("STRING", {"default": "0-18", "multiline": True}),
                    "single_blocks": ("STRING", {"default": "0-37", "multiline": True}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def parse_blocks(self, blocks_str):
        blocks = []
        for part in blocks_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                blocks.extend(range(start, end + 1))
            else:
                blocks.append(int(part))
        return blocks

    def patch(self, model, backend, mode, fullgraph, single_blocks, double_blocks, dynamic):
        single_block_list = self.parse_blocks(single_blocks)
        double_block_list = self.parse_blocks(double_blocks)
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")

        if not self._compiled:
            try:
                for i, block in enumerate(diffusion_model.double_blocks):
                    if i in double_block_list:
                        #print("Compiling double_block", i)
                        m.add_object_patch(f"diffusion_model.double_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                for i, block in enumerate(diffusion_model.single_blocks):
                    if i in single_block_list:
                        #print("Compiling single block", i)
                        m.add_object_patch(f"diffusion_model.single_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model")

        return (m, )
        # rest of the layers that are not patched
        # diffusion_model.final_layer = torch.compile(diffusion_model.final_layer, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.guidance_in = torch.compile(diffusion_model.guidance_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.img_in = torch.compile(diffusion_model.img_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.time_in = torch.compile(diffusion_model.time_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.txt_in = torch.compile(diffusion_model.txt_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.vector_in = torch.compile(diffusion_model.vector_in, mode=mode, fullgraph=fullgraph, backend=backend)

class TorchCompileModelHyVideo:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile single blocks"}),
                "compile_double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile double blocks"}),
                "compile_txt_in": ("BOOLEAN", {"default": False, "tooltip": "Compile txt_in layers"}),
                "compile_vector_in": ("BOOLEAN", {"default": False, "tooltip": "Compile vector_in layers"}),
                "compile_final_layer": ("BOOLEAN", {"default": False, "tooltip": "Compile final layer"}),

            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def patch(self, model, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_single_blocks, compile_double_blocks, compile_txt_in, compile_vector_in, compile_final_layer):
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        if not self._compiled:
            try:
                if compile_single_blocks:
                    for i, block in enumerate(diffusion_model.single_blocks):
                        compiled_block = torch.compile(block, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                        m.add_object_patch(f"diffusion_model.single_blocks.{i}", compiled_block)
                if compile_double_blocks:
                    for i, block in enumerate(diffusion_model.double_blocks):
                        compiled_block = torch.compile(block, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                        m.add_object_patch(f"diffusion_model.double_blocks.{i}", compiled_block)
                if compile_txt_in:
                    compiled_block = torch.compile(diffusion_model.txt_in, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                    m.add_object_patch("diffusion_model.txt_in", compiled_block)
                if compile_vector_in:
                    compiled_block = torch.compile(diffusion_model.vector_in, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                    m.add_object_patch("diffusion_model.vector_in", compiled_block)
                if compile_final_layer:
                    compiled_block = torch.compile(diffusion_model.final_layer, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                    m.add_object_patch("diffusion_model.final_layer", compiled_block)
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model")
        return (m, )

class TorchCompileModelWanVideo:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_transformer_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile all transformer blocks"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def patch(self, model, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer_blocks):
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        if not self._compiled:
            try:
                if compile_transformer_blocks:
                    for i, block in enumerate(diffusion_model.blocks):
                        compiled_block = torch.compile(block, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                        m.add_object_patch(f"diffusion_model.blocks.{i}", compiled_block)
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model")
        return (m, )

class TorchCompileVAE:
    def __init__(self):
        self._compiled_encoder = False
        self._compiled_decoder = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("VAE",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "compile_encoder": ("BOOLEAN", {"default": True, "tooltip": "Compile encoder"}),
                    "compile_decoder": ("BOOLEAN", {"default": True, "tooltip": "Compile decoder"}),
                }}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "compile"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def compile(self, vae, backend, mode, fullgraph, compile_encoder, compile_decoder):
        if compile_encoder:
            if not self._compiled_encoder:
                encoder_name = "encoder"
                if hasattr(vae.first_stage_model, "taesd_encoder"):
                    encoder_name = "taesd_encoder"

                try:
                    setattr(
                        vae.first_stage_model,
                        encoder_name,
                        torch.compile(
                            getattr(vae.first_stage_model, encoder_name),
                            mode=mode,
                            fullgraph=fullgraph,
                            backend=backend,
                        ),
                    )
                    self._compiled_encoder = True
                except:
                    raise RuntimeError("Failed to compile model")
        if compile_decoder:
            if not self._compiled_decoder:
                decoder_name = "decoder"
                if hasattr(vae.first_stage_model, "taesd_decoder"):
                    decoder_name = "taesd_decoder"

                try:
                    setattr(
                        vae.first_stage_model,
                        decoder_name,
                        torch.compile(
                            getattr(vae.first_stage_model, decoder_name),
                            mode=mode,
                            fullgraph=fullgraph,
                            backend=backend,
                        ),
                    )
                    self._compiled_decoder = True
                except:
                    raise RuntimeError("Failed to compile model")
        return (vae, )

class TorchCompileControlNet:
    def __init__(self):
        self._compiled= False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "controlnet": ("CONTROL_NET",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                }}
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "compile"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def compile(self, controlnet, backend, mode, fullgraph):
        if not self._compiled:
            try:
                # for i, block in enumerate(controlnet.control_model.double_blocks):
                #     print("Compiling controlnet double_block", i)
                #     controlnet.control_model.double_blocks[i] = torch.compile(block, mode=mode, fullgraph=fullgraph, backend=backend)
                controlnet.control_model = torch.compile(controlnet.control_model, mode=mode, fullgraph=fullgraph, backend=backend)
                self._compiled = True
            except:
                self._compiled = False
                raise RuntimeError("Failed to compile model")

        return (controlnet, )

class TorchCompileLTXModel:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def patch(self, model, backend, mode, fullgraph, dynamic):
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")

        if not self._compiled:
            try:
                for i, block in enumerate(diffusion_model.transformer_blocks):
                        compiled_block = torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend)
                        m.add_object_patch(f"diffusion_model.transformer_blocks.{i}", compiled_block)
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)

            except:
                raise RuntimeError("Failed to compile model")

        return (m, )

class TorchCompileCosmosModel:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                    "dynamo_cache_size_limit": ("INT", {"default": 64, "tooltip": "Set the dynamo cache size limit"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "KJNodes/torchcompile"
    EXPERIMENTAL = True

    def patch(self, model, backend, mode, fullgraph, dynamic, dynamo_cache_size_limit):

        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit

        if not self._compiled:
            try:
                for name, block in diffusion_model.blocks.items():
                    #print(f"Compiling block {name}")
                    compiled_block = torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend)
                    m.add_object_patch(f"diffusion_model.blocks.{name}", compiled_block)
                    #diffusion_model.blocks[name] = compiled_block

                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)

            except:
                raise RuntimeError("Failed to compile model")

        return (m, )


#teacache

try:
    from comfy.ldm.wan.model import sinusoidal_embedding_1d
except:
    pass
from einops import repeat
from unittest.mock import patch
from contextlib import nullcontext
import numpy as np

def relative_l1_distance(last_tensor, current_tensor):
    l1_distance = torch.abs(last_tensor - current_tensor).mean()
    norm = torch.abs(last_tensor).mean()
    relative_l1_distance = l1_distance / norm
    return relative_l1_distance.to(torch.float32)

#for now as there doesn't seem to be a way to pass transformer_options to the forward_orig currently
def teacache_wanvideo_forward(self, x, timestep, context, clip_fea=None, **kwargs):
        bs, c, t, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
        img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs, **kwargs)[:, :, :t, :h, :w]

def teacache_wanvideo_forward_orig(self, x, t, context, clip_fea=None, freqs=None, **kwargs):
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)
        if clip_fea is not None and self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        #teacache for cond and uncond separately
        rel_l1_thresh = kwargs["transformer_options"]["rel_l1_thresh"]
        cache_device = kwargs["transformer_options"]["teacache_device"]
        is_cond = True if kwargs["transformer_options"]["cond_or_uncond"] == [0] else False

        should_calc = True
        suffix = "cond" if is_cond else "uncond"

        # Init cache dict if not exists
        if not hasattr(self, 'teacache_state'):
            self.teacache_state = {
                'cond': {'accumulated_rel_l1_distance': 0, 'prev_input': None,
                        'teacache_skipped_steps': 0, 'previous_residual': None},
                'uncond': {'accumulated_rel_l1_distance': 0, 'prev_input': None,
                          'teacache_skipped_steps': 0, 'previous_residual': None}
            }
            logging.info("\nTeaCache: Initialized")

        cache = self.teacache_state[suffix]

        if cache['prev_input'] is not None:
            if kwargs["transformer_options"]["coefficients"] == []:
                temb_relative_l1 = relative_l1_distance(cache['prev_input'], e0)
                curr_acc_dist = cache['accumulated_rel_l1_distance'] + temb_relative_l1
            else:
                rescale_func = np.poly1d(kwargs["transformer_options"]["coefficients"])
                curr_acc_dist = cache['accumulated_rel_l1_distance'] + rescale_func(((e-cache['prev_input']).abs().mean() / cache['prev_input'].abs().mean()).cpu().item())
            try:
                if curr_acc_dist < rel_l1_thresh:
                    should_calc = False
                    cache['accumulated_rel_l1_distance'] = curr_acc_dist
                else:
                    should_calc = True
                    cache['accumulated_rel_l1_distance'] = 0
            except:
                should_calc = True
                cache['accumulated_rel_l1_distance'] = 0

        if kwargs["transformer_options"]["coefficients"] == []:
            cache['prev_input'] = e0.clone().detach()
        else:
            cache['prev_input'] = e.clone().detach()

        if not should_calc:
            x += cache['previous_residual'].to(x.device)
            cache['teacache_skipped_steps'] += 1
            #print(f"TeaCache: Skipping {suffix} step")

        if should_calc:
            original_x = x.clone().detach()
            # arguments
            block_wargs = dict(
                e=e0,
                freqs=freqs,
                context=context)

            for block in self.blocks:
                x = block(x, **block_wargs)

            cache['previous_residual']  = (x - original_x).to(cache_device)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

class WanVideoTeaCacheKJ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Threshold for to determine when to apply the cache, compromise between speed and accuracy."}),
                "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of the steps to use with TeaCache."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of the steps to use with TeaCache."}),
                "cache_device": (["main_device", "offload_device"], {"default": "offload_device", "tooltip": "Device to cache to"}),
                "coefficients": (["disabled", "1.3B", "14B", "i2v_480", "i2v_720"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_teacache"
    CATEGORY = "KJNodes/teacache"
    DESCRIPTION = """
Patch WanVideo model to use TeaCache. Speeds up inference by caching the output and applying it instead of doing the step.
Best results are achieved by choosing the appropriate coefficients for the model.
Early steps should never be skipped, with too aggressive values this can happen and the motion suffers. Starting later can help with that too.
When NOT using coefficients the threshold value should be about 10 times smaller than the value used with coefficients.
"""
    EXPERIMENTAL = True

    def patch_teacache(self, model, rel_l1_thresh, start_percent, end_percent, cache_device, coefficients):
        if rel_l1_thresh == 0:
            return (model,)

        # type_str = str(type(model.model.model_config).__name__)
        if model.model.diffusion_model.dim == 1536:
            model_type ="1.3B"
        # else:
        #     if "WAN21_T2V" in type_str:
        #         model_type = "14B"
        #     elif "WAN21_I2V" in type_str:
        #         model_type = "i2v_480"
        #     else:
        #         model_type = "i2v_720" #how to detect this?


        teacache_coefficients_map = {
            "disabled": [],
            "1.3B": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
            "14B": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
            "i2v_480": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
            "i2v_720": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
        }
        coefficients = teacache_coefficients_map[coefficients]

        teacache_device = mm.get_torch_device() if cache_device == "main_device" else mm.unet_offload_device()

        model_clone = model.clone()
        if 'transformer_options' not in model_clone.model_options:
            model_clone.model_options['transformer_options'] = {}
        model_clone.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        model_clone.model_options["transformer_options"]["teacache_device"] = teacache_device
        model_clone.model_options["transformer_options"]["coefficients"] = coefficients
        diffusion_model = model_clone.get_model_object("diffusion_model")

        def outer_wrapper(start_percent, end_percent):
            def unet_wrapper_function(model_function, kwargs):
                input = kwargs["input"]
                timestep = kwargs["timestep"]
                c = kwargs["c"]
                sigmas = c["transformer_options"]["sample_sigmas"]
                cond_or_uncond = kwargs["cond_or_uncond"]
                last_step = (len(sigmas) - 1)

                matched_step_index = (sigmas == timestep[0] ).nonzero()
                if len(matched_step_index) > 0:
                    current_step_index = matched_step_index.item()
                else:
                    for i in range(len(sigmas) - 1):
                        # walk from beginning of steps until crossing the timestep
                        if (sigmas[i] - timestep) * (sigmas[i + 1] - timestep) <= 0:
                            current_step_index = i
                            break
                    else:
                        current_step_index = 0

                if current_step_index == 0:
                    if hasattr(diffusion_model, "teacache_state"):
                        delattr(diffusion_model, "teacache_state")
                        logging.info("\nResetting TeaCache state")

                current_percent = current_step_index / (len(sigmas) - 1)
                if start_percent <= current_percent <= end_percent:
                    c["transformer_options"]["teacache_enabled"] = True

                    context = patch.multiple(
                        diffusion_model,
                        forward=teacache_wanvideo_forward.__get__(diffusion_model, diffusion_model.__class__),
                        forward_orig=teacache_wanvideo_forward_orig.__get__(diffusion_model, diffusion_model.__class__)
                    )
                else:
                    context = nullcontext()
                with context:
                    out = model_function(input, timestep, **c)
                    if current_step_index+1 == last_step and hasattr(diffusion_model, "teacache_state"):
                        if cond_or_uncond[0] == 0:
                            skipped_steps_cond = diffusion_model.teacache_state["cond"]["teacache_skipped_steps"]
                            skipped_steps_uncond = diffusion_model.teacache_state["uncond"]["teacache_skipped_steps"]

                            logging.info("-----------------------------------")
                            logging.info(f"TeaCache skipped:")
                            logging.info(f"{skipped_steps_cond} cond steps")
                            logging.info(f"{skipped_steps_uncond} uncond step")
                            logging.info(f"out of {last_step} steps")
                            logging.info("-----------------------------------")
                    return out
            return unet_wrapper_function

        model_clone.set_model_unet_function_wrapper(outer_wrapper(start_percent=start_percent, end_percent=end_percent))

        return (model_clone,)
