# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Explicitly set the environment variable TOKENIZERS_PARALLELISM to false
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from PIL import Image
from transformers import CLIPVisionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TransformerWanWithFeatureOutput(WanTransformer3DModel):
    """
    A version of WanTransformer3DModel that can output intermediate features.
    This is useful for feature extraction for downstream tasks.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        output_hidden_states: bool = False,
        output_layers: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: Input tensor of shape [batch_size, channels, frames, height, width]
            timestep: Timestep embedding
            encoder_hidden_states: Text encoder hidden states
            encoder_hidden_states_image: Optional image encoder hidden states
            return_dict: Whether to return a dictionary
            attention_kwargs: Additional kwargs for attention
            output_hidden_states: Whether to output all hidden states
            output_layers: List of specific layer indices to output features from
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
        ) = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        # Initialize dict to store features if needed
        all_hidden_states = {}

        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )

            # Save features if requested
            if output_hidden_states or (
                output_layers is not None and i in output_layers
            ):
                # Reshape to proper dimensions for feature extraction
                features = hidden_states.detach().clone()
                all_hidden_states[i] = features

        # Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (
                (output, all_hidden_states)
                if output_hidden_states or output_layers
                else (output,)
            )

        # If return_dict is True, include features in the return object
        if output_hidden_states or output_layers:
            return Transformer2DModelOutput(sample=output), all_hidden_states
        else:
            return Transformer2DModelOutput(sample=output)


class OneStepWanPipeline(WanImageToVideoPipeline):
    """
    Pipeline for extracting features from intermediate layers of Wan's transformer model.

    This pipeline adapts WanImageToVideoPipeline to allow extraction of intermediate features from a single denoising step.
    """

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        video: List[Image.Image],
        t: int,
        output_layers: List[int],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_hidden_states: bool = False,
    ):
        """
        Args:
            video_tensor (`torch.FloatTensor`): Input video tensor of shape [batch_size, channels, frames, height, width]
            t (`int`): Timestep for which to add noise and extract features
            output_layers (`List[int]`): List of transformer layer indices to extract features from
            negative_prompt (`str` or `List[str]`, *optional*): Negative prompts for guidance
            generator (`torch.Generator`, *optional*): Generator for random noise
            prompt_embeds (`torch.FloatTensor`, *optional*): Pre-computed prompt embeddings
            callback (`Callable`, *optional*): Callback function
            callback_steps (`int`, *optional*): Steps between callbacks
            cross_attention_kwargs (`Dict[str, Any]`, *optional*): Additional kwargs for cross attention
            output_hidden_states (`bool`, *optional*): Whether to output all hidden states or only specified layers

        Returns:
            `Dict[str, torch.FloatTensor]`: Dictionary containing features from specified transformer layers
        """
        device = self._execution_device
        height = 480  # height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = 832  # width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = 81
        video = self.video_processor.preprocess_video(
            video, height=height, width=width
        ).to(device, dtype=torch.float32)
        image_embeds = self.encode_image(image, "cuda").to(self.transformer.dtype)
        self.image_encoder.cpu()
        torch.cuda.empty_cache()

        num_inference_steps = 1000
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Encode video to latent space using VAE
        latents = self.vae.encode(video).latent_dist.mean
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)

        latents = (latents - latents_mean) * latents_std

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=torch.float32
        )
        _, condition = self.prepare_latents(
            image,
            1,
            self.vae.config.z_dim,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # Convert timestep to tensor and prepare noise
        t_idx = torch.tensor([t], dtype=torch.long, device=device)
        t_input = self.scheduler.timesteps[t_idx].to(device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t_input)

        # Ensure the transformer supports feature output
        if not isinstance(self.transformer, TransformerWanWithFeatureOutput):
            raise ValueError(
                "The transformer model must be an instance of TransformerWanWithFeatureOutput for feature extraction."
            )

        latent_model_input = torch.cat([latents_noisy, condition], dim=1).to(
            self.transformer.dtype
        )
        transformer_output, all_hidden_states = self.transformer(
            hidden_states=latent_model_input,
            timestep=t_input.expand(latents.shape[0]),
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=image_embeds,
            attention_kwargs=cross_attention_kwargs,
            return_dict=True,
            output_hidden_states=output_hidden_states,
            output_layers=output_layers,
        )

        return transformer_output, all_hidden_states


from functools import lru_cache


@lru_cache(maxsize=None)
def get_wan_featurizer_i2v(
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", null_prompt: str = ""
):
    """
    Build the full WanFeaturizer pipeline exactly **once** per (model_id, device).

    The first call incurs checkpoint loading; later calls return the
    very same WanFeaturizer object (modules & tensors already live in RAM/GPU).
    """
    return WanFeaturizerI2V(
        model_id=model_id,
        null_prompt=null_prompt,
    )


class WanFeaturizerI2V:
    """
    A wrapper class to simplify feature extraction from Wan video models.

    This class provides a simplified interface for extracting features from
    intermediate layers of the Wan transformer model.
    """

    def __init__(self, model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", null_prompt=""):
        """
        Initialize the WanFeaturizer with a specific Wan model.

        Args:
            model_id (`str`): Identifier for the Wan model to use
            null_prompt (`str`): Prompt used for unconditional generation
        """
        # Load the transformer model with feature output capability
        transformer = TransformerWanWithFeatureOutput.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        image_encoder = CLIPVisionModel.from_pretrained(
            model_id, subfolder="image_encoder", torch_dtype=torch.float16
        )
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16
        )

        # Initialize the pipeline with the modified transformer
        onestep_pipe = OneStepWanPipeline.from_pretrained(
            model_id,
            vae=vae,
            transformer=transformer,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16,
        )

        # Move to GPU and enable optimizations
        onestep_pipe = onestep_pipe.to("cuda")

        # Precompute null prompt embeddings
        with torch.no_grad():
            null_prompt_embeds, _ = onestep_pipe.encode_prompt(
                prompt=null_prompt,
                do_classifier_free_guidance=False,
                device="cuda",
            )

        self.null_prompt_embeds = null_prompt_embeds
        self.null_prompt = null_prompt
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(
        self,
        video: List[Image.Image],
        prompt: str = "",
        t: int = 261,
        output_layer_indices: List[int] = [1],
        ensemble_size: int = 1,
    ) -> Dict[int, torch.FloatTensor]:
        """
        Extract features from specific layers of the Wan transformer.

        Args:
            video_tensor (`torch.FloatTensor`): Input video tensor of shape [1, C, F, H, W] or [C, F, H, W]
            prompt (`str`): The prompt to use for conditioning
            t (`int`): Timestep for feature extraction (0-1000)
            output_layer_indices (`List[int]`): Which transformer blocks to extract features from
            ensemble_size (`int`): Number of repeated videos in batch for feature extraction

        Returns:
            `Dict[int, torch.FloatTensor]`: Dictionary mapping layer index to extracted features
                                           with shape [1, sequence_length, feature_dim]
        """
        # Get the appropriate prompt embeddings
        if prompt == self.null_prompt:
            prompt_embeds = self.null_prompt_embeds
        else:
            prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                device="cuda",
            )

        # offload to cpu
        self.pipe.text_encoder.cpu()

        # load and process the image
        image = video[-1]
        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        mod_value = (
            self.pipe.vae_scale_factor_spatial
            * self.pipe.transformer.config.patch_size[1]
        )
        height_image = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width_image = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width_image, height_image))

        # Repeat prompt embeddings for ensemble
        if ensemble_size > 1:
            prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)

        # Extract features from the transformer
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, all_hidden_states = self.pipe(
                image=image,
                video=video,
                t=t,
                output_layers=output_layer_indices,
                prompt_embeds=prompt_embeds,
            )

        # Get features from the specified layers
        extracted_features = {
            idx: all_hidden_states[idx]
            for idx in output_layer_indices
            if idx in all_hidden_states
        }

        return extracted_features
