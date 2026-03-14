from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.models import CogVideoXTransformer3DModel
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import (
    AutoencoderKLCogVideoX,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import (
    retrieve_latents,
    retrieve_timesteps,
)
from diffusers.schedulers.scheduling_ddim_cogvideox import CogVideoXDDIMScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    export_to_video,
    load_image,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from PIL import Image
from transformers import T5EncoderModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TransformerCogVideoXWithFeatureOutput(CogVideoXTransformer3DModel):
    """
    A version of CogVideoXTransformer3DModel that can output intermediate features.
    This is useful for feature extraction for downstream tasks.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_layers: Optional[List[int]] = None,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        all_hidden_states = {}
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )
            # Save features if requested
            if output_hidden_states or (
                output_layers is not None and i in output_layers
            ):
                # Reshape to proper dimensions for feature extraction
                features = hidden_states.detach().clone()
                all_hidden_states[i] = features

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(
                batch_size, num_frames, height // p, width // p, -1, p, p
            )
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size,
                (num_frames + p_t - 1) // p_t,
                height // p,
                width // p,
                -1,
                p_t,
                p,
                p,
            )
            output = (
                output.permute(0, 1, 5, 4, 2, 6, 3, 7)
                .flatten(6, 7)
                .flatten(4, 5)
                .flatten(1, 2)
            )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
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



class OneStepCogVideoXPipeline(CogVideoXImageToVideoPipeline):
    """
    Pipeline for extracting features from intermediate layers of CogVideoX's transformer model.

    This pipeline adapts CogVideoX to allow extraction of intermediate features from a single denoising step.
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
        attention_kwargs: Optional[Dict[str, Any]] = None,
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
            attention_kwargs (`Dict[str, Any]`, *optional*): Additional kwargs for cross attention
            output_hidden_states (`bool`, *optional*): Whether to output all hidden states or only specified layers

        Returns:
            `Dict[str, torch.FloatTensor]`: Dictionary containing features from specified transformer layers
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        height = 480  # height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = 720  # width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = 49
        video = self.video_processor.preprocess_video(
            video, height=height, width=width
        ).to(device, dtype=torch.float16)

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=torch.float16
        )
        _, image_latents = self.prepare_latents(
            image,
            1,
            self.transformer.config.in_channels // 2,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )  # [1, 11, 16, 60, 90]

        num_inference_steps = 50
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0)
        # timesteps = self.scheduler.timesteps

        # Encode video to latent space using VAE
        self.vae: AutoencoderKLCogVideoX
        latents = self.vae.encode(video).latent_dist.mean.permute(
            0, 2, 1, 3, 4
        )  # [1, 11, 16, 60, 90]
        if not self.vae.config.invert_scale_latents:
            latents = self.vae_scaling_factor_image * latents  # here
        else:
            latents = 1 / self.vae_scaling_factor_image * latents

        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(
                height, width, latents.size(1), device
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        ofs_emb = (
            None
            if self.transformer.config.ofs_embed_dim is None
            else latents.new_full((1,), fill_value=2.0)
        )

        torch.cuda.empty_cache()

        video = self.decode_latents(latents)
        video = self.video_processor.postprocess_video(video=video)
        export_to_video(video[0], "output_vae.mp4", fps=8)

        # Convert timestep to tensor and prepare noise
        t_idx = torch.tensor([t], dtype=torch.long, device=device)
        self.scheduler: CogVideoXDDIMScheduler
        t_input = self.scheduler.timesteps[t_idx].to(device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t_input)

        video = self.decode_latents(latents_noisy)
        video = self.video_processor.postprocess_video(video=video)
        export_to_video(video[0], "output_noisy.mp4", fps=8)

        # Ensure the transformer supports feature output
        if not isinstance(self.transformer, TransformerCogVideoXWithFeatureOutput):
            raise ValueError(
                "The transformer model must be an instance of TransformerCogVideoXWithFeatureOutput for feature extraction."
            )

        from tqdm import tqdm

        for i in tqdm(range(t, num_inference_steps)):
            t_idx = torch.tensor([t], dtype=torch.long, device=device)
            t_input = self.scheduler.timesteps[i].to(device)
            latent_model_input = torch.cat([latents_noisy, image_latents], dim=2)

            noise_pred, all_hidden_states = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=t_input.expand(latents.shape[0]),
                ofs=ofs_emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=True,
                output_hidden_states=output_hidden_states,
                output_layers=output_layers,
            )
            latents_noisy = self.scheduler.step(
                noise_pred.sample.float(),
                t_input,
                latents_noisy,
                **extra_step_kwargs,
                return_dict=False,
            )[0]

        video = self.decode_latents(latents_noisy)
        video = self.video_processor.postprocess_video(video=video)
        export_to_video(video[0], "output_denoised.mp4", fps=8)

        return noise_pred, all_hidden_states


class CogVideoXFeaturizer:
    """
    A wrapper class to simplify feature extraction from CogVideoX video models.

    This class provides a simplified interface for extracting features from
    intermediate layers of the CogVideoX transformer model.
    """

    def __init__(self, model_id="THUDM/CogVideoX-5b-I2V", null_prompt=""):
        """
        Initialize the CogVideoXFeaturizer with a specific CogVideoX model.

        Args:
            model_id (`str`): Identifier for the CogVideoX model to use
            null_prompt (`str`): Prompt used for unconditional generation
        """

        # Load the transformer model with feature output capability
        transformer = TransformerCogVideoXWithFeatureOutput.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float16
        )
        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float16
        )
        text_encoder = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float16
        )

        # Initialize the pipeline with the modified transformer
        onestep_pipe = OneStepCogVideoXPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.float16,
        ).to("cuda")

        # onestep_pipe.enable_sequential_cpu_offload()

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

        # load and process the image
        image = video[0]

        # Extract features from the transformer
        output, all_hidden_states = self.pipe(
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


if __name__ == "__main__":
    model_id = "THUDM/CogVideoX-5b-I2V"
    frame_folder = "path/to/frames"  # folder containing frame_00001.png, ...
    video = []
    for i in range(1, 50):
        img_path = f"{frame_folder}/frame_{i:05d}.png"
        img = Image.open(img_path).convert("RGB")
        video.append(img)

    featurizer = CogVideoXFeaturizer(model_id=model_id)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        features = featurizer.forward(video, t=25, output_layer_indices=[1])

    for layer_idx, feature in features.items():
        print(f"Layer {layer_idx} feature shape: {feature.shape}")
