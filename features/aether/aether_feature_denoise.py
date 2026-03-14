import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import imageio.v3 as iio
import numpy as np
import torch
from aether.pipelines.aetherv1_pipeline_cogvideox import (  # noqa: E402
    AetherV1PipelineCogVideoX,
    AetherV1PipelineOutput,
    retrieve_latents,
    retrieve_timesteps,
)
from aether.utils.postprocess_utils import colorize_depth
from diffusers.image_processor import PipelineImageInput
from diffusers.models import CogVideoXTransformer3DModel
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import (
    AutoencoderKLCogVideoX,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    export_to_video,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from einops import rearrange
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

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
        p = self.config.patch_size
        p_t = self.config.patch_size_t

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
                assert p_t is None
                features = features.reshape(
                    batch_size, num_frames, height // p, width // p, -1
                )
                all_hidden_states[i] = features

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
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


class OneStepAetherPipeline(AetherV1PipelineCogVideoX):
    """
    Pipeline for extracting features from intermediate layers of CogVideoX's transformer model.

    This pipeline adapts CogVideoX to allow extraction of intermediate features from a single denoising step.
    """

    def visualize(self, latents: torch.Tensor, postfix: str = ""):
        rgb_latents = latents[:, :, : self.vae.config.latent_channels]
        disparity_latents = latents[
            :, :, self.vae.config.latent_channels : self.vae.config.latent_channels * 2
        ]

        rgb_video = self.decode_latents(rgb_latents)
        rgb_video = self.video_processor.postprocess_video(
            video=rgb_video, output_type="np"
        )[0]

        disparity_video = self.decode_latents(disparity_latents)
        disparity_video = disparity_video.mean(dim=1, keepdim=False)
        disparity_video = disparity_video * 0.5 + 0.5
        disparity_video = torch.square(disparity_video)
        disparity_video = disparity_video.float().cpu().numpy()[0]

        iio.imwrite(
            f"output_rgb{postfix}.mp4",
            (np.clip(rgb_video, 0, 1) * 255).astype(np.uint8),
            fps=8,
        )
        iio.imwrite(
            f"output_disparity{postfix}.mp4",
            (colorize_depth(disparity_video) * 255).astype(np.uint8),
            fps=8,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        task: Optional[str] = None,
        image: Optional[PipelineImageInput] = None,
        video: Optional[PipelineImageInput] = None,
        goal: Optional[PipelineImageInput] = None,
        raymap: Optional[Union[torch.Tensor, np.ndarray]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        guidance_scale: Optional[float] = None,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict] = None,
        fps: Optional[int] = None,
    ) -> Union[AetherV1PipelineOutput, Tuple]:
        if task is None:
            if video is not None:
                task = "reconstruction"
            elif goal is not None:
                task = "planning"
            else:
                task = "prediction"

        height = (
            height
            or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        )
        width = (
            width
            or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        )
        num_frames = num_frames or self.transformer.config.sample_frames
        fps = fps or self._base_fps

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            task=task,
            image=image,
            video=video,
            goal=goal,
            raymap=raymap,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )

        # 2. Preprocess inputs
        image, goal, video, raymap = self.preprocess_inputs(
            image=image,
            goal=goal,
            video=video,
            raymap=raymap,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        batch_size = 1

        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds = self.empty_prompt_embeds.to(device)

        num_inference_steps = (
            num_inference_steps or self._default_num_inference_steps[task]
        )
        guidance_scale = guidance_scale or self._default_guidance_scale[task]
        use_dynamic_cfg = use_dynamic_cfg or self._default_use_dynamic_cfg[task]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latents, condition_latents = self.prepare_latents(
            image,
            goal,
            video,
            raymap,
            batch_size * num_videos_per_prompt,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(
                height, width, latents.size(1), device, fps=fps
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = (
            None
            if self.transformer.config.ofs_embed_dim is None
            else latents.new_full((1,), fill_value=2.0)
        )

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if do_classifier_free_guidance:
                    if task == "planning":
                        assert goal is not None
                        uncond = condition_latents.clone()
                        uncond[:, :, : self.vae.config.latent_channels] = 0
                        latent_condition = torch.cat([uncond, condition_latents])
                    elif task == "prediction":
                        uncond = condition_latents.clone()
                        uncond[:, :1, : self.vae.config.latent_channels] = 0
                        latent_condition = torch.cat([uncond, condition_latents])
                    else:
                        raise ValueError(
                            f"Task {task} not supported for classifier-free guidance."
                        )

                else:
                    latent_condition = condition_latents

                latent_model_input = torch.cat(
                    [latent_model_input, latent_condition], dim=2
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds.repeat(
                        latent_model_input.shape[0], 1, 1
                    ),
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (
                            1
                            - math.cos(
                                math.pi
                                * (
                                    (num_inference_steps - t.item())
                                    / num_inference_steps
                                )
                                ** 5.0
                            )
                        )
                        / 2
                    )

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None
        self.visualize(latents, postfix="_recon")

        # rgb_latents = latents[:, :, : self.vae.config.latent_channels]
        # disparity_latents = latents[
        #     :, :, self.vae.config.latent_channels : self.vae.config.latent_channels * 2
        # ]
        # camera_latents = latents[:, :, self.vae.config.latent_channels * 2 :]

        # rgb_video = self.decode_latents(rgb_latents)
        # rgb_video = self.video_processor.postprocess_video(
        #     video=rgb_video, output_type="np"
        # )

        # disparity_video = self.decode_latents(disparity_latents)
        # disparity_video = disparity_video.mean(dim=1, keepdim=False)
        # disparity_video = disparity_video * 0.5 + 0.5
        # disparity_video = torch.square(disparity_video)
        # disparity_video = disparity_video.float().cpu().numpy()

        # raymap = (
        #     rearrange(camera_latents, "b t (n c) h w -> b (n t) c h w", n=4)[
        #         :, -rgb_video.shape[1] :, :, :
        #     ]
        #     .float()
        #     .cpu()
        #     .numpy()
        # )

        geometry_latents = latents[:, :, self.vae.config.latent_channels :]
        return geometry_latents

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        video: List[Image.Image],
        t: int,
        output_layers: List[int],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        output_hidden_states: bool = True,
        task: str = "videogen",
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
        assert task in ["videogen", "reconstruction"], f"Task {task} not supported."
        device = "cuda" if torch.cuda.is_available() else "cpu"
        height = 480  # height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = 720  # width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = 41
        fps = 8

        # get the geometry latents
        geometry_latents = self.reconstruct(
            task="reconstruction",
            video=video,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            generator=generator,
        )

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=torch.float16
        )
        video = self.video_processor.preprocess_video(
            video, height=height, width=width
        ).to(device=self._execution_device, dtype=torch.bfloat16)
        video_cond = video.squeeze(0).permute(1, 0, 2, 3)  # [41, 3, 480, 720]
        prompt_embeds = self.empty_prompt_embeds.to(device)
        _, condition_latents = self.prepare_latents(
            image if task == "videogen" else None,
            None,
            video_cond if task == "reconstruction" else None,
            None,
            1,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

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

        # Concatenate geometry latents with video latents
        latents = torch.cat([latents, geometry_latents], dim=2)

        self.visualize(latents, postfix="_swapped")

        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(
                height, width, latents.size(1), device, fps=fps
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

        # Convert timestep to tensor and prepare noise
        t_idx = torch.tensor([t], dtype=torch.long, device=device)
        # t_idx: 261 -> t_input: 894 (reversed)
        t_input = self.scheduler.timesteps[t_idx].to(device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t_input)

        self.visualize(latents_noisy, postfix="_noisy")

        # Ensure the transformer supports feature output
        if not isinstance(self.transformer, TransformerCogVideoXWithFeatureOutput):
            raise ValueError(
                "The transformer model must be an instance of TransformerCogVideoXWithFeatureOutput for feature extraction."
            )

        from tqdm import tqdm

        old_pred_original_sample = None
        for i in tqdm(range(t, num_inference_steps)):
            t_idx = torch.tensor([t], dtype=torch.long, device=device)
            t_input = self.scheduler.timesteps[i].to(device)
            latent_model_input = torch.cat([latents_noisy, condition_latents], dim=2)

            noise_pred, all_hidden_states = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=t_input.expand(latents.shape[0]),
                ofs=ofs_emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
                output_hidden_states=output_hidden_states,
                output_layers=output_layers,
            )
            noise_pred = noise_pred.float()
            if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                latents_noisy = self.scheduler.step(
                    noise_pred,
                    t_input,
                    latents_noisy,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
            else:
                latents_noisy, old_pred_original_sample = self.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t_input,
                    timesteps[i - 1] if i > 0 else None,
                    latents_noisy,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents_noisy = latents.to(prompt_embeds.dtype)
        self.visualize(latents_noisy, postfix="_final")

        return all_hidden_states


from functools import lru_cache


@lru_cache(maxsize=None)
def get_aether_featurizer():
    """
    Build the full AetherFeaturizer pipeline exactly **once** per (model_id, device).

    The first call incurs checkpoint loading; later calls return the
    very same AetherFeaturizer object (modules & tensors already live in RAM/GPU).
    """
    return AetherFeaturizer()


class AetherFeaturizer:
    """
    A wrapper class to simplify feature extraction from Aether video models.

    This class provides a simplified interface for extracting features from
    intermediate layers of the Aether transformer model.
    """

    def __init__(self):
        """
        Initialize the AetherFeaturizer with a specific Aether model.
        """
        pipeline = OneStepAetherPipeline(
            tokenizer=AutoTokenizer.from_pretrained(
                "THUDM/CogVideoX-5b-I2V",
                subfolder="tokenizer",
            ),
            text_encoder=T5EncoderModel.from_pretrained(
                "THUDM/CogVideoX-5b-I2V", subfolder="text_encoder"
            ),
            vae=AutoencoderKLCogVideoX.from_pretrained(
                "THUDM/CogVideoX-5b-I2V",
                subfolder="vae",
                torch_dtype=torch.bfloat16,
            ),
            scheduler=CogVideoXDPMScheduler.from_pretrained(
                "THUDM/CogVideoX-5b-I2V", subfolder="scheduler"
            ),
            transformer=TransformerCogVideoXWithFeatureOutput.from_pretrained(
                "AetherWorldModel/AetherV1",
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            ),
        )
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
        pipeline.to("cuda")
        self.pipe = pipeline

    @torch.no_grad()
    def forward(
        self,
        video: List[Image.Image],
        t: int = 499,
        output_layer_indices: List[int] = [1],
        task: str = "videogen",
    ) -> Dict[int, torch.FloatTensor]:
        """
        Extract features from specific layers of the Aether transformer.

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

        # Extract features from the transformer
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            all_hidden_states = self.pipe(
                video[0], video, t=t, output_layers=output_layer_indices, task=task
            )

        # Get features from the specified layers
        extracted_features = {
            idx: all_hidden_states[idx]
            for idx in output_layer_indices
            if idx in all_hidden_states
        }

        return extracted_features


if __name__ == "__main__":
    # Example usage
    model_id = "THUDM/CogVideoX-5b-I2V"
    # load first 81 images under vidfm3d/data/DL3DV/DL3DV-10K/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/images_8
    frame_folder = "vidfm3d/data/DL3DV/DL3DV-raw/DL3DV-10K/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/images_4"
    # frame_folder = "vidfm3d/data/CO3D/CO3D-raw/apple/110_13054_23182/images"
    video = []
    for i in range(1, 42):
        img_path = f"{frame_folder}/frame_{(i*2-1):05d}.png"
        img = Image.open(img_path).convert("RGB")
        video.append(img)

    featurizer = get_aether_featurizer()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        features = featurizer.forward(
            video, t=25, output_layer_indices=[1], task="reconstruction"
        )

    for layer_idx, feature in features.items():
        print(f"Layer {layer_idx} feature shape: {feature.shape}")
