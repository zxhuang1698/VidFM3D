import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel

model_id = "THUDM/CogVideoX-5b-I2V"


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


transformer = TransformerCogVideoXWithFeatureOutput.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.float16
)
text_encoder = T5EncoderModel.from_pretrained(
    model_id, subfolder="text_encoder", torch_dtype=torch.float16
)
vae = AutoencoderKLCogVideoX.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float16
)

# Create pipeline and run inference
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.float16,
)

pipe.enable_sequential_cpu_offload()

prompt = "A indoor playground with a castle in the middle."
image = load_image(
    "vidfm3d/data/DL3DV/DL3DV-raw/DL3DV-10K/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/images_4/frame_00001.png"
)

video = pipe(
    image=image,
    prompt=prompt,
    guidance_scale=1,
    use_dynamic_cfg=False,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=8)
