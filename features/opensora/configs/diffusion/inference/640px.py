_base_ = [  # inherit grammer from mmengine
    "256px.py",
    "plugins/sp.py",  # use sequence parallel
]

sampling_option = dict(
    resolution="640px",
)

model = dict(
    from_pretrained="./features/opensora/ckpts/Open_Sora_v2.safetensors",
)
ae = dict(
    from_pretrained="./features/opensora/ckpts/hunyuan_vae.safetensors",
)
t5 = dict(
    from_pretrained="./features/opensora/ckpts/google/t5-v1_1-xxl",
)
clip = dict(
    from_pretrained="./features/opensora/ckpts/openai/clip-vit-large-patch14",
)
