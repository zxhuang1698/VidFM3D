import os

import hydra
import torch
from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from omegaconf import OmegaConf

from f3r.models.fast3r import Fast3R
from f3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule


# -------------------------------
# load_lightning_checkpoint
# -------------------------------
def load_lightning_checkpoint(checkpoint_dir, device: torch.device):
    """
    Loads a model from a Lightning checkpoint.

    Args:
        checkpoint_dir: Path to the Lightning checkpoint directory
        device: Device to load the model on

    Returns: model, lit_module
    """
    print(f"Loading Lightning checkpoint from {checkpoint_dir}")

    # Create an empty model to hold the weights
    print("Creating an empty lightning module to hold the weights...")

    # Load the config from the checkpoint directory
    config_path = os.path.join(checkpoint_dir, ".hydra/config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at {config_path}. Make sure this is a valid Lightning checkpoint directory."
        )

    cfg = OmegaConf.load(config_path)

    # set these flags so that the model can inference on arbitrary aspect ratio images
    cfg.model.net.encoder_args.patch_embed_cls = "PatchEmbedDust3R"
    cfg.model.net.head_args.landscape_only = False

    lit_module = hydra.utils.instantiate(
        cfg.model, train_criterion=None, validation_criterion=None
    )

    # Check if checkpoint is a DeepSpeed checkpoint, if so, convert it to a regular checkpoint
    ds_ckpt_path = os.path.join(checkpoint_dir, "checkpoints/last.ckpt")
    regular_ckpt_path = os.path.join(checkpoint_dir, "checkpoints/last.ckpt")
    aggregated_ckpt_path = os.path.join(
        checkpoint_dir, "checkpoints/last_aggregated.ckpt"
    )

    if os.path.isdir(ds_ckpt_path):
        # It is a DeepSpeed checkpoint, convert it to a regular checkpoint
        print("DeepSpeed checkpoint detected, converting to FP32 state dict...")
        if not os.path.exists(aggregated_ckpt_path):
            convert_zero_checkpoint_to_fp32_state_dict(
                checkpoint_dir=ds_ckpt_path, output_file=aggregated_ckpt_path, tag=None
            )
        ckpt_path = aggregated_ckpt_path
    else:
        ckpt_path = regular_ckpt_path

    # Load the checkpoint into the lit_module
    print(f"Loading checkpoint from {ckpt_path}")
    lit_module = MultiViewDUSt3RLitModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        net=lit_module.net,
        train_criterion=lit_module.train_criterion,
        validation_criterion=lit_module.validation_criterion,
    )

    # Set model to evaluation mode
    lit_module.eval()
    model = lit_module.net.to(device)

    return model, lit_module


# -------------------------------
# load_model
# -------------------------------
def load_model(checkpoint_dir, device: torch.device, is_lightning_checkpoint=False):
    """
    Loads the model from the checkpoint.

    Args:
        checkpoint_dir: Path to the checkpoint or HF model name
        device: Device to load the model on
        is_lightning_checkpoint: Whether the checkpoint is from Lightning (default: False)

    Returns: model, lit_module.
    """
    if is_lightning_checkpoint:
        return load_lightning_checkpoint(checkpoint_dir, device)

    # it is a HF checkpoint, so load the model directly with HF `from_pretrained`
    model = Fast3R.from_pretrained(checkpoint_dir)
    model = model.to(device)

    # Create a lightweight lit_module wrapper for the model
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)

    # Set model to evaluation mode
    model.eval()
    lit_module.eval()

    return model, lit_module


def convert_checkpoint_to_hf_checkpoint(
    checkpoint_dir: str, output_path: str, push_to_hub: str = None
):
    """
    Converts a Lightning checkpoint to a HuggingFace checkpoint format and optionally pushes to hub.

    Args:
        checkpoint_dir: Path to the Lightning checkpoint directory
        output_path: Path where to save the HuggingFace checkpoint
        push_to_hub: Optional repository name to push the model to HuggingFace Hub (e.g. "username/model_name")

    Returns:
        None
    """
    # Load the model using the existing load_model function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, lit_module = load_model(checkpoint_dir, device, is_lightning_checkpoint=True)

    # Prepare config for saving
    config = {}

    # add all attributes of the model that ends with "_args" to the config
    for attr in dir(model):
        if attr.endswith("_args"):
            config[attr] = getattr(model, attr)

    # Save the model locally in HuggingFace format
    model.save_pretrained(output_path, config=config)
    print(f"Model saved to {output_path}")

    # Optionally push to HuggingFace Hub
    if push_to_hub is not None:
        print(f"Pushing model to HuggingFace Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub, config=config)
        print(f"Model successfully pushed to {push_to_hub}")


if __name__ == "__main__":
    # Example usage
    os.environ["pretrained_fast3r"] = "placeholder_that_does_not_matter"
    checkpoint_dir = "/path/to/lightning/checkpoint"
    output_path = "Fast3R_HF_Checkpoint"

    convert_checkpoint_to_hf_checkpoint(
        checkpoint_dir=checkpoint_dir,
        output_path=output_path,
        # push_to_hub="my_hf_username/my_hf_repo_name"
    )

    # Uncomment to test loading the model back
    # model_load_back = Fast3R.from_pretrained(output_path)
    # lit_module_load_back = MultiViewDUSt3RLitModule.load_for_inference(model_load_back)
    # print(model_load_back)
    # print(lit_module_load_back)
