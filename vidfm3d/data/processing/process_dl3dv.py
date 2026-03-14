#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Wrapper to run extract_points.py on DL3DV scenes."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory for DL3DV (e.g., vidfm3d/DL3DV)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Subset to process (e.g., '1K', '2K', ..., '11K', or 'all'). Default is all.",
    )
    # Parse known arguments; forward any additional (unknown) arguments to extract_points.py.
    args, unknown = parser.parse_known_args()

    # Define input and output base directories.
    input_base = os.path.join(args.root, "DL3DV-10K")
    output_base = os.path.join(args.root, "DL3DV-processed")
    os.makedirs(output_base, exist_ok=True)

    # Set up logging: log to a file under DL3DV-processed and to stdout.
    log_file = os.path.join(output_base, f"processing-{args.subset}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Starting DL3DV extraction wrapper.")
    logging.info(f"Subsets to process: {args.subset}")
    logging.info(f"Input base: {input_base}")
    logging.info(f"Output base: {output_base}")

    # Determine which subsets to process.
    if args.subset.lower() == "all":
        subsets = [
            d
            for d in os.listdir(input_base)
            if os.path.isdir(os.path.join(input_base, d))
        ]
    else:
        subsets = [args.subset]

    # Process each subset.
    for subset in subsets:
        subset_input_dir = os.path.join(input_base, subset)
        if not os.path.isdir(subset_input_dir):
            logging.warning(
                f"Subset directory {subset_input_dir} does not exist. Skipping."
            )
            continue

        # Each scene is in a subfolder (named by a hash) that contains an "images_8" folder.
        hash_dirs = [
            d
            for d in os.listdir(subset_input_dir)
            if os.path.isdir(os.path.join(subset_input_dir, d))
        ]
        for hash_dir in hash_dirs:
            scene_dir = os.path.join(subset_input_dir, hash_dir, "images_4")
            if not os.path.isdir(scene_dir):
                logging.warning(
                    f"Scene directory {scene_dir} does not exist. Skipping."
                )
                continue

            # Build the output file path as: {output_base}/{subset}/{hash}.sft
            subset_output_dir = os.path.join(output_base, subset)
            os.makedirs(subset_output_dir, exist_ok=True)
            output_file = os.path.join(subset_output_dir, f"{hash_dir}.sft")

            # if the output already exists and we can load it, skip this scene
            if os.path.isfile(output_file):
                try:
                    from safetensors.torch import load_file

                    load_file(output_file)
                    logging.info(f"Output file {output_file} already exists. Skipping.")
                    continue
                except Exception as e:
                    logging.warning(
                        f"Failed to load existing output file {output_file}: {e}"
                    )

            # Build the command to call extract_points.py.
            # The updated extract_points.py accepts an output file path via the --output flag.
            cmd = [
                sys.executable,
                "-m",
                "vidfm3d.data.processing.dl3dv.extract_points",
                "--scene-dir",
                scene_dir,
                "--output",
                output_file,
            ]
            cmd.extend(unknown)

            logging.info(f"Processing scene: subset '{subset}', hash '{hash_dir}'")
            logging.info(f"Command: {' '.join(cmd)}")
            start_time = time.time()
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,  # or encoding='utf-8'
                )
                elapsed = time.time() - start_time
                logging.info(f"Finished scene '{hash_dir}' in {elapsed:.2f} seconds.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error processing scene {scene_dir}: {e}")
                logging.error(f"STDOUT: {e.stdout}")
                logging.error(f"STDERR: {e.stderr}")

    logging.info(f"DL3DV processing finished (log saved to {log_file}).")


if __name__ == "__main__":
    main()
