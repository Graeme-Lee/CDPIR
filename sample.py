# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample / reconstruct images from a pre-trained SiT model.
For SDE reconstruction, all supported images in a folder can be processed.
Optional evaluation computes PSNR/SSIM directly from numpy arrays.
"""

import argparse
import sys
from pathlib import Path
from time import time

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image

from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args

from transport_folder import create_transport, Sampler



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _to_2d_float32(image):
    arr = np.asarray(image)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image after squeeze, got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def evaluate_pair_from_arrays(filename, image1, image2):
    """Compute PSNR/SSIM for a pair of numpy arrays."""
    img1 = _to_2d_float32(image1)
    img2 = _to_2d_float32(image2)

    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_AREA)

    combined_min = float(min(np.min(img1), np.min(img2)))
    combined_max = float(max(np.max(img1), np.max(img2)))
    data_range = combined_max - combined_min

    if data_range == 0:
        psnr_value = float("inf") if np.array_equal(img1, img2) else 0.0
        ssim_value = 1.0 if np.array_equal(img1, img2) else 0.0
    else:
        psnr_value = peak_signal_noise_ratio(img1, img2, data_range=data_range)
        ssim_value = ssim(img1, img2, data_range=data_range)

    return {
        "filename": filename,
        "psnr": float(psnr_value),
        "ssim": float(ssim_value),
    }


def write_evaluation_report(metrics, output_file_path):
    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    psnr_values = [m["psnr"] for m in metrics if np.isfinite(m["psnr"])]
    ssim_values = [m["ssim"] for m in metrics]

    with output_file_path.open("w", encoding="utf-8") as f_out:
        f_out.write("Filename\tPSNR\tSSIM\n")
        print("Filename\tPSNR\tSSIM")
        for item in metrics:
            psnr_text = "inf" if not np.isfinite(item["psnr"]) else f"{item['psnr']:.2f}"
            line = f"{item['filename']}\t{psnr_text}\t{item['ssim']:.4f}"
            print(line)
            f_out.write(line + "\n")

        if psnr_values and ssim_values:
            mean_psnr = float(np.mean(psnr_values))
            var_psnr = float(np.var(psnr_values))
            mean_ssim = float(np.mean(ssim_values))
            var_ssim = float(np.var(ssim_values))

            summary_header = "\n--- Summary Statistics ---"
            summary_psnr = f"Mean PSNR: {mean_psnr:.2f}, Variance PSNR: {var_psnr:.2f}"
            summary_ssim = f"Mean SSIM: {mean_ssim:.4f}, Variance SSIM: {var_ssim:.4f}"
            print(summary_header)
            print(summary_psnr)
            print(summary_ssim)
            f_out.write(summary_header + "\n")
            f_out.write(summary_psnr + "\n")
            f_out.write(summary_ssim + "\n")
        else:
            message = "\nNo valid data to calculate mean and variance."
            print(message)
            f_out.write(message + "\n")

    print(f"\nEvaluation results saved to {output_file_path}")


def main(mode, args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download."
        learn_sigma = args.image_size == 256
    else:
        learn_sigma = False

    latent_size = args.image_size
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)

    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )
    sampler = Sampler(transport)

    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse,
            )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            num_steps=args.num_sampling_steps,
            use_cg=args.use_cg,
            use_asd_pocs=args.use_asd_pocs,
            cg_inner=args.cg_inner,
            asd_pocs_iters=args.asd_pocs_iters,
            asd_pocs_subsets=args.asd_pocs_subsets,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    class_labels = [args.class_label]
    y = torch.tensor(class_labels, device=device)

    if mode == "SDE":
        if not args.input_dir:
            raise ValueError("For SDE reconstruction, --input-dir must be provided.")
        if  int(args.use_cg) == 0 and int(args.use_asd_pocs) == 0:
            print("Warning: --use-dc=1 but both --use-cg and --use-asd-pocs are 0; data consistency will be skipped.")

        model_kwargs = dict(y=y)
        if args.cfg_scale > 1.0:
            model_kwargs["cfg_scale"] = args.cfg_scale
            model_kwargs["null_class"] = args.null_class if args.null_class is not None else args.num_classes

        eval_fn = evaluate_pair_from_arrays if args.eval else None
        eval_output = args.eval_output
        if args.eval and eval_output is None:
            out_dir = Path(args.output_dir) if args.output_dir else Path(args.input_dir) / "recon"
            eval_output = out_dir / "comparison_results.txt"

        init = torch.randn(len(class_labels), 1, latent_size, latent_size, device=device)
        start_time = time()
        results = sample_fn(
            init,
            model.forward,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            ori_dir=args.ori_dir,
            num_angles=args.num_angles,
            image_size=args.image_size,
            save_npy=args.save_npy,
            allowed_suffixes=args.input_suffixes,
            eval_fn=eval_fn,
            **model_kwargs,
        )
        print(f"Reconstructed {len(results)} files in {time() - start_time:.2f} seconds.")

        if args.eval:
            metrics = [item["metrics"] for item in results if item.get("metrics") is not None]
            if metrics:
                write_evaluation_report(metrics, eval_output)
            else:
                print("--eval is enabled, but no valid metrics were produced.")
        return results

    z = torch.randn(len(class_labels), 1, latent_size, latent_size, device=device)
    use_cfg = args.cfg_scale > 1.0
    if use_cfg:
        null_class = args.null_class if args.null_class is not None else args.num_classes
        y_null = torch.tensor([null_class] * len(class_labels), device=device)
        ys = torch.cat([y, y_null], 0)
        zs = torch.cat([z, z], 0)
        model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = model.forward_with_cfg
        z = zs
    else:
        model_kwargs = dict(y=y)
        model_fn = model.forward

    start_time = time()
    samples = sample_fn(z, model_fn, **model_kwargs)[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve")

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)

    mode = sys.argv[1]
    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=r"LOCATION/OF/YOUR/CHECKPOINT.pt",
        help="Optional path to a SiT checkpoint.",
    )

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)

    parser.add_argument(
        "--use-dc",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set to 1 to enable data consistency during reconstruction; 0 disables it.",
    )
    parser.add_argument(
        "--use-cg",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set to 1 to enable the CG data consistency update; 0 disables CG.",
    )
    parser.add_argument(
        "--use-asd-pocs",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set to 1 to enable the ASD-POCS data consistency update; 0 disables ASD-POCS.",
    )
    parser.add_argument(
        "--cg-inner",
        type=int,
        default=10,
        help="Number of inner CG iterations when CG data consistency is enabled.",
    )
    parser.add_argument(
        "--asd-pocs-iters",
        type=int,
        default=10,
        help="Number of ASD-POCS outer iterations when ASD-POCS is enabled.",
    )
    parser.add_argument(
        "--asd-pocs-subsets",
        type=int,
        default=5,
        help="Number of ASD-POCS subsets when ASD-POCS is enabled.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"LOCATION/OF/YOUR/INPUT/DIRECTORY",
        help="Folder that contains the input images to reconstruct (.mat/.npy/.png/.jpg by default).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"LOCATION/OF/YOUR/OUTPUT/DIRECTORY",
        help="Folder to save reconstructed images. Defaults to <input-dir>/recon if omitted.",
    )
    parser.add_argument(
        "--ori-dir",
        type=str,
        default=r"LOCATION/OF/YOUR/ORIGINAL/DIRECTORY",
        help="Optional folder to save the normalized original inputs as PNG.",
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=55,
        help="Number of sparse-view projection angles used by LEAP CT.",
    )
    parser.add_argument(
        "--save-npy",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set to 1 to additionally save the final reconstruction as .npy.",
    )
    parser.add_argument(
        "--input-suffixes",
        type=str,
        default=".mat,.npy,.png,.jpg,.jpeg,.bmp,.tif,.tiff",
        help="Comma-separated list of input suffixes to reconstruct.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate each reconstruction with PSNR/SSIM using the original input array and save a text report.",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Path to the evaluation txt file. Defaults to <output-dir>/comparison_results.txt.",
    )
    parser.add_argument(
        "--class-label",
        type=int,
        default=0,
        help="Conditional class label used by the SiT model.",
    )
    parser.add_argument(
        "--null-class",
        type=int,
        default=None,
        help="Null/unconditional class label used by CFG. Defaults to --num-classes.",
    )

    args = parser.parse_known_args()[0]
    main(mode, args)
