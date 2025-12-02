#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_folder_stats(folder):
    """
    Load all .pkl files in `folder` and extract:
      - betas: shape (n_frames, n_betas)
      - transl_norms: shape (n_frames,)
    """
    folder = Path(folder)
    pkl_files = sorted(glob.glob(str(folder / "*.pkl")))

    if not pkl_files:
        print(f"[WARN] No .pkl files found in {folder}")
        return None, None

    betas_list = []
    transl_norms = []

    for f in pkl_files:
        arr = np.load(f, allow_pickle=True)
        # sometimes np.load with allow_pickle returns a 0-d array holding a dict
        if isinstance(arr, np.ndarray) and arr.shape == ():
            arr = arr.item()

        if "betas" not in arr or "transl" not in arr:
            print(f"[WARN] File {f} missing 'betas' or 'transl'; skipping.")
            continue

        betas = np.asarray(arr["betas"]).reshape(-1)  # (n_betas,)
        transl = np.asarray(arr["transl"]).reshape(-1)  # (3,) expected
        transl_norm = float(np.linalg.norm(transl))

        betas_list.append(betas)
        transl_norms.append(transl_norm)

    if not betas_list:
        print(f"[WARN] No valid SMPLX data found in {folder}")
        return None, None

    betas_arr = np.vstack(betas_list)          # (n_frames, n_betas)
    transl_norms_arr = np.asarray(transl_norms)  # (n_frames,)

    return betas_arr, transl_norms_arr


def main():
    parser = argparse.ArgumentParser(
        description="Create boxplots of SMPLX betas and translation norms "
                    "for multiple folders of .pkl files."
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="List of folders, each containing SMPLX .pkl files."
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="List of names corresponding to each folder (same order)."
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory to save the plots."
    )
    args = parser.parse_args()

    if len(args.folders) != len(args.names):
        raise ValueError("The number of folders must match the number of names.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all stats
    all_betas = []
    all_transl_norms = []
    valid_names = []

    for folder, name in zip(args.folders, args.names):
        print(f"[INFO] Processing folder: {folder} (label: {name})")
        betas_arr, transl_norms_arr = load_folder_stats(folder)
        if betas_arr is None:
            print(f"[WARN] Skipping folder {folder} (no usable data).")
            continue

        all_betas.append(betas_arr)
        all_transl_norms.append(transl_norms_arr)
        valid_names.append(name)

    if not all_betas:
        print("[ERROR] No valid data loaded from any folder. Exiting.")
        return

    n_folders = len(all_betas)
    n_betas = all_betas[0].shape[1]

    # ------------------------
    # Boxplots for betas (all folders on one plot)
    # ------------------------
    fig, ax = plt.subplots(figsize=(min(10, 2 * n_betas), 6))

    group_width = 0.8
    box_width = group_width / n_folders

    data = []
    positions = []
    beta_indices = np.arange(n_betas)

    for b_idx in range(n_betas):
        base_x = beta_indices[b_idx]
        for f_idx in range(n_folders):
            betas_arr = all_betas[f_idx]
            data.append(betas_arr[:, b_idx])

            offset = (f_idx - (n_folders - 1) / 2) * box_width
            positions.append(base_x + offset)

    bplot = ax.boxplot(
        data,
        positions=positions,
        widths=box_width * 0.9,
        patch_artist=True,
        manage_ticks=False,
    )

    cmap = plt.get_cmap("tab10")
    for k, box in enumerate(bplot["boxes"]):
        f_idx = k % n_folders
        box.set_facecolor(cmap(f_idx))
        box.set_alpha(0.7)

    legend_patches = [
        mpatches.Patch(color=cmap(i), label=valid_names[i])
        for i in range(n_folders)
    ]
    ax.legend(handles=legend_patches, title="SMPLX model", loc="upper right")

    ax.set_xlabel("beta index")
    ax.set_ylabel("beta value")
    ax.set_title("Distribution of SMPLX betas per SMPLX model")
    ax.set_xticks(beta_indices)
    ax.set_xticklabels(beta_indices, rotation=45, ha="right")

    fig.tight_layout()
    out_path_betas = os.path.join(args.out_dir, "boxplot_betas_all_folders.png")
    fig.savefig(out_path_betas, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved betas boxplot to {out_path_betas}")

    # ------------------------
    # Boxplots for translation norms
    # ------------------------
    fig2, ax2 = plt.subplots(figsize=(max(6, 2 * n_folders), 4))

    ax2.boxplot(all_transl_norms, labels=valid_names)
    ax2.set_ylabel("Translation L2 norm [m]")
    ax2.set_xlabel("SMPLX model")
    ax2.set_title("Distribution of translation norm per SMPLX model")
    fig2.tight_layout()

    out_path_transl = os.path.join(args.out_dir, "boxplot_transl_norm_per_folder.png")
    fig2.savefig(out_path_transl, dpi=200)
    plt.close(fig2)
    print(f"[INFO] Saved translation norm boxplot to {out_path_transl}")


if __name__ == "__main__":
    main()
