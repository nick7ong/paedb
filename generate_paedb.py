import argparse
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from algorithms.elae import extract_ambience


def collect_wav_files(dirs):
    wav_paths = []
    for d in dirs:
        d = Path(d).expanduser().resolve()
        if not d.exists():
            print(f"[Warning] Directory does not exist: {d}")
            continue
        for path in d.rglob("*.wav"):
            wav_paths.append(str(path))
    return wav_paths


def build_output_path(wav_path, output_root, split):
    """Constructs the PAEDB output directory for a given wav file."""
    wav_path = Path(wav_path)
    output_root = Path(output_root)
    song_title = wav_path.parent.name
    stem_name = wav_path.stem

    name = f"{song_title}_{stem_name}"
    name = re.sub(r'_+', '_', re.sub(r'\s+', '', name.lower().replace('-', '_')))

    out_dir = os.path.join(output_root, split, name)
    return out_dir


def split_dataset(wav_files, ratios, seed=42):
    """Randomly split wav_files into train/valid/test lists using ratios."""
    assert len(ratios) == 3, "Ratios must be: train valid test"
    train_r, valid_r, test_r = ratios
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)
    wav_files = wav_files[:]  # shallow copy
    random.shuffle(wav_files)

    n = len(wav_files)
    n_train = int(train_r * n)
    n_valid = int(valid_r * n)
    # test = remaining

    train_list = wav_files[:n_train]
    valid_list = wav_files[n_train:n_train + n_valid]
    test_list = wav_files[n_train + n_valid:]

    return {
        "train": train_list,
        "valid": valid_list,
        "test": test_list,
    }


def process(wav_path, split_name, output_root):
    """Worker function executed by each thread."""
    output_dir = build_output_path(wav_path, output_root, split_name)
    os.makedirs(output_dir, exist_ok=True)

    input_stereo, fs = sf.read(wav_path)
    sf.write(os.path.join(output_dir, "mixture.wav"), input_stereo, fs)

    extract_ambience(
        input_stereo=input_stereo,
        window_size=1024,
        overlap=2,
        fs=fs,
        output_path=output_dir,
    )

    return wav_path


def main(dirs, output_root, pae_type, ratios, max_workers=4):
    wav_files = collect_wav_files(dirs)
    os.makedirs(output_root, exist_ok=True)

    print("----------------------------------------")
    print(f"Found {len(wav_files)} WAV files.")
    print("----------------------------------------")

    splits = split_dataset(wav_files, ratios)

    print("========================================")
    print("     Generating PAE-DB                  ")
    print("========================================")
    print(f"PAE Type: {pae_type}")
    print(f"Directories: {dirs}")
    print(f"Output Directory: {output_root}")
    print(f"Num Workers: {max_workers}")
    print(f"Split summary (train={ratios[0]}, valid={ratios[1]}, test={ratios[2]}):")
    print(f"  Train: {len(splits['train'])} files")
    print(f"  Valid: {len(splits['valid'])} files")
    print(f"  Test:  {len(splits['test'])} files")
    print("----------------------------------------")

    for split_name, paths in splits.items():
        print(f"\n=== Processing {split_name} ({len(paths)} files) ===")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process, wav_path, split_name, output_root)
                for wav_path in paths
            ]

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split_name}", unit="file"):
                f.result()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PAE dataset")
    parser.add_argument("--dirs", nargs="+", required=True, help="List of directories to WAV files.")
    parser.add_argument("--output", type=str, default="dataset/paedb", help="Output directory.")
    parser.add_argument("--type", type=str, default="elae", help="PAE algorithm type.")
    parser.add_argument("--split-ratios", nargs=3, type=float, default=[0.9, 0.1, 0.0],
                        help="Ratios for train valid test (must sum to 1).")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker threads.")

    args = parser.parse_args()

    main(args.dirs, args.output, args.type, args.split_ratios, args.workers)
