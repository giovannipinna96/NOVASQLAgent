"""
This file is responsible for downloading and organizing datasets required for training, evaluating,
and fine-tuning LLMs. The datasets include both structured benchmarks and synthetic corpora, and
must be stored in a clean, hierarchical directory format. The system should also include features
for progress tracking, selective downloads, and automated extraction of compressed files.

DATA SOURCES TO DOWNLOAD:

1. ARCHER-BENCH DATA:
    - Source URL: https://sig4kg.github.io/archer-bench/
    - Download both the train and test datasets in English.
    - Files are hosted as direct links; implement scraping or static URL-based retrieval if needed.

2. BIRD BENCHMARK DATA:
    - Source URL: https://bird-bench.github.io/
    - Download both train and dev datasets.
    - Format and access mechanism may differ; implement handlers accordingly.

3. SYNTHETIC SQL DATA (SynSQL-2.5M):
    - Hugging Face Dataset Hub: https://huggingface.co/datasets/seeklhy/SynSQL-2.5M
    - Dataset name for loading via `datasets` library: "seeklhy/SynSQL-2.5M"
    - Ensure correct split loading (e.g., train/test) and save locally in the standardized structure.

DIRECTORY STRUCTURE & FILE MANAGEMENT:

- All downloaded and extracted datasets should be stored under a clearly defined hierarchical structure:
    Example:
    NOVASQLAgent/
     └── 
        data/
        └── raw_data/
            ├── archer_bench/
            │   ├── train/
            │   └── test/
            ├── bird/
            │   ├── train/
            │   └── dev/
            └── synsql_2.5m/
                ├── train/
                └── test/
- Automatically create all necessary directories if they do not exist.
- Include utilities to navigate and validate directory structures.

ZIP FILE HANDLING:

- If downloaded datasets are in `.zip` or other compressed formats:
    • Automatically detect compressed files.
    • Extract the contents into the appropriate destination folders.
    • Optionally delete the original compressed file after successful extraction.

DOWNLOAD CONTROL & PROGRESS TRACKING:

- Use the `tqdm` library to track download progress and provide visual feedback.
- Allow fine-grained control over which datasets and which splits to download via boolean flags.
    • Example: `download_archer = True`, `download_bird = False`, `download_synsql = True`
    • And: `download_train = True`, `download_test = False`, `download_dev = True`

PYTHON BEST PRACTICES:

- Adhere to all standard Python conventions and best practices:
    • Full use of type annotations for clarity and safety.
    • Descriptive variable names and method-level docstrings.
    • Error handling with informative logging and exception messages.
    • Modular code with reusable functions and minimal repetition.
    • PEP8 compliance and well-organized logic blocks.
    • Use `pathlib` for OS-independent file path handling where appropriate.

This module plays a foundational role in ensuring that all training and evaluation data is
consistently downloaded, stored, and organized, enabling seamless access and management for
subsequent stages of model development.
"""


import os
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import requests
import zipfile
import argparse

logging.basicConfig(level=logging.INFO)

DEFAULT_DATA_ROOT = Path(__file__).parent.parent / "data" / "raw_data"

class RawDataManager:
    """
    Manages downloading, organizing, and extracting datasets for LLM training and evaluation.
    """
    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = data_root or DEFAULT_DATA_ROOT
        self.archer_dir = self.data_root / "archer_bench"
        self.bird_dir = self.data_root / "bird"
        self.synsql_dir = self.data_root / "synsql_2.5m"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create all required directories if they do not exist."""
        for d in [
            self.archer_dir / "train",
            self.archer_dir / "test",
            self.bird_dir / "train",
            self.bird_dir / "dev",
            self.synsql_dir / "train",
            self.synsql_dir / "test",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, dest: Path) -> None:
        """Download a file from a URL with progress bar."""
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(dest, 'wb') as file, tqdm(
            desc=f"Downloading {dest.name}",
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    def extract_zip(self, zip_path: Path, extract_to: Path, delete_zip: bool = True) -> None:
        """Extract a zip file to a directory and optionally delete the zip."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        if delete_zip:
            zip_path.unlink()

    def download_archer(self, download_train: bool = True, download_test: bool = True) -> None:
        """
        Download Archer-Bench English dataset and extract train/test splits.
        The zip contains both train and test data inside.
        """
        url = "https://sig4kg.github.io/archer-bench/dataset/en_data.zip"
        dest = self.archer_dir / "en_data.zip"
        self.download_file(url, dest)
        self.extract_zip(dest, self.archer_dir)
        # Move extracted files to train/test folders if needed
        # Assumes extracted files are named 'train.json' and 'test.json'
        train_src = self.archer_dir / "train.json"
        test_src = self.archer_dir / "test.json"
        if download_train and train_src.exists():
            train_dst = self.archer_dir / "train" / "train.json"
            train_src.replace(train_dst)
        if download_test and test_src.exists():
            test_dst = self.archer_dir / "test" / "test.json"
            test_src.replace(test_dst)

    def download_bird(self, download_train: bool = True, download_dev: bool = True) -> None:
        """Download Bird Benchmark train and/or dev datasets."""
        # URLs should be updated to actual dataset links
        urls = {
            "train": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
            "dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
        }
        if download_train:
            dest = self.bird_dir / "train" / "train.zip"
            self.download_file(urls["train"], dest)
            self.extract_zip(dest, self.bird_dir / "train")
        if download_dev:
            dest = self.bird_dir / "dev" / "dev.zip"
            self.download_file(urls["dev"], dest)
            self.extract_zip(dest, self.bird_dir / "dev")

    def download_synsql(self, download_train: bool = True) -> None:
        """Download SynSQL-2.5M train and/or test splits using Hugging Face datasets library."""
        try:
            from datasets import load_dataset
        except ImportError:
            logging.error("Please install the 'datasets' library to download SynSQL-2.5M.")
            return
        if download_train:
            ds = load_dataset("seeklhy/OmniSQL-datasets")  # ? is this correct?
            print(f"Loaded {len(ds)} records from SynSQL-2.5M train split. INFO: {ds.info}")
            out_dir = self.synsql_dir / "train"
            out_file = out_dir / f"train.jsonl"
            ds.to_json(str(out_file))

def main(
    download_archer: bool = True,
    download_bird: bool = True,
    download_synsql: bool = True,
    download_train: bool = True,
    download_test: bool = True,
    download_dev: bool = True,
    data_root: Optional[Path] = None
) -> None:
    manager = RawDataManager(data_root=data_root)
    if download_archer:
        manager.download_archer(download_train=download_train, download_test=download_test)
    if download_bird:
        manager.download_bird(download_train=download_train, download_dev=download_dev)
    if download_synsql:
        # manager.download_file(url="https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/blob/main/data.json", dest=Path("/leonardo_work/uTS25_Pinna/phd_proj/SQLAgent/NOVASQLAgent/data/raw_data/synsql_2.5m/train/data.json"))
        # manager.download_file(url="https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/blob/main/tables.json", dest=Path("/leonardo_work/uTS25_Pinna/phd_proj/SQLAgent/NOVASQLAgent/data/raw_data/synsql_2.5m/tables.json"))
        # manager.download_file(url="https://huggingface.co/datasets/seeklhy/SynSQL-2.5M/blob/main/databases.zip", dest=Path("/leonardo_work/uTS25_Pinna/phd_proj/SQLAgent/NOVASQLAgent/data/raw_data/synsql_2.5m/databases.zip"))
        # manager.extract_zip(
            # zip_path=Path("/leonardo_work/uTS25_Pinna/phd_proj/SQLAgent/NOVASQLAgent/data/raw_data/synsql_2.5m/databases.zip"),
            # extract_to=Path("/leonardo_work/uTS25_Pinna/phd_proj/SQLAgent/NOVASQLAgent/data/raw_data/synsql_2.5m/databases/")
        # )
        manager.download_synsql(download_train=download_train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and organize LLM datasets.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory for storing raw datasets (default: %(default)s)"
    )
    parser.add_argument("--download_archer", action="store_true", help="Download Archer-Bench dataset")
    parser.add_argument("--download_bird", action="store_true", help="Download Bird Benchmark dataset")
    parser.add_argument("--download_synsql", action="store_true", help="Download SynSQL-2.5M dataset")
    parser.add_argument("--download_train", action="store_true", help="Download train split")
    parser.add_argument("--download_test", action="store_true", help="Download test split")
    parser.add_argument("--download_dev", action="store_true", help="Download dev split (Bird)")
    args = parser.parse_args()

    # If no split flags are provided, default to all True
    if not (args.download_train or args.download_test or args.download_dev):
        args.download_train = True
        args.download_test = True
        args.download_dev = True

    # If no dataset flags are provided, default to all True
    if not (args.download_archer or args.download_bird or args.download_synsql):
        args.download_archer = False
        args.download_bird = False
        args.download_synsql = True

    main(
        download_archer=args.download_archer,
        download_bird=args.download_bird,
        download_synsql=args.download_synsql,
        download_train=args.download_train,
        download_test=args.download_test,
        download_dev=args.download_dev,
        data_root=Path(args.data_root)
    )
