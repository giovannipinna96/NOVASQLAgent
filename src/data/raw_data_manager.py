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
import logging
import os
import zipfile
import requests # For direct downloads
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import shutil


# Third-party libraries
try:
    from tqdm.auto import tqdm # Progress tracking
except ImportError:
    print("tqdm library not found. Please install it: pip install tqdm")
    # Basic fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs): # type: ignore
        return iterable

try:
    from datasets import load_dataset, Dataset, DatasetDict # HuggingFace datasets
except ImportError:
    print("HuggingFace datasets library not found. Please install it: pip install datasets")
    # Define dummy classes if not installed, so the rest of the file can be parsed.
    class Dataset: pass # type: ignore
    class DatasetDict: pass # type: ignore
    def load_dataset(*args, **kwargs): # type: ignore
        raise NotImplementedError("HuggingFace datasets library is not installed.")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory for all raw data
# Assuming this script is in NOVASQLAgent/src/data/
# Then PROJECT_ROOT is NOVASQLAgent/
# And BASE_DATA_DIR will be NOVASQLAgent/data/raw_data/
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError: # __file__ is not defined if running in some interactive environments
    PROJECT_ROOT = Path(".").resolve() # Fallback to current working directory

BASE_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"


# --- Helper Functions ---
def _download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Downloads a file from a URL to a destination path with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
        total_size = int(response.headers.get('content-length', 0))

        dest_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists

        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk: # filter out keep-alive new chunks
                    size = f.write(chunk)
                    bar.update(size)
        logger.info(f"Successfully downloaded {url} to {dest_path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        if dest_path.exists(): # Clean up partial download
            dest_path.unlink()
        raise
    except IOError as e:
        logger.error(f"Error writing file {dest_path}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise


def _extract_zip(zip_path: Path, extract_to: Path, delete_zip: bool = True) -> None:
    """Extracts a zip file and optionally deletes it."""
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc=f"Extracting {zip_path.name}"):
                try:
                    zip_ref.extract(member, extract_to)
                except zipfile.error as e: # Catch issues with specific files in zip
                    logger.error(f"Error extracting {member.filename} from {zip_path.name}: {e}")

        logger.info(f"Successfully extracted {zip_path} to {extract_to}")
        if delete_zip:
            zip_path.unlink()
            logger.info(f"Deleted zip file: {zip_path}")
    except zipfile.BadZipFile:
        logger.error(f"Error: {zip_path} is not a valid zip file or is corrupted.")
        # Don't delete if it's a bad zip, user might want to inspect it
        raise
    except Exception as e:
        logger.error(f"An error occurred during zip extraction of {zip_path}: {e}")
        raise


def _save_hf_dataset_split(dataset: Dataset, path: Path, split_name: str) -> None:
    """Saves a HuggingFace dataset split to disk in JSON format (common for text data)."""
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{split_name}.jsonl" # Using .jsonl for line-delimited JSON
    try:
        # datasets.save_to_disk is for the entire Dataset object, not individual files usually.
        # To save as json, we can iterate or use .to_json()
        dataset.to_json(file_path, orient="records", lines=True, force_ascii=False)
        logger.info(f"Saved {split_name} split of dataset to {file_path}")
    except Exception as e:
        logger.error(f"Error saving dataset split {split_name} to {file_path}: {e}")
        raise


# --- Dataset Specific Downloaders ---

class ArcherBenchDownloader:
    """Handles downloading and organizing the Archer-Bench dataset."""
    NAME = "archer_bench"
    BASE_URL = "https://raw.githubusercontent.com/SIGKDD/SPIDER/master/spider_databases/" # Placeholder, Archer actual URLs needed
    # Actual URLs for Archer files (these are illustrative, replace with real ones)
    # Typically, Archer files might be part of a larger benchmark suite or specific research page.
    # The provided URL "https://sig4kg.github.io/archer-bench/" is a project page, not direct download links.
    # We need to find the actual file URLs. Assuming they are like:
    # For this example, let's assume some dummy URLs as actual Archer files are not directly linked here.
    # If they are inside a zip, the logic would be: download zip, then extract.
    # For now, let's assume direct file links for train/test JSONs.
    # If the files are SQL dumps and JSONs, that's common.
    # Let's assume the Archer data comes as individual JSON files for train/test.
    TRAIN_URL = "https://example.com/archer_train_english.json" # Replace with actual URL
    TEST_URL = "https://example.com/archer_test_english.json"   # Replace with actual URL
    # If they are within a zip:
    # ARCHER_ZIP_URL = "https://example.com/archer_bench_english.zip"

    def __init__(self, root_dir: Path):
        self.dataset_dir = root_dir / self.NAME
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"

    def download(self, download_train: bool = True, download_test: bool = True) -> None:
        logger.info(f"Starting download for {self.NAME}...")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # This is highly dependent on how Archer-Bench distributes its files.
        # The task description says "Files are hosted as direct links".
        # We'll proceed assuming direct JSON links. If it's a zip, logic changes.

        if download_train:
            self.train_dir.mkdir(parents=True, exist_ok=True)
            train_file_path = self.train_dir / "archer_train.json" # Assuming JSON format
            logger.warning(f"Archer-Bench train URL is a placeholder: {self.TRAIN_URL}. Download will likely fail.")
            try:
                _download_file(self.TRAIN_URL, train_file_path)
            except Exception as e:
                logger.error(f"Failed to download Archer-Bench train data: {e}. Please check the URL or manually place the file.")

        if download_test:
            self.test_dir.mkdir(parents=True, exist_ok=True)
            test_file_path = self.test_dir / "archer_test.json"
            logger.warning(f"Archer-Bench test URL is a placeholder: {self.TEST_URL}. Download will likely fail.")
            try:
                _download_file(self.TEST_URL, test_file_path)
            except Exception as e:
                logger.error(f"Failed to download Archer-Bench test data: {e}. Please check the URL or manually place the file.")

        logger.info(f"{self.NAME} download process finished.")


class BirdBenchDownloader:
    """Handles downloading and organizing the BIRD benchmark dataset."""
    NAME = "bird"
    # BIRD benchmark page: https://bird-bench.github.io/
    # Usually, datasets like BIRD are hosted on GitHub releases, Zenodo, or similar.
    # Let's assume BIRD provides a ZIP file for its train and dev sets.
    # Example (hypothetical URL for the ZIP file):
    BIRD_ZIP_URL = "https://bird-bench.github.io/assets/BIRD.zip" # From the website's download button link
    # Inside the ZIP, we expect to find train and dev folders or files.

    def __init__(self, root_dir: Path):
        self.dataset_dir = root_dir / self.NAME
        # Expected structure after extraction:
        # bird/
        #   train/
        #     train.json (or similar)
        #     database/ (SQL files)
        #   dev/
        #     dev.json
        #     database/
        self.temp_zip_path = self.dataset_dir / "BIRD_temp.zip"

    def download(self, download_train: bool = True, download_dev: bool = True) -> None:
        logger.info(f"Starting download for {self.NAME}...")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Download the main zip file
        try:
            _download_file(self.BIRD_ZIP_URL, self.temp_zip_path)
        except Exception as e:
            logger.error(f"Failed to download BIRD main zip file from {self.BIRD_ZIP_URL}: {e}")
            return

        # Extract the zip file
        # The extraction path should place contents directly into self.dataset_dir
        # e.g., if BIRD.zip contains a top-level "BIRD/" folder, extract to self.dataset_dir.parent
        # or extract to self.dataset_dir and then move contents if needed.
        # For now, assume extraction directly into self.dataset_dir is fine.
        try:
            _extract_zip(self.temp_zip_path, self.dataset_dir, delete_zip=True)
            logger.info(f"BIRD dataset extracted to {self.dataset_dir}")
        except Exception as e:
            logger.error(f"Failed to extract BIRD zip file: {e}")
            return

        # Post-extraction: Verify expected train/dev folders.
        # The BIRD zip structure from the website is:
        # BIRD.zip -> BIRD/
        #               dev.json
        #               train.json
        #               dev_databases/
        #               train_databases/
        #               ... (other files like LICENSE, README)
        # So, after extracting to self.dataset_dir, we'll have self.dataset_dir/BIRD/...
        # We need to move these contents up or adjust paths.

        extracted_bird_folder = self.dataset_dir / "BIRD" # Common pattern for zips
        if extracted_bird_folder.exists() and extracted_bird_folder.is_dir():
            logger.info(f"Found BIRD contents in subfolder: {extracted_bird_folder}. Moving contents up.")
            # Move all contents from extracted_bird_folder to self.dataset_dir
            for item in extracted_bird_folder.iterdir():
                shutil.move(str(item), str(self.dataset_dir / item.name))
            shutil.rmtree(extracted_bird_folder) # Remove the now-empty BIRD subfolder
            logger.info(f"Moved BIRD contents to {self.dataset_dir}")

        # Now, create the target train/dev structure based on the problem description
        # data/raw_data/bird/train/ and data/raw_data/bird/dev/

        train_output_dir = self.dataset_dir / "train"
        dev_output_dir = self.dataset_dir / "dev"

        # Files we expect from the BIRD zip (now in self.dataset_dir)
        source_train_json = self.dataset_dir / "train.json"
        source_train_databases = self.dataset_dir / "train_databases"
        source_dev_json = self.dataset_dir / "dev.json"
        source_dev_databases = self.dataset_dir / "dev_databases"

        if download_train:
            train_output_dir.mkdir(parents=True, exist_ok=True)
            if source_train_json.exists():
                shutil.move(str(source_train_json), str(train_output_dir / "train.json"))
            else:
                logger.warning(f"train.json not found in {self.dataset_dir}")
            if source_train_databases.exists() and source_train_databases.is_dir():
                shutil.move(str(source_train_databases), str(train_output_dir / "databases"))
            else:
                logger.warning(f"train_databases/ not found in {self.dataset_dir}")
            logger.info(f"Organized BIRD train data into {train_output_dir}")

        if download_dev:
            dev_output_dir.mkdir(parents=True, exist_ok=True)
            if source_dev_json.exists():
                shutil.move(str(source_dev_json), str(dev_output_dir / "dev.json"))
            else:
                logger.warning(f"dev.json not found in {self.dataset_dir}")
            if source_dev_databases.exists() and source_dev_databases.is_dir():
                 shutil.move(str(source_dev_databases), str(dev_output_dir / "databases"))
            else:
                logger.warning(f"dev_databases/ not found in {self.dataset_dir}")
            logger.info(f"Organized BIRD dev data into {dev_output_dir}")

        # Clean up any other files from the root of self.dataset_dir if desired (e.g. LICENSE, README)
        # For now, leave them.

        logger.info(f"{self.NAME} download and organization process finished.")


class SynSQLDownloader:
    """Handles downloading and organizing the SynSQL-2.5M dataset from HuggingFace."""
    NAME = "synsql_2.5m"
    HF_DATASET_NAME = "seeklhy/SynSQL-2.5M"

    def __init__(self, root_dir: Path):
        self.dataset_dir = root_dir / self.NAME
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test" # SynSQL might only have a train split

    def download(self, download_train: bool = True, download_test: bool = True) -> None:
        logger.info(f"Starting download for {self.NAME} from HuggingFace...")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load the dataset from HuggingFace Hub
            # This will download and cache it locally by the `datasets` library.
            # We then need to save it to our target structure.
            logger.info(f"Loading dataset '{self.HF_DATASET_NAME}' from HuggingFace Hub. This may take a while...")
            dataset_dict = load_dataset(self.HF_DATASET_NAME)

            if not isinstance(dataset_dict, DatasetDict):
                logger.error(f"Expected a DatasetDict from load_dataset, got {type(dataset_dict)}. Cannot process.")
                return

            logger.info(f"Dataset '{self.HF_DATASET_NAME}' loaded. Available splits: {list(dataset_dict.keys())}")

            # SynSQL-2.5M seems to only have a 'train' split on HuggingFace.
            # We need to handle this: save 'train' to train_dir.
            # If a 'test' split is required but not available, we might need to split 'train' or log a warning.

            if download_train and 'train' in dataset_dict:
                _save_hf_dataset_split(dataset_dict['train'], self.train_dir, "train")
            elif download_train:
                logger.warning(f"Train split requested but not found in '{self.HF_DATASET_NAME}'.")

            # Handle test split:
            if download_test:
                if 'test' in dataset_dict:
                    _save_hf_dataset_split(dataset_dict['test'], self.test_dir, "test")
                else:
                    logger.warning(f"Test split requested but not found in '{self.HF_DATASET_NAME}'. No test data saved for {self.NAME}.")
                    # Optionally, one could create a test split from train here if needed.
                    # For now, just warn.

        except Exception as e:
            logger.error(f"Error downloading or processing {self.HF_DATASET_NAME} from HuggingFace: {e}", exc_info=True)

        logger.info(f"{self.NAME} download process finished.")


class RawDataManager:
    """Main class to manage downloading and organizing all datasets."""

    def __init__(self, base_data_path: Union[str, Path] = BASE_DATA_DIR):
        self.root_dir = Path(base_data_path)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"RawDataManager initialized. Data will be stored in: {self.root_dir}")

        self.downloaders = {
            "archer_bench": ArcherBenchDownloader(self.root_dir),
            "bird": BirdBenchDownloader(self.root_dir),
            "synsql_2.5m": SynSQLDownloader(self.root_dir),
        }

    def download_all(
        self,
        download_archer: bool = True,
        download_bird: bool = True,
        download_synsql: bool = True,
        # Split controls (can be more granular per dataset if needed)
        download_train_splits: bool = True,
        download_test_splits: bool = True, # General 'test' or 'dev'
        download_dev_splits: bool = True,   # Explicitly for 'dev' like in BIRD
    ) -> None:
        """
        Downloads all specified datasets and their splits.

        Args:
            download_archer: Whether to download Archer-Bench.
            download_bird: Whether to download BIRD benchmark.
            download_synsql: Whether to download SynSQL-2.5M.
            download_train_splits: Global flag for downloading 'train' splits.
            download_test_splits: Global flag for downloading 'test' splits.
            download_dev_splits: Global flag for downloading 'dev' splits (applies to BIRD).
        """
        logger.info("Starting all dataset downloads...")

        if download_archer:
            try:
                self.downloaders["archer_bench"].download(
                    download_train=download_train_splits,
                    download_test=download_test_splits
                )
            except Exception as e:
                logger.error(f"Error during Archer-Bench download: {e}", exc_info=True)

        if download_bird:
            try:
                # BIRD uses 'dev' instead of 'test' for one of its splits.
                self.downloaders["bird"].download(
                    download_train=download_train_splits,
                    download_dev=download_dev_splits # BIRD has train/dev
                )
            except Exception as e:
                logger.error(f"Error during BIRD benchmark download: {e}", exc_info=True)

        if download_synsql:
            try:
                self.downloaders["synsql_2.5m"].download(
                    download_train=download_train_splits,
                    download_test=download_test_splits # SynSQL might not have a test split
                )
            except Exception as e:
                logger.error(f"Error during SynSQL-2.5M download: {e}", exc_info=True)

        logger.info("All dataset download processes finished.")

    def list_downloaded_datasets(self) -> Dict[str, List[str]]:
        """Lists downloaded datasets and their subdirectories (splits/files)."""
        downloaded: Dict[str, List[str]] = {}
        if not self.root_dir.exists():
            return downloaded

        for dataset_name in self.downloaders.keys():
            dataset_path = self.root_dir / dataset_name
            if dataset_path.exists() and dataset_path.is_dir():
                contents = [item.name for item in dataset_path.iterdir()]
                if contents:
                    downloaded[dataset_name] = contents
        return downloaded


if __name__ == "__main__":
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Base Data Directory: {BASE_DATA_DIR}")

    manager = RawDataManager()

    # Example: Download specific datasets
    # Set flags to False for datasets you don't want to download during testing.
    # Archer-Bench URLs are placeholders, so it will likely log errors for that.
    # BIRD download should work if the URL is still valid.
    # SynSQL download should work.

    # Before running, ensure you have `requests`, `tqdm`, and `datasets` installed.
    # `pip install requests tqdm datasets`

    # Create dummy files for Archer-Bench to simulate successful download for testing flow
    # This part is only for local testing if actual Archer URLs are not working.
    # In a real run, these dummy creations should be removed.
    enable_archer_dummies = True # Set to True to create dummy Archer files for testing
    if enable_archer_dummies:
        logger.warning("Creating DUMMY Archer-Bench files for testing flow.")
        dummy_archer_dir = BASE_DATA_DIR / "archer_bench"
        (dummy_archer_dir / "train").mkdir(parents=True, exist_ok=True)
        (dummy_archer_dir / "test").mkdir(parents=True, exist_ok=True)
        dummy_train_file = dummy_archer_dir / "train" / "archer_train.json"
        dummy_test_file = dummy_archer_dir / "test" / "archer_test.json"
        with open(dummy_train_file, "w") as f:
            f.write('{"dummy_train_data": true}')
        with open(dummy_test_file, "w") as f:
            f.write('{"dummy_test_data": true}')
        # Replace downloader URLs with local file paths for dummy test
        # Ensure these are file URLs for requests to handle them if _download_file expects http/https
        # Forcing file:// scheme
        ArcherBenchDownloader.TRAIN_URL = f"file://{dummy_train_file.resolve()}"
        ArcherBenchDownloader.TEST_URL = f"file://{dummy_test_file.resolve()}"


    logger.info("Starting selective dataset download...")
    manager.download_all(
        download_archer=enable_archer_dummies,  # Only try archer if dummies are enabled for this test
        download_bird=True, # Focus on BIRD
        download_synsql=False, # Disable SynSQL for quicker test
        download_train_splits=False, # Disable BIRD train split
        download_test_splits=enable_archer_dummies, # For Archer dummy test only
        download_dev_splits=True    # Focus on BIRD dev split
    )

    logger.info("\n--- Downloaded Datasets ---")
    downloaded_info = manager.list_downloaded_datasets()
    if downloaded_info:
        for name, contents in downloaded_info.items():
            logger.info(f"Dataset: {name}")
            for item_name in contents:
                logger.info(f"  - {item_name}")
    else:
        logger.info("No datasets found in the raw_data directory.")

    logger.info("\nRaw Data Manager example usage finished.")
    logger.info(f"Please check the directory: {BASE_DATA_DIR}")

    # To clean up dummy Archer files if created:
    if enable_archer_dummies and (BASE_DATA_DIR / "archer_bench").exists():
         logger.warning("Removing DUMMY Archer-Bench files.")
         shutil.rmtree(BASE_DATA_DIR / "archer_bench")
