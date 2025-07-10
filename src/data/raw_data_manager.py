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
