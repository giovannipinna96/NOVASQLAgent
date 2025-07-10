"""
Tests for src/data/raw_data_manager.py
As per instructions, this file will only contain the code, without attempting to run it
or install dependencies like 'pytest', 'requests', 'datasets', 'tqdm'.
Actual download/extraction logic will be mocked or tested structurally.
"""
import unittest
from pathlib import Path
import tempfile
import sys
import shutil
import zipfile
import os

# Add src directory to sys.path
try:
    SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))
    from data.raw_data_manager import RawDataManager, ArcherBenchDownloader, BirdBenchDownloader, SynSQLDownloader
    # For mocking, we might need to access _download_file, _extract_zip, etc.
    # Or mock the external libraries (requests, datasets.load_dataset) directly.
except ImportError as e:
    print(f"Test_RawDataManager: Could not import RawDataManager or components. Error: {e}")
    RawDataManager = None # type: ignore
    ArcherBenchDownloader = None # type: ignore
    BirdBenchDownloader = None # type: ignore
    SynSQLDownloader = None # type: ignore

# Mock external libraries that would be imported by raw_data_manager
# This is for structural testing if the actual libraries are not present.
if 'requests' not in sys.modules:
    class MockResponse:
        def __init__(self, content=b"dummy content", status_code=200, headers=None):
            self.content_bytes = content
            self.status_code = status_code
            self.headers = headers or {'content-length': str(len(content))}
        def raise_for_status(self):
            if self.status_code >= 400: raise Exception(f"HTTP Error {self.status_code}")
        def iter_content(self, chunk_size): yield self.content_bytes
        @property
        def content(self): return self.content_bytes # For some uses

    class MockRequests:
        def get(self, url, stream=False, **kwargs):
            if "example.com/archer_train_english.json" in url: return MockResponse(b'{"archer_train": true}')
            if "example.com/archer_test_english.json" in url: return MockResponse(b'{"archer_test": true}')
            if "bird-bench.github.io/assets/BIRD.zip" in url:
                # Create a dummy zip file for BIRD mock download
                temp_zip_path = Path(tempfile.gettempdir()) / "BIRD_mock.zip"
                with zipfile.ZipFile(temp_zip_path, 'w') as zf:
                    # Create expected internal structure for BIRD
                    zf.writestr("BIRD/train.json", '{"bird_train": true}')
                    zf.writestr("BIRD/dev.json", '{"bird_dev": true}')
                    zf.writestr("BIRD/train_databases/db1.sqlite", "dummy sql data")
                    zf.writestr("BIRD/dev_databases/db_dev1.sqlite", "dummy dev sql data")
                with open(temp_zip_path, 'rb') as f:
                    zip_content = f.read()
                temp_zip_path.unlink() # Clean up temp zip
                return MockResponse(zip_content)
            return MockResponse(b"default mock content", status_code=404) # Default to 404 for other URLs
        class RequestException(Exception): pass
        exceptions = type('MockRequestsExceptions', (object,), {'RequestException': RequestException})()

    sys.modules['requests'] = MockRequests() # type: ignore
    print("Test_RawDataManager: Mocked 'requests' library.")

if 'datasets' not in sys.modules:
    class MockHFDataset:
        def __init__(self, data_dict): self.data = data_dict
        def __getitem__(self, key): return self # Simplistic, assumes key exists and returns self for split
        def to_json(self, path, orient, lines, force_ascii):
            with open(path, 'w') as f: f.write(f'{{"mock_hf_data": "{list(self.data.keys())[0]}"}}') # Write something
        @property
        def column_names(self): return list(self.data.keys()) # Mock column names

    class MockHFDatasetDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Ensure keys are accessible as attributes for dot notation if used by code
            for key, value in self.items():
                if isinstance(value, dict): # If value is a dict, make it a MockHFDataset
                     self[key] = MockHFDataset(value)


    class MockDatasetsModule:
        def load_dataset(self, dataset_name, **kwargs):
            if dataset_name == "seeklhy/SynSQL-2.5M":
                # Return a DatasetDict-like object with a 'train' split
                return MockHFDatasetDict({'train': {'query': ["SELECT 1"], 'sql': ["SELECT 1"]}})
            raise FileNotFoundError(f"Mock dataset {dataset_name} not found.")

        # Define Dataset and DatasetDict for type hinting if original module not loaded
        Dataset = MockHFDataset
        DatasetDict = MockHFDatasetDict


    sys.modules['datasets'] = MockDatasetsModule() # type: ignore
    print("Test_RawDataManager: Mocked 'datasets' library.")

if 'tqdm' not in sys.modules:
    class MockTqdm:
        def __init__(self, iterable=None, *args, **kwargs): self.iterable = iterable
        def __iter__(self): return iter(self.iterable) if self.iterable else iter([])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, n=1): pass

    # If raw_data_manager imports tqdm.auto, mock that path
    if 'tqdm.auto' not in sys.modules:
        tqdm_auto_mock = type('MockTqdmAuto', (object,), {'tqdm': MockTqdm})()
        sys.modules['tqdm.auto'] = tqdm_auto_mock # type: ignore
    sys.modules['tqdm'] = type('MockTqdmModule', (object,), {'tqdm': MockTqdm, 'auto': tqdm_auto_mock})() # type: ignore
    print("Test_RawDataManager: Mocked 'tqdm' library.")


@unittest.skipIf(RawDataManager is None, "RawDataManager or dependencies not loaded, skipping tests.")
class TestRawDataManager(unittest.TestCase):
    """Unit tests for RawDataManager and its downloader components."""

    def setUp(self):
        self.temp_main_data_dir = tempfile.TemporaryDirectory()
        self.base_data_path = Path(self.temp_main_data_dir.name) / "raw_data"
        # Ensure the mock URLs for Archer are set if not using live downloads
        # These URLs are placeholders in the actual ArcherBenchDownloader
        if ArcherBenchDownloader:
            ArcherBenchDownloader.TRAIN_URL = "http://example.com/archer_train_english.json"
            ArcherBenchDownloader.TEST_URL = "http://example.com/archer_test_english.json"

        # Ensure the BIRD URL is set for the mock
        if BirdBenchDownloader:
            BirdBenchDownloader.BIRD_ZIP_URL = "https://bird-bench.github.io/assets/BIRD.zip"


    def tearDown(self):
        self.temp_main_data_dir.cleanup()

    def test_raw_data_manager_initialization(self):
        """Test RawDataManager initialization creates base directory."""
        manager = RawDataManager(base_data_path=self.base_data_path)
        self.assertTrue(self.base_data_path.exists())
        self.assertTrue(self.base_data_path.is_dir())
        self.assertIn("archer_bench", manager.downloaders)
        self.assertIn("bird", manager.downloaders)
        self.assertIn("synsql_2.5m", manager.downloaders)

    @unittest.mock.patch('data.raw_data_manager._download_file') # type: ignore
    @unittest.mock.patch('data.raw_data_manager._extract_zip') # type: ignore
    def test_archer_bench_downloader_structure(self, mock_extract_zip, mock_download_file):
        """Test ArcherBenchDownloader structure and calls (mocked downloads)."""
        if not ArcherBenchDownloader: self.skipTest("ArcherBenchDownloader not loaded.")

        downloader = ArcherBenchDownloader(self.base_data_path)
        downloader.download(download_train=True, download_test=True)

        expected_train_path = self.base_data_path / "archer_bench" / "train" / "archer_train.json"
        expected_test_path = self.base_data_path / "archer_bench" / "test" / "archer_test.json"

        mock_download_file.assert_any_call(ArcherBenchDownloader.TRAIN_URL, expected_train_path)
        mock_download_file.assert_any_call(ArcherBenchDownloader.TEST_URL, expected_test_path)
        self.assertTrue((self.base_data_path / "archer_bench" / "train").exists())
        self.assertTrue((self.base_data_path / "archer_bench" / "test").exists())
        # mock_extract_zip should not be called if direct links are assumed for Archer

    @unittest.mock.patch('data.raw_data_manager._download_file') # type: ignore
    @unittest.mock.patch('data.raw_data_manager._extract_zip') # type: ignore
    @unittest.mock.patch('shutil.move') # type: ignore
    @unittest.mock.patch('shutil.rmtree') # type: ignore
    def test_bird_bench_downloader_structure(self, mock_rmtree, mock_move, mock_extract_zip, mock_download_file):
        """Test BirdBenchDownloader structure and calls (mocked downloads/extraction)."""
        if not BirdBenchDownloader: self.skipTest("BirdBenchDownloader not loaded.")

        downloader = BirdBenchDownloader(self.base_data_path)
        temp_zip_display_path = self.base_data_path / downloader.NAME / "BIRD_temp.zip"

        # Simulate that the zip extraction creates a "BIRD" subfolder
        def side_effect_extract_zip(zip_path, extract_to, delete_zip):
            (extract_to / "BIRD").mkdir(parents=True, exist_ok=True)
            (extract_to / "BIRD" / "train.json").touch()
            (extract_to / "BIRD" / "dev.json").touch()
            (extract_to / "BIRD" / "train_databases").mkdir()
            (extract_to / "BIRD" / "dev_databases").mkdir()
        mock_extract_zip.side_effect = side_effect_extract_zip

        downloader.download(download_train=True, download_dev=True)

        mock_download_file.assert_called_once_with(BirdBenchDownloader.BIRD_ZIP_URL, temp_zip_display_path)
        mock_extract_zip.assert_called_once_with(temp_zip_display_path, self.base_data_path / downloader.NAME, delete_zip=True)

        # Check if shutil.move was called to move contents from BIRD/ subfolder
        self.assertTrue(mock_move.called)
        # Check if the target train/dev structure was attempted
        self.assertTrue((self.base_data_path / downloader.NAME / "train").exists())
        self.assertTrue((self.base_data_path / downloader.NAME / "dev").exists())


    @unittest.mock.patch('data.raw_data_manager.load_dataset') # type: ignore
    @unittest.mock.patch('data.raw_data_manager._save_hf_dataset_split') # type: ignore
    def test_synsql_downloader_structure(self, mock_save_split, mock_load_hf_dataset):
        """Test SynSQLDownloader structure and calls (mocked HuggingFace ops)."""
        if not SynSQLDownloader or not sys.modules.get('datasets'): self.skipTest("SynSQLDownloader or mocked datasets not loaded.")

        # Setup mock for load_dataset
        mock_train_dataset = sys.modules['datasets'].Dataset({'text': ['query1']}) # type: ignore
        mock_test_dataset = sys.modules['datasets'].Dataset({'text': ['query_test']}) # type: ignore
        mock_dataset_dict = sys.modules['datasets'].DatasetDict({ # type: ignore
            'train': mock_train_dataset,
            'test': mock_test_dataset
        })
        mock_load_hf_dataset.return_value = mock_dataset_dict

        downloader = SynSQLDownloader(self.base_data_path)
        downloader.download(download_train=True, download_test=True)

        mock_load_hf_dataset.assert_called_once_with(SynSQLDownloader.HF_DATASET_NAME)

        expected_train_dir = self.base_data_path / downloader.NAME / "train"
        expected_test_dir = self.base_data_path / downloader.NAME / "test"

        mock_save_split.assert_any_call(mock_train_dataset, expected_train_dir, "train")
        mock_save_split.assert_any_call(mock_test_dataset, expected_test_dir, "test")


    @unittest.mock.patch.object(ArcherBenchDownloader, 'download') # type: ignore
    @unittest.mock.patch.object(BirdBenchDownloader, 'download') # type: ignore
    @unittest.mock.patch.object(SynSQLDownloader, 'download') # type: ignore
    def test_raw_data_manager_download_all_calls_individual_downloaders(
        self, mock_synsql_download, mock_bird_download, mock_archer_download):
        """Test that download_all calls the correct download methods with flags."""
        if not RawDataManager: self.skipTest("RawDataManager not loaded.")

        manager = RawDataManager(self.base_data_path)
        manager.download_all(
            download_archer=True, download_bird=False, download_synsql=True,
            download_train_splits=True, download_test_splits=False, download_dev_splits=True
        )
        mock_archer_download.assert_called_once_with(download_train=True, download_test=False)
        mock_bird_download.assert_not_called() # download_bird was False
        mock_synsql_download.assert_called_once_with(download_train=True, download_test=False)

        mock_archer_download.reset_mock()
        mock_synsql_download.reset_mock()

        manager.download_all(
            download_archer=False, download_bird=True, download_synsql=False,
            download_train_splits=False, download_test_splits=True, download_dev_splits=True # dev for BIRD
        )
        mock_archer_download.assert_not_called()
        mock_bird_download.assert_called_once_with(download_train=False, download_dev=True)
        mock_synsql_download.assert_not_called()

    def test_list_downloaded_datasets_structure(self):
        """Test listing downloaded datasets (structural, relies on dir existing)."""
        if not RawDataManager: self.skipTest("RawDataManager not loaded.")
        manager = RawDataManager(self.base_data_path)

        # Create some dummy dataset directories and files
        (self.base_data_path / "archer_bench" / "train").mkdir(parents=True, exist_ok=True)
        (self.base_data_path / "archer_bench" / "train" / "file.json").touch()
        (self.base_data_path / "bird" / "dev").mkdir(parents=True, exist_ok=True)
        (self.base_data_path / "bird" / "dev" / "data.json").touch()
        # synsql_2.5m is not created, so it shouldn't be listed

        downloaded = manager.list_downloaded_datasets()
        self.assertIn("archer_bench", downloaded)
        self.assertIn("train", downloaded["archer_bench"])
        self.assertIn("bird", downloaded)
        self.assertIn("dev", downloaded["bird"])
        self.assertNotIn("synsql_2.5m", downloaded)
        self.assertEqual(len(downloaded), 2)


if __name__ == "__main__":
    if RawDataManager is not None and hasattr(unittest, 'mock'):
        print("Running RawDataManager tests (illustrative execution with mocks)...")
        unittest.main(verbosity=2)
    else:
        reason = "RawDataManager module not loaded"
        if not hasattr(unittest, 'mock'): reason += " or unittest.mock not available"
        print(f"Skipping RawDataManager tests: {reason}.")
        print("This is expected if dependencies are not installed or src path is incorrect in a non-execution environment.")
