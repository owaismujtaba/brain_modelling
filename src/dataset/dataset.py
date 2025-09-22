import math
import os
import pandas as pd
import pdb  # Consider removing if unused

import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import log_info
from src.dataset.utils import get_all_files, load_h5py_file
from src.inference.utils import LOGIT_TO_PHONEME


import pdb
from typing import Dict, List, Tuple, Any


class DatasetLLM:
    """Dataset handler for LLM training and validation data."""

    def __init__(self, config: Dict[str, Any], logger):
        """
        Initialize the dataset object.

        Args:
            config (dict): Configuration dictionary.
            logger: Logger instance for logging.
        """
        log_info(logger=logger, text="LLM Dataset Initialization")
        self.config = config
        self.logger = logger
        self._setup_config_params()

        self.val_filepaths = get_all_files(
            parent_dir=self.dataset_dir, kind="val"
        )
        self.train_filepaths = get_all_files(
            parent_dir=self.dataset_dir, kind="train"
        )

        self._extract_data()
        self._save_data()

    def _setup_config_params(self) -> None:
        """Set dataset configuration parameters."""
        self.dataset_dir = self.config["dataset"]["info"]["dataset_dir"]
        self.llm_dir = self.config['directory']['llm_dataset_dir']
        self.cur_dir = os.getcwd()

    def _get_phenome_text(self, seq_class_ids: List[int]) -> str:
        """
        Convert class IDs to phoneme text.

        Args:
            seq_class_ids (list): List of class IDs.

        Returns:
            str: Space-separated phoneme string.
        """
        seq_class_ids = [
            int(p)
            for i, p in enumerate(seq_class_ids)
            if p != 0 and (i == 0 or p != seq_class_ids[i - 1])
        ]
        phenome_text = [LOGIT_TO_PHONEME[p] for p in seq_class_ids]
        return " ".join(phenome_text)

    def _get_data(self, filepath: str) -> Tuple[List[str], List[str]]:
        """
        Load text and phoneme sequences from an HDF5 file.

        Args:
            filepath (str): Path to the HDF5 file.

        Returns:
            tuple: (sentences_text, sentences_phenomes)
        """
        data = load_h5py_file(filepath)
        sentences_text = data["sentence_label"]
        sentences_phenomes = [
            self._get_phenome_text(seq) for seq in data["seq_class_ids"]
        ]
        return sentences_text, sentences_phenomes

    def _extract_from_filepaths(
        self, filepaths: Dict[str, List[str]], label: str
    ) -> Tuple[List[str], List[str]]:
        """
        Extract data from filepaths for a dataset split.

        Args:
            filepaths (dict): Mapping of session → filepaths.
            label (str): Label for logging (e.g., "Train" or "Val").

        Returns:
            tuple: (text_sentences, phenome_sentences)
        """
        self.logger.info(f"Extracting data for {label} filepaths")
        text_sentences, phenome_sentences = [], []

        for session, paths in filepaths.items():
            if not paths:
                continue
            path = paths[0]
            sentences_text, sentences_phenomes = self._get_data(path)

            text_sentences.extend(sentences_text)
            phenome_sentences.extend(sentences_phenomes)

            self.logger.info(
                f"Extracted {len(sentences_text)} trials from {session}"
            )

        return text_sentences, phenome_sentences

    def _extract_data(self) -> None:
        """Extract train and validation data."""
        train_texts, train_phenomes = self._extract_from_filepaths(
            self.train_filepaths, "Train"
        )
        val_texts, val_phenomes = self._extract_from_filepaths(
            self.val_filepaths, "Val"
        )

        # Concatenate results
        self.text_sentences = train_texts + val_texts
        self.phenome_sentences = train_phenomes + val_phenomes
    
    def _save_data(self):
        self.logger.info("Saving LLM dataset")
        folder = Path(self.cur_dir, self.llm_dir)
        os.makedirs(folder)
        path = Path(folder, 'llm_data.csv')

        dataset = {'text':self.text_sentences, 'phenomes':self.phenome_sentences}

        dataset = pd.DataFrame(dataset)
        dataset.to_csv(path)
        self.logger.info(f"LLM dataset saved to {path}")



class BrainToTextDataset(Dataset):
    """
    Dataset for brain-to-text data.

    Returns an entire batch of data instead of a single example.
    """

    def __init__(self, config, logger, trial_indices, kind):
        self.logger = logger
        self.kind = kind
        self.trial_indices = trial_indices
        self.config = config

        log_info(logger=logger, text=f"Creating Dataset for KIND:: {self.kind}")

        self._setup_conf_params()
        self._check_config_params()

        if self.kind == "train":
            self.batch_index = self.create_batch_index_train()
        else:
            self.batch_index = self.create_batch_index_test()
            self.n_batches = len(self.batch_index)

    def __len__(self):
        """
        Number of batches in the dataset.

        For training, this is fixed from config.
        For test/validation, it depends on dataset size.
        """
        return self.n_batches

    def __getitem__(self, idx):
        """
        Get an entire batch of data from the dataset.
        """
        batch = {
            "input_features": [],
            "seq_class_ids": [],
            "n_time_steps": [],
            "phone_seq_lens": [],
            "day_indices": [],
            "transcriptions": [],
            "block_nums": [],
            "trial_nums": [],
        }

        index = self.batch_index[idx]
        total_bad_trials = 0
        # Iterate through each day in the index
        for day in index.keys():
            session_path = self.trial_indices[day]["session_path"]
            with h5py.File(session_path, "r") as f:
                for trial in index[day]:
                    try:
                        g = f[f"trial_{trial:04d}"]
                        # Load features
                        input_features = torch.from_numpy(
                            g["input_features"][:]
                        )
                        if len(self.feature_subset)>0:
                            input_features = input_features[:, self.feature_subset]

                        # Append features and metadata
                        batch["input_features"].append(input_features)
                        batch["seq_class_ids"].append(
                            torch.from_numpy(g["seq_class_ids"][:])
                        )
                        batch["transcriptions"].append(
                            torch.from_numpy(g["transcription"][:])
                        )
                        batch["n_time_steps"].append(g.attrs["n_time_steps"])
                        batch["phone_seq_lens"].append(g.attrs["seq_len"])
                        batch["day_indices"].append(int(day))
                        batch["block_nums"].append(g.attrs["block_num"])
                        batch["trial_nums"].append(g.attrs["trial_num"])

                    except Exception as e:
                        
                        print(f"Error loading trial {trial} from {session_path}: {e}")
                        total_bad_trials += 1
                        continue
        
        
        # Pad sequences to form a cohesive batch
        batch["input_features"] = pad_sequence(
            batch["input_features"], batch_first=True, padding_value=0
        )
        batch["seq_class_ids"] = pad_sequence(
            batch["seq_class_ids"], batch_first=True, padding_value=0
        )

        # Convert lists to tensors
        batch["n_time_steps"] = torch.tensor(batch["n_time_steps"])
        batch["phone_seq_lens"] = torch.tensor(batch["phone_seq_lens"])
        batch["day_indices"] = torch.tensor(batch["day_indices"])
        batch["transcriptions"] = torch.stack(batch["transcriptions"])
        batch["block_nums"] = torch.tensor(batch["block_nums"])
        batch["trial_nums"] = torch.tensor(batch["trial_nums"])

        return batch

    def _check_config_params(self):
        """Validate dataset configuration parameters."""
        self.logger.info("Validating configuration arguments")

        if self.kind not in ["train", "test"]:
            raise ValueError(
                f'split must be either "train" or "test". Received {self.kind}'
            )

        # Count total number of trials
        for day in self.trial_indices:
            self.n_trials += len(self.trial_indices[day]["trials"])

        if (
            self.must_include_days is not None
            and len(self.must_include_days) > self.days_per_batch
        ):
            raise ValueError(
                f"must_include_days ({self.must_include_days}) must be <= days_per_batch ({self.days_per_batch})"
            )

        if (
            self.must_include_days is not None
            and len(self.must_include_days) > self.n_days
            and self.kind != "train"
        ):
            raise ValueError(
                f"must_include_days is invalid for test data. "
                f"Received {self.must_include_days} but only {self.n_days} days available"
            )

        # Handle negative indices in must_include_days
        if self.must_include_days is not None:
            self.must_include_days = [
                self.n_days + d if d < 0 else d for d in self.must_include_days
            ]

        if self.kind == "train" and self.days_per_batch > self.n_days:
            raise ValueError(
                f"Requested days_per_batch={self.days_per_batch} > available days {self.n_days}."
            )

        self.logger.info("Configuration arguments OK!!!")

    def _setup_conf_params(self):
        """Set up configuration parameters."""
        self.logger.info("Setting up configuration parameters")

        train_conf = self.config["train"]
        self.batch_size = train_conf["batch_size"]
        self.n_batches = train_conf["n_train_batches"]
        self.days_per_batch = train_conf["days_per_batch"]
        self.must_include_days = train_conf["must_include_days"]
        self.feature_subset = train_conf['feature_subset']

        self.days = {}
        self.n_trials = 0
        self.n_days = len(self.trial_indices)

        seed = self.config["dataset"]["info"]["seed"]
        if seed != -1:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def create_batch_index_train(self):
        """
        Create index mapping batch_number to trials.

        Each batch contains days_per_batch unique days,
        with trials evenly split across days (as close as possible).
        """
        self.logger.info("Creating batch index for KIND:: train")
        batch_index = {}

        # Precompute optional days
        non_must_include_days = []
        if self.must_include_days:
            non_must_include_days = [
                d for d in self.trial_indices if d not in self.must_include_days
            ]

        for batch_idx in range(self.n_batches):
            batch = {}

            # Select days
            if len(self.must_include_days) > 0:
                days = np.concatenate(
                    (
                        self.must_include_days,
                        np.random.choice(
                            non_must_include_days,
                            size=self.days_per_batch - len(self.must_include_days),
                            replace=False,
                        ),
                    )
                )
            else:
                days = np.random.choice(
                    list(self.trial_indices.keys()),
                    size=self.days_per_batch,
                    replace=False,
                )

            # Assign trials
            num_trials = math.ceil(self.batch_size / self.days_per_batch)
            for day in days:
                trials = np.random.choice(
                    self.trial_indices[day]["trials"],
                    size=num_trials,
                    replace=True,
                )
                batch[day] = trials

            # Remove extra trials if needed
            extra_trials = num_trials * len(days) - self.batch_size
            while extra_trials > 0:
                day = np.random.choice(days)
                if len(batch[day]) > 0:
                    batch[day] = batch[day][:-1]
                    extra_trials -= 1

            batch_index[batch_idx] = batch

        self.logger.info("Dataset Index created")
        return batch_index

    def create_batch_index_test(self):
        """
        Create index for validation/testing data.

        Ensures every trial is seen once in batches of up to batch_size.
        """
        self.logger.info("Creating batch index for KIND:: test/val")

        batch_index = {}
        batch_idx = 0

        for day, trials in self.trial_indices.items():
            num_trials = len(trials["trials"])
            num_batches = (num_trials + self.batch_size - 1) // self.batch_size

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_trials)
                batch_trials = trials["trials"][start_idx:end_idx]
                batch_index[batch_idx] = {day: batch_trials}
                batch_idx += 1

        self.logger.info("Dataset Index created")
        return batch_index
