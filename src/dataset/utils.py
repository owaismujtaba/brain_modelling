import os
import h5py
import numpy as np
from pathlib import Path
import pdb

from src.utils import log_info


def train_test_split_indices(config, logger, file_paths):
    """
    Split data from file_paths into train and test splits.

    Returns two dictionaries that detail which trials in each day
    will be a part of that split. Example:

        {
            0: {'trials': [1, 2, 3], 'session_path': 'path'},
            1: {'trials': [2, 5, 6], 'session_path': 'path'}
        }

    Args:
        config (dict): Configuration dictionary with dataset and training info.
        file_paths (list[str]): List of file paths to the HDF5 files.
            Each file contains trial data.
        test_percentage (float): Percentage of trials to use for testing.
            0 → all training, 1 → all testing.
        seed (int): Seed for reproducibility. -1 for random.
        bad_trials_dict (dict): Trials to exclude. Format:

            {
                'session_name_1': {block_num_1: [trial_nums], ...},
                'session_name_2': {block_num_1: [trial_nums], ...}
            }

    Returns:
        tuple: (train_trials, test_trials), each a dict with structure:

            {
                day_idx: {
                    'trials': [trial_indices],
                    'session_path': path
                }
            }
    """
    
   
    dataset_conf = config["dataset"]['info']
    seed = dataset_conf.get("seed", -1)
    bad_trials_dict = dataset_conf.get("bad_trials_dict", None)
    test_percentage = config["train"].get("test_percentage", 0.2)

    log_info(
        logger=logger,
        text=f"Generating train and text indicies with test percentage:: {test_percentage}"
    )
    
    if seed != -1:
        np.random.seed(seed)

    trials_per_day = {}
    i = 0
    for session, paths in file_paths.items(): 
        if not session.startswith('t15'):
            continue       
        if len(paths)<1:
            print(f"Day:{i} Path:{session}")
            trials_per_day[i] = {
                "num_trials": 0,
                "trial_indices": [],
                "session_path": session,
                }
            i += 1
            continue
        path = str(paths[0])
        print(f"Day:{i} Path:{path}")
        good_trial_indices = []
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                for t, key in enumerate(f.keys()):
                    trial = f[key]
                    block_num = trial.attrs["block_num"]
                    trial_num = trial.attrs["trial_num"]

                    if (
                        bad_trials_dict
                        and session in bad_trials_dict
                        and str(block_num) in bad_trials_dict[session]
                        and trial_num in bad_trials_dict[session][str(block_num)]
                    ):
                        continue

                    good_trial_indices.append(t)

        trials_per_day[i] = {
            "num_trials": len(good_trial_indices),
            "trial_indices": good_trial_indices,
            "session_path": path,
        }
        i += 1

    train_trials, test_trials = {}, {}

    for day, info in trials_per_day.items():
        all_indices = info["trial_indices"]
        num_trials = info["num_trials"]
        session_path = info["session_path"]

        if test_percentage in (0, 1):
            train_indices = all_indices if test_percentage == 0 else []
            test_indices = all_indices if test_percentage == 1 else []
        else:
            num_test = max(1, int(num_trials * test_percentage))
            test_indices = np.random.choice(
                all_indices, size=num_test, replace=False
            ).tolist()
            train_indices = [idx for idx in all_indices if idx not in test_indices]

        train_trials[day] = {"trials": train_indices, "session_path": session_path}
        test_trials[day] = {"trials": test_indices, "session_path": session_path}

    return train_trials, test_trials



def train_test_split_indices1(config, logger, file_paths):
    """
    Split data from file_paths into train and test splits.

    Returns two dictionaries that detail which trials in each day
    will be a part of that split. Example:

        {
            0: {'trials': [1, 2, 3], 'session_path': 'path'},
            1: {'trials': [2, 5, 6], 'session_path': 'path'}
        }

    Args:
        config (dict): Configuration dictionary with dataset and training info.
        file_paths (list[str]): List of file paths to the HDF5 files.
            Each file contains trial data.
        test_percentage (float): Percentage of trials to use for testing.
            0 → all training, 1 → all testing.
        seed (int): Seed for reproducibility. -1 for random.
        bad_trials_dict (dict): Trials to exclude. Format:

            {
                'session_name_1': {block_num_1: [trial_nums], ...},
                'session_name_2': {block_num_1: [trial_nums], ...}
            }

    Returns:
        tuple: (train_trials, test_trials), each a dict with structure:

            {
                day_idx: {
                    'trials': [trial_indices],
                    'session_path': path
                }
            }
    """
    
   
    dataset_conf = config["dataset"]['info']
    seed = dataset_conf.get("seed", -1)
    bad_trials_dict = dataset_conf.get("bad_trials_dict", None)
    test_percentage = config["train"].get("test_percentage", 0.2)

    log_info(
        logger=logger,
        text=f"Generating train and text indicies with test percentage:: {test_percentage}"
    )
    
    if seed != -1:
        np.random.seed(seed)

    trials_per_day = {}
    pdb.set_trace()
    for i, path in enumerate(file_paths):
        path = str(path)  # ensure string if Path object
        session = next(
            s for s in Path(path).parts
            if s.startswith("t15.20") or s.startswith("t12.20")
        )

        good_trial_indices = []
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                for t, key in enumerate(f.keys()):
                    trial = f[key]
                    block_num = trial.attrs["block_num"]
                    trial_num = trial.attrs["trial_num"]

                    if (
                        bad_trials_dict
                        and session in bad_trials_dict
                        and str(block_num) in bad_trials_dict[session]
                        and trial_num in bad_trials_dict[session][str(block_num)]
                    ):
                        continue

                    good_trial_indices.append(t)

        trials_per_day[i] = {
            "num_trials": len(good_trial_indices),
            "trial_indices": good_trial_indices,
            "session_path": path,
        }

    train_trials, test_trials = {}, {}

    for day, info in trials_per_day.items():
        all_indices = info["trial_indices"]
        num_trials = info["num_trials"]
        session_path = info["session_path"]

        if test_percentage in (0, 1):
            train_indices = all_indices if test_percentage == 0 else []
            test_indices = all_indices if test_percentage == 1 else []
        else:
            num_test = max(1, int(num_trials * test_percentage))
            test_indices = np.random.choice(
                all_indices, size=num_test, replace=False
            ).tolist()
            train_indices = [idx for idx in all_indices if idx not in test_indices]

        train_trials[day] = {"trials": train_indices, "session_path": session_path}
        test_trials[day] = {"trials": test_indices, "session_path": session_path}

    return train_trials, test_trials



def get_all_files(parent_dir, kind='train', extensions='.hdf5', logger=None):
    """
    Recursively loads all files in the parent directory and its subdirectories.
    If extensions is provided, only files with those extensions are returned.

    Args:
        parent_dir (str): Path to the parent directory.
        kind (str): Substring that must be present in the filename.
        extensions (str or tuple, optional): File extensions to filter by,
            e.g. '.hdf5' or ('.hdf5', '.npy').
        logger (logging.Logger, optional): Logger for info messages.

    Returns:
        dict: {folder_name: [file paths]}
    """
    if logger is not None:
        log_info(
            logger=logger,
            text=f"Getting data filepaths for KIND:: {kind}"
        )

    folder_files = {}

    for root, _, files in os.walk(parent_dir):
        valid_files = []
        for file in files:
            if extensions is None or file.lower().endswith(extensions):
                if kind in file:
                    valid_files.append(os.path.join(root, file))

       
        folder_name = os.path.basename(root)
        folder_files[folder_name] = sorted(valid_files)
        

    return dict(sorted(folder_files.items()))



def get_all_files1(parent_dir,kind='train', extensions='hdf5', logger=None):
    """
    Recursively loads all files in the parent directory and its subdirectories.
    If extensions is provided, only files with those extensions are returned.

    Args:
        parent_dir (str): Path to the parent directory.
        extensions (tuple, optional): File extensions to filter by, e.g. ('.hdf5')

    Returns:
        List[str]: List of file paths.
    """
    if logger is not None:
        log_info(
            logger=logger,
            text=f"Getting data filepaths for KIND:: {kind}"
        )
    all_files = []
    
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if extensions is None or file.lower().endswith(extensions):
                if kind in file:
                    all_files.append(os.path.join(root, file))
                
    return sorted(all_files)


