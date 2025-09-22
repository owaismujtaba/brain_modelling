import h5py
import pandas as pd
import pdb
from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torchaudio.functional as F

from src.train.utils import gauss_smooth

LOGIT_TO_PHONEME = [
    'BLANK', 'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
    'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH',
    'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH',
    'UW', 'V', 'W', 'Y', 'Z', 'ZH', ' | ',
]


def run_single_decoding_phenome_step(x, input_layer, model, model_args, device):
    """Run a single decoding step through the GRUDecoder."""

    x = gauss_smooth(
        inputs=x,
        device=device,
        smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
        smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
        padding='valid'
    )

    with torch.no_grad():
        logits, _ = model(
            x=x,
            day_idx=torch.tensor([input_layer], device=device),
            states=None,
            return_state=True
        )

    return logits.float().cpu().numpy()



def load_h5py_file(file_path: str, b2txt_csv_df: pd.DataFrame) -> dict:
    """Load trial data from an HDF5 file and match it with CSV metadata.

    Args:
        file_path (str): Path to the HDF5 file.
        b2txt_csv_df (pd.DataFrame): DataFrame containing metadata with
            columns ['Date', 'Block number', 'Corpus'].

    Returns:
        dict: A dictionary with lists of neural features, metadata, and labels.
    """
    data = {
        "neural_features": [],
        "n_time_steps": [],
        "seq_class_ids": [],
        "seq_len": [],
        "transcriptions": [],
        "sentence_label": [],
        "session": [],
        "block_num": [],
        "trial_num": [],
        "corpus": [],
    }

    with h5py.File(file_path, "r") as h5_file:
        for key in h5_file.keys():
            group = h5_file[key]

            neural_features = group["input_features"][:]
            n_time_steps = group.attrs.get("n_time_steps")
            seq_class_ids = group["seq_class_ids"][:] if "seq_class_ids" in group else None
            seq_len = group.attrs.get("seq_len")
            transcription = group["transcription"][:] if "transcription" in group else None
            sentence_label = group.attrs.get("sentence_label")
            session = group.attrs.get("session")
            block_num = group.attrs.get("block_num")
            trial_num = group.attrs.get("trial_num")

            # Match with CSV metadata
            year, month, day = session.split(".")[1:]
            date = f"{year}-{month}-{day}"
            row = b2txt_csv_df[
                (b2txt_csv_df["Date"] == date)
                & (b2txt_csv_df["Block number"] == block_num)
            ]
            corpus_name = row["Corpus"].values[0] if not row.empty else None

            data["neural_features"].append(neural_features)
            data["n_time_steps"].append(n_time_steps)
            data["seq_class_ids"].append(seq_class_ids)
            data["seq_len"].append(seq_len)
            data["transcriptions"].append(transcription)
            data["sentence_label"].append(sentence_label)
            data["session"].append(session)
            data["block_num"].append(block_num)
            data["trial_num"].append(trial_num)
            data["corpus"].append(corpus_name)

    return data



def calculate_per(
    logits_list: List[torch.Tensor], 
    true_seqs: List[torch.Tensor], 
    phone_seq_lens: List[int], 
    day_indices: List[int] = None
) -> Tuple[float, Dict[int, float]]:
    """
    Calculate Phone Error Rate (PER) using raw logits.

    Args:
        logits_list (list): List of logits tensors (batch_size x time x n_classes).
        true_seqs (list): List of ground truth label tensors.
        phone_seq_lens (list): Lengths of each true sequence.
        day_indices (list, optional): Day index for per-day aggregation.

    Returns:
        avg_per (float): Overall phone error rate.
        day_per (dict): PER per day.
    """
    total_edit_distance = 0
    total_seq_length = 0
    day_per = {}

    if day_indices is not None:
        unique_days = np.unique(day_indices)
        for day in unique_days:
            day_per[day] = {"total_edit_distance": 0, "total_seq_length": 0}

    idx = 0  # flat index over all samples
    for batch_idx, logits in enumerate(logits_list):
        batch_size = logits.shape[0]
        for i in range(batch_size):
            seq_len = phone_seq_lens[idx]
            pred_seq = torch.argmax(logits[i, :seq_len, :], dim=-1)
            pred_seq = torch.unique_consecutive(pred_seq)
            pred_seq = pred_seq[pred_seq != 0].cpu().numpy()
            true_seq = true_seqs[batch_idx][i, :seq_len].cpu().numpy()

            edit_distance = F.edit_distance(torch.tensor(pred_seq), torch.tensor(true_seq))
            total_edit_distance += edit_distance
            total_seq_length += seq_len

            if day_indices is not None:
                day = day_indices[idx]
                day_per[day]["total_edit_distance"] += edit_distance
                day_per[day]["total_seq_length"] += seq_len

            idx += 1  # move to next sample

    avg_per = total_edit_distance / total_seq_length

    if day_indices is not None:
        for day in day_per:
            if day_per[day]["total_seq_length"] > 0:
                day_per[day] = day_per[day]["total_edit_distance"] / day_per[day]["total_seq_length"]
            else:
                day_per[day] = None

    return avg_per, day_per
