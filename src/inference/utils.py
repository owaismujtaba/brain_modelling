import h5py
import pandas as pd
import pdb
import torch


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
