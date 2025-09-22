import torch
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from src.train.models import GRUDecoder
from src.dataset.utils import get_all_files
from src.inference.utils import load_h5py_file
from src.train.utils import gauss_smooth

from src.inference.utils import LOGIT_TO_PHONEME, run_single_decoding_phenome_step



def evaluate_pipeline(config, logger, kind='val'):
    """Evaluate the GRUDecoder model on a dataset."""

    logger.info("Evaluating the model")

    model = GRUDecoder(config=config)

    description_csv = pd.read_csv(config['dataset']['info']['description_filepath'])

    model_dir = config['directory']['checkpoint_dir']
    model_path = Path(model_dir, 'best_model')
    model_args = OmegaConf.load(Path(model_dir, 'args.yaml'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model.to(device)

    checkpoint = torch.load(model_path, weights_only=False)
    
    # Remove module prefixes if present
    for key in list(checkpoint['model_state_dict'].keys()):
        new_key = key.replace("module.", "").replace("_orig_mod.", "")
        checkpoint['model_state_dict'][new_key] = checkpoint['model_state_dict'].pop(key)

    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model loaded")
    model.eval()

    filepaths_dic = get_all_files(
        parent_dir=config['dataset']['info']['dataset_dir'], kind=kind
    )

    test_data = {}
    total_trials = 0
    day_index_map = {}
    day_index = 0

    for key, session in filepaths_dic.items():
        if len(session) < 1:
            continue

        day_index_map[key] = day_index
        path = session[0]
        data = load_h5py_file(path, description_csv)

        test_data[key] = data
        total_trials += len(data['neural_features'])
        day_index += 1

        # Remove break if you want to process all sessions
        break

    print("Total number of trials:", total_trials)

    # Prediction loop
    with tqdm(total=total_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
        for session, data in test_data.items():
            data['logits'] = []
            input_layer = day_index_map[session]

            for trial_idx, neural_input in enumerate(data['neural_features']):
                # Add batch dimension and convert to torch tensor
                neural_input = torch.tensor(
                    np.expand_dims(neural_input, axis=0),
                    device=device,
                    dtype=torch.float32
                )

                logits = run_single_decoding_phenome_step(
                    neural_input, input_layer, model, model_args, device
                )
                data['logits'].append(logits)
                pbar.update(1)

    # Decode predicted sequences
    for session, data in test_data.items():
        data['pred_seq'] = []

        for trial_idx, logits in enumerate(data['logits']):
            logits_trial = logits[0]
            pred_seq = np.argmax(logits_trial, axis=-1)

            # Remove blanks and consecutive duplicates
            pred_seq = [int(p) for i, p in enumerate(pred_seq)
                        if p != 0 and (i == 0 or p != pred_seq[i - 1])]

            pred_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq]
            data['pred_seq'].append(pred_seq)

            block_num = data['block_num'][trial_idx]
            trial_num = data['trial_num'][trial_idx]

            print(f'Session: {session}, Block: {block_num}, Trial: {trial_num}')

            if kind == 'val':
                sentence_label = data['sentence_label'][trial_idx]
                true_seq = data['seq_class_ids'][trial_idx][:data['seq_len'][trial_idx]]
                true_seq = [LOGIT_TO_PHONEME[p] for p in true_seq]

                print(f'Sentence label:      {sentence_label}')
                print(f'True sequence:       {" ".join(true_seq)}')

            print(f'Predicted sequence:  {" ".join(pred_seq)}\n')


