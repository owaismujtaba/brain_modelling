from src.utils import load_yaml_config
from src.train.trainer import training_pipeline
from src.inference.evaluate import evaluate_pipeline
from src.logging.log import setup_logger

from src.dataset.dataset import DatasetNeural

config = load_yaml_config('config.yaml')

logger = setup_logger(
            config=config
        )
    
if config['run'] == 'train':
    training_pipeline(config=config, logger=logger)

if config['run'] == 'inference':
    evaluate_pipeline(config=config, logger=logger)


if config['create_llm_dataset']:
    dataset = DatasetLLM(config=config, logger=logger)