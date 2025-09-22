from src.utils import load_yaml_config
from src.train.trainer import training_pipeline
from src.logging.log import setup_logger

config = load_yaml_config('config.yaml')

logger = setup_logger(
            config=config
        )
    
if config['run'] == 'train':
    training_pipeline(config=config, logger=logger)
