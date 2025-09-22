
import os
from omegaconf import OmegaConf, DictConfig


def load_yaml_config(config_path: str) -> DictConfig:
    """
    Loads a YAML configuration file with OmegaConf, 
    supporting variable interpolation.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        DictConfig: Parsed configuration as an OmegaConf DictConfig.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"The configuration file {config_path} does not exist."
        )

    return OmegaConf.load(config_path)



def log_info(logger, text):
    """
    logs text with heading
    """
    logger.info('*'*60)
    logger.info(text)
    logger.info('*'*60)
