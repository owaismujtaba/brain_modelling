import logging
import os
import sys
import pdb

def setup_logger(config):
    """
    Returns a reusable logger object with the specified name and level,
    configured to log to both a file and the console.

    Args:
        name (str): Name of the logger. If None, uses the root logger.
        log_file_path (str): Full path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        fmt (str): Log message format. If None, a default format is used.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_config = config['logging']
    name=log_config['name']
    log_file_path=log_config['log_file_path']
    level=log_config['level']

    # Ensure the directory exists
    log_dir = os.path.dirname(str(log_file_path))
    os.makedirs(log_dir, exist_ok=True)

    # Default format
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)
    # Create or get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_file_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Stream handler for console output
    # Always print logs to stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
