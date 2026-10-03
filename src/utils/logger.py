import logging
import os
import sys

def get_logger(name: str = "KiTS21", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get configured logger with console and file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler if log_file is specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
    return logger
