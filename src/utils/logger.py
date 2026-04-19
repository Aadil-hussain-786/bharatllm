import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Standardized logger for the MLOps pipeline.
    Formats logs with timestamps and severity for CloudWatch/Datadog compatibility.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
        logger.addHandler(handler)
    return logger
