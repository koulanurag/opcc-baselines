import logging
import os


def init_logger(base_path: str, name: str):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]'
                                  '[%(filename)s>%(funcName)s] => %(message)s')
    file_path = os.path.join(base_path, name + '.log')
    logger = logging.getLogger(name)
    logging.getLogger().handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.FileHandler(file_path, mode='w')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
