import logging
import logging.config
import logging.handlers


logging.config.fileConfig('items_classifier/configs/logger.conf')


def get_logger(name):
    logger = logging.getLogger(name)
    return logger
