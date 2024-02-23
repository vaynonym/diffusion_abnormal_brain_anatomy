import logging
def setup_basic_logger():
    LOGGER = logging.getLogger("basic")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    return LOGGER

LOGGER = setup_basic_logger()