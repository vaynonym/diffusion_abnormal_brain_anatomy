import logging
from contextlib import contextmanager
from src.directory_management import OUTPUT_DIRECTORY
import os.path

def setup_basic_logger():
    LOGGER = logging.getLogger("basic")
    if not LOGGER.handlers:
        LOGGER.propagate = False
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
        filepath = os.path.join(OUTPUT_DIRECTORY, "logfile.txt")
        file_handler = logging.FileHandler(filepath)
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
        LOGGER.setLevel(logging.INFO)
        LOGGER.info(f"Logger configured to write to {filepath}")
    return LOGGER

LOGGER = setup_basic_logger()

@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    # taken from gist.github.com/simon-weber/7853144
    """ 
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)