import logging

class AutoMLBase:
    def __init__(self):
        self.logger = logging.getLogger("AutoML")
        logging.basicConfig(level=logging.INFO)

    def log(self, msg):
        self.logger.info(msg)