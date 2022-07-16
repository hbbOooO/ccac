import logging

class Logger:
    level_relation = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    logger = None
    filename = None
    level = None
    format = None

    __NAME__ = 'ccac'

    @classmethod
    def set_config(cls, filename=None, level='info', format='%(asctime)s - %(levelname)s: %(message)s'):
        Logger.filename = filename
        Logger.level = level
        Logger.format = format

    @classmethod
    def get_logger(cls):
        if not Logger.logger:
            format_str = logging.Formatter(Logger.format)
            logger = logging.getLogger(Logger.__NAME__)
            logger.setLevel(Logger.level_relation[Logger.level])
            sh = logging.StreamHandler()
            sh.setFormatter(format_str)
            logger.addHandler(sh)
            fh = logging.FileHandler(Logger.filename)
            fh.setFormatter(format_str)
            logger.addHandler(fh)
            Logger.logger = logger
        return Logger.logger