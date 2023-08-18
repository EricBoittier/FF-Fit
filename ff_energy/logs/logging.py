import logging


def hide_logs():
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.texmanager").disabled = True
    logging.getLogger("matplotlib.dviread").disabled = True
    logging.getLogger("matplotlib").disabled = True
    logging.getLogger("matplotlib.axes").disabled = True
    logging.getLogger("matplotlib.pyplot").disabled = True
    logging.getLogger("PngImagePlugin").disabled = True

    logger.propagate = False


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(filename="info.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

hide_logs()
