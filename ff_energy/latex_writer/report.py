import jinja2 as j2
from pathlib import Path

from ff_energy.ffe.constants import FFEPATH, CONFIG_PATH

REPORTS_PATH = FFEPATH / "latex_reports"

class Report:
    """
    A report is a class that combines tables, figures, summaries into a single
    latex document. It is the main latex document to compile.
    """
    def __init__(self):
        self.figures = []
        self.tables = []
        self.sections = []

