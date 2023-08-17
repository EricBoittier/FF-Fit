import os
import jinja2 as j2
from pathlib import Path
import shutil

from ff_energy.ffe.constants import FFEPATH, CONFIG_PATH, REPORTS_PATH
from ff_energy.latex_writer.templates import TEMPLATE_ENV, REPORT_TEMPLATE
from ff_energy.latex_writer.figure import Figure
from ff_energy.logs.logging import logger


class Report:
    """
    A report is a class that combines tables, figures, summaries into a single
    latex document. It is the main latex document to compile.
    """

    def __init__(self, name, path=None):
        self.name = name
        #  the location of the files
        if path is None:
            path = REPORTS_PATH / name
        if not isinstance(path, Path):
            path = Path(path)
        # make if not exists
        path.mkdir(parents=True, exist_ok=True)
        self.fig_path = path / "figures"
        self.fig_path.mkdir(parents=True, exist_ok=True)

        self.filename = name + ".tex"
        self.path = path

        self.abstract = ""
        self.body = ""
        self.title = ""
        self.short_title = ""

        self.figures = []
        self.tables = []
        self.sections = []

    def set_abstract(self, abstract):
        """
        Set the abstract of the report
        :param abstract:
        :return:
        """
        self.abstract = abstract

    def set_title(self, title):
        """
        Set the title of the report
        :param title:
        :return:
        """
        self.title = title

    def set_short_title(self, short_title):
        """
        Set the short title of the report
        :param short_title:
        :return:
        """
        self.short_title = short_title

    def add_figure(self, figure: Figure):
        """
        Add a figure to the report
        :param figure:
        :return:
        """
        self.figures.append(figure)

    def add_table(self, table):
        """
        Add a table to the report
        :param table:
        :return:
        """
        self.tables.append(table)

    def add_section(self, section, width=0.55):
        """
        Add a section to the report
        :param section:
        :return:
        """
        if isinstance(section, Figure):
            # move the file to the figures directory
            old = section.filepath
            new = self.fig_path / section.filepath.name
            if old != new:
                shutil.copy(old, new)
            section.set_filepath(f"figures/{section.filepath.stem}")
            section = section.make_figure(width=width)

        logger.info(section)
        self.sections.append(section)

    def join_sections(self):
        """
        Join the sections into a single string
        :return:
        """
        self.body = "\n".join(self.sections)

    def render(self):
        return REPORT_TEMPLATE.render(
            TITLE=self.title,
            SHORTTITLE=self.short_title,
            ABSTRACT=self.abstract,
            BODY=self.body,
        )

    def write_document(self):
        """
        Write the document to the path
        :return:
        """
        self.join_sections()
        with open(self.path / self.filename, "w") as f:
            f.write(self.render())

    def compile(self):
        os.chdir(self.path)
        logger.info(f"Compiling {self.filename}")
        logger.info(f"Current directory: {os.getcwd()}")
        command = "pdflatex {} -output-directory {}".format(self.filename, self.path)
        logger.info(command)
        os.system(command)
