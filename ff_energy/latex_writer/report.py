import os
import jinja2 as j2
from pathlib import Path

from ff_energy.ffe.constants import FFEPATH, CONFIG_PATH, REPORTS_PATH
from ff_energy.latex_writer.templates import TEMPLATE_ENV, REPORT_TEMPLATE

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
            # make if not exists
            path.mkdir(parents=True, exist_ok=True)

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

    def add_figure(self, figure):
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

    def add_section(self, section):
        """
        Add a section to the report
        :param section:
        :return:
        """
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
        with open(self.path / self.filename, "w") as f:
            f.write(self.render())

    def compile(self):
        os.chdir(self.path)
        os.system("pdflatex {} ".format(
            str(self.path / self.filename)
        )
        )



