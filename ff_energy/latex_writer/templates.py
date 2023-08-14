import jinja2 as j2
from pathlib import Path

TEMPLATES_PATH = Path(__file__).parent / "templates"
TEMPLATE_ENV = j2.Environment(
    loader=j2.FileSystemLoader(TEMPLATES_PATH),
    trim_blocks=True,
    lstrip_blocks=True,
    comment_start_string="%",
    comment_end_string="%",
)

REPORT_TEMPLATE = TEMPLATE_ENV.get_template("report.tex")
FIGURE_TEMPLATE = TEMPLATE_ENV.get_template("figure.tex")
