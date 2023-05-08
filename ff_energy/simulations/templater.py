import jinja2
from jinja2 import Template
from pathlib import Path

template_files = Path(__file__).parent / "templates"
input_files = template_files / "inputs"

with open(template_files / "simulation") as file_:
    sim_template = Template(file_.read())


def open_toppar(toppar):
    filenames = list([_.name for _ in input_files.glob("*")])
    if toppar in filenames:
        with open(input_files / toppar) as file_:
            return file_.read()
    else:
        raise FileNotFoundError(f"{toppar} not found in {input_files}")


class Templater:
    def __init__(self):
        self.templates = {}

    def render(self, template, **kwargs):
        return self.templates[template].render(**kwargs)


class SimTemplate(Templater):
    def __init__(self, kwargs):
        super().__init__()
        self.templates['sim'] = sim_template
        self.kwargs = kwargs
        #  assign kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)

        if 'PAR' not in self.kwargs.keys():
            self.kwargs["PAR"] = None
        if 'TOP' not in self.kwargs.keys():
            self.kwargs["TOP"] = None

        if self.kwargs["PAR"] is not None:
            self.kwargs["PAR"] = open_toppar(self.kwargs["PAR"])
        if self.kwargs["TOP"] is not None:
            self.kwargs["TOP"] = open_toppar(self.kwargs["TOP"])

    def get_sim(self):
        return self.render('sim', **self.kwargs)
