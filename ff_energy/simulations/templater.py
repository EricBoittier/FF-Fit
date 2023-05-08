from jinja2 import Template
from pathlib import Path

template_files = Path(__file__).parent / "templates"
config_files = Path(__file__).parent / "configs"
input_files = template_files / "inputs"

with open(template_files / "simulation") as file_:
    sim_template = Template(file_.read())

with open(template_files / "job.sh") as file_:
    submit_template = Template(file_.read())


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
        self.templates['submit'] = submit_template
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

        self.kwargs["JOBNAME"] = self.kwargs["JOBNAME"] \
                                 + "_{}".format(self.kwargs["TEMP"])

    def get_sim(self):
        return self.render('sim', **self.kwargs)

    def get_submit(self):
        return self.render('submit', **self.kwargs)
