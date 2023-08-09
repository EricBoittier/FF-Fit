
from ff_energy.latex_writer.templates import (
    TEMPLATE_ENV, FIGURE_TEMPLATE
)



class Figure:
    """
    Class for writing figures to latex
    - all figures have an id, a caption, and a path
    The figure may also contain multiple figures, which are subfigures.
    """
    def __init__(self, filepath, caption, label):
        self.filepath = filepath
        self.caption = caption
        self.label = label

    def make_figure(self):
        """
        Make the figure
        :return:
        """
        return FIGURE_TEMPLATE.render(
            filepath=self.filepath,
            caption=self.caption,
            label=self.label,
        )
