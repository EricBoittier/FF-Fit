from pathlib import Path

from ff_energy.latex_writer.templates import TEMPLATE_ENV, FIGURE_TEMPLATE


class Figure:
    """
    Class for writing figures to latex
    - all figures have an id, a caption, and a path
    The figure may also contain multiple figures, which are subfigures.
    """

    def __init__(self, filepath, caption, label):
        self.caption = caption
        self.label = label

        if not isinstance(filepath, Path):
            self.filepath = Path(filepath)
        elif isinstance(filepath, Path):
            self.filepath = filepath

        #  check the file exists
        if not self.filepath.exists():
            raise FileNotFoundError(f"File {self.filepath} does not exist")

    def set_filepath(self, filepath):
        """
        Set the filepath
        :param filepath:
        :return:
        """
        if not isinstance(filepath, Path):
            self.filepath = Path(filepath)

    def make_figure(self, path=None, width=0.55):
        """
        Make the figure
        :return:
        """
        if path is None:
            path = self.filepath
        else:  # use the default path
            if not isinstance(path, Path):
                path = Path(path)
            path = path / self.filepath.name

        return FIGURE_TEMPLATE.render(
            FILE=path,
            CAPTION=self.caption,
            LABEL=self.label,
            WIDTH=f"{width}\\textwidth",
        )
