
class Figure:
    """
    Class for writing figures to latex
    - all figures have an id, a caption, and a path
    The figure may also contain multiple figures, which are subfigures.
    """
    def __init__(self):
        self.filepath = None
        self.caption = None
        self.label = None