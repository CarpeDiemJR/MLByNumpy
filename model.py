"""
Design the general format of a model.

A Model() class should at least contain:
    1. name
"""

class Model:
    """
    General design.
    """

    def __init__(self, name=None):
        if name is not None:
            self.name = name
        else:
            self.name = "ModelObject"

    def __str__(self):
        output_format = "<\"{}\">"
        return output_format.format(self.name)


if __name__ == "__main__":
    model = Model()
    print(model)