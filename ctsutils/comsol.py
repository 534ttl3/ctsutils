import numpy as np
from io import StringIO

def read_comsol_table_complex(filename, skip_header=0, **kwargs):
    """ Comsol exports it's tables with i instead of j for
    imaginary unit. This function replaces i with j and makes
    it possible to read in all columns.
    """
    file_str = open(filename).read()
    trimmed_header_str = "\n".join(file_str.split("\n")[skip_header:])
    replaced_str = trimmed_header_str.replace("i", "j")
    # print(replaced_str)
    return np.genfromtxt(StringIO(replaced_str), dtype=np.complex, skip_header=0, **kwargs)
