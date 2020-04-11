# Create and display textarea widget
import ipywidgets as wdg
from ipywidgets import Layout
import mplcursors
import matplotlib.pyplot as plt


# ------ prepare browser notebook --------

def prepare_browser_nb():
    """
    Use with care, or just copy stuff from here to your nb.
    """

    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))
    import matplotlib
    # %matplotlib notebook
    matplotlib.use('nbagg')


# ------- 2d data point picker ---------

# enable this to use it in a jupter notebook
# %matplotlib notebook
# matplotlib.use('nbagg')

def myonclick_insert_into_TextArea_sel(sel, x_value, y_value, index, textarea):
    mystr = ("{:E}".format(x_value)
                   + " " + "{:E}".format(y_value)
                   + " " + str(index) + "\n")
    sel.annotation.set_text(mystr)
    textarea.value += mystr

def datapoints_extract_utility_2d_scatter_plot(fig, ax, x_data, y_data):
    """ Run this in a jupyter notebook to extract specific 2d data points by clicking on them.
        Suggestion: copy the TextArea's output into a multiline string, say `buf_str` in the next cell and
        parse that with

        from io import StringIO
        np.genfromtxt(StringIO(buf_str), unpack=True) """
    # cursor = mplcursors.cursor(ax, hover=True)

    print("Click to get data point \n x \t y \t index")

    txt = wdg.Textarea(
        value='',
        placeholder='',
        disabled=False,
        layout=Layout(width='100%', height='200px'))

    # this requires a notebook environment
    display(txt)

    mplcursors.cursor().connect(
    "add", lambda sel: myonclick_insert_into_TextArea_sel(sel, x_data[sel.target.index], y_data[sel.target.index], sel.target.index, txt))
