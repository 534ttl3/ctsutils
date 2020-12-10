import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_values_from_meshgrid(mg, np_bool_array):
    """ """
    _ = np.where(np_bool_array)
    return list((map(lambda n: (mg[n][np_bool_array]).item(), range(len(mg)))))


def get_indices_from_np_where_quer(np_where_query):
    """ """
    return tuple(map(lambda el: np.int(el), np.where(Ts == np.max(Ts))))


def shape_arrays_for_pcolor_plotting(ps, indexing_list_indep_vars, ordered_params, dep_var_mgf):
    """ """

    X = ps.get_mgf_arr(ordered_params[0].name)[
        tuple(indexing_list_indep_vars)]
    Y = ps.get_mgf_arr(ordered_params[1].name)[
        tuple(indexing_list_indep_vars)]
    Z = None

    if (np.shape(dep_var_mgf) == np.shape(X) and
        np.shape(dep_var_mgf) == np.shape(Y)):
        Z = dep_var_mgf
        # delivered in the appropriate shape to be plotted (e.g. after integrating out one dimension, i.e.
        # function with signature (N^d x N^d x ...) -> N^(d-1) evaluated and passed to this plotting function
        # in the right shape)
        print("plot: Z is already in the right shape")
    else:
        dep_var_mgf_dim = len(np.shape(dep_var_mgf))
        dim_reduction = ps.get_dimension() - dep_var_mgf_dim

        indexing_list_dep_var = indexing_list_indep_vars[dim_reduction:]
        # dimensionality reduced by an operation -> e.g. integration
        # -> cannot index the result with the original number of indices
        # index it from back to front until dep_var_mgf's dimensions run out
        # -> then you will get 2 dimensions for the pcolor plot

        Z = dep_var_mgf[tuple(indexing_list_dep_var)] # function with signature (N^d x N^d x ...) -> N^d evaluated

    return [X, Y, Z]

class CParam:
    def __init__(self, name, np_arr, unit=None):
        """
        Args:
            name: string
            np_arr: numpy array
            unit: set the unit as a string
        """
        self.name = name
        self.np_arr = np_arr
        self.unit = unit

    def get_unit_str(self):
        """ """
        if self.unit is None:
            return ""
        return self.unit

    def get_label_str(self):
        """ """
        if self.unit is not None:
            return self.name + " in " + self.get_unit_str()

        return self.name

class CSlider(Slider):
    # def __init__(self, param, *mpl_slider_args, **mpl_slider_kwargs):
    #     """
    #     Args:
    #         mpl_slider_args: (ax, label, valmin, valmax) """
    #     self.mpl_slider = Slider(*mpl_slider_args, **mpl_slider_kwargs)

    def on_changed(self, func_with_val_and_args, args_opt=()):
        """
        Allows to extend the function to have more than just val as it's argument
        """
        on_changed_func = lambda val, args=args_opt: func_with_val_and_args(val, *args) # this sets fixed references to the args
        # matplotlib.widgets can still call func(val) as before, just that I packed args into it

        Slider.on_changed(self, on_changed_func)

class CParameterSpace:
    def __init__(self, cparams_list):
        """ """
        self.cparams_list = cparams_list

        for cp in self.cparams_list:
            vars(self)[cp.name] = cp

        self._meshgrid = None
        self._make_meshgrid()

    def _make_meshgrid(self):
        """ once the cparams_list is initialized, use numpy's meshgrid to
        create multidimensional arrays of the same shape for each parameter.
        Always make sure that indexing is 'ij', otherwise the dimensions will
        switch around. """
        just_arrays = [cparam.np_arr for cparam in self.cparams_list]
        self._meshgrid = np.meshgrid(*just_arrays, indexing="ij")

    def get_mgf_arr(self, param_name):
        """ a meshgridified array is one of the return values of A, B, _ = np.meshgrid(a, b, ...);
        they all have the same dimensionality so that they can be plugged into a normal mathematical
        python function """
        if self._meshgrid is not None:
            return self._meshgrid[self.get_index_of(param_name)]
        else:
            print("no _meshgrid generated yet")
            exit(1)

    def get_index_of(self, name):
        """ """
        els = []
        for i, cp in enumerate(self.cparams_list):
            if cp.name == name:
                els.append(i)

        if len(els) == 1:
            return els[0]
        else:
            print("parameter contained twice: ", els)
            exit(1)

    def get_param_by_name(self, name):
        """ """
        els = []
        for i, cp in enumerate(self.cparams_list):
            if cp.name == name:
                els.append(cp)

        if len(els) == 1:
            return els[0]
        else:
            print("parameter contained twice: ", els)
            exit(1)

    def get_arr(self, name):
        """ """
        els = []
        for cp in self.cparams_list:
            if cp.name == name:
                els.append(cp)

        if len(els) == 1:
            return els[0].np_arr
        else:
            print("parameter contained twice: ", els)
            exit(1)

    def calc_integral(self, dep_var_mgf,
                      param_to_integrate_over_name):
        """
        Args:
            dep_var_mgf: these are the actual y values
            param_to_integrate_over: name of param to integrate over -> x values
        """

        return np.trapz(dep_var_mgf,
                        x=self.get_arr(param_to_integrate_over_name),
                        axis=self.get_index_of(param_to_integrate_over_name))

    def calc_function(self, f, args_param_names=()):
        """ """
        if len(args_param_names) == 0:
            print("nothing sampled!")
            return None

        # check that all shapes are equal. only then they can be processed correctly by numpy
        shapes = np.array([np.shape(self.get_mgf_arr(name)) for name in args_param_names])
        assert (shapes == shapes[0]).all()

        meshgridifed_arrays = [self.get_mgf_arr(name) for name in args_param_names]
        return f(*meshgridifed_arrays)

    def get_dimension(self):
        """ """
        return len(self.cparams_list)

    def get_param_names(self):
        """ """
        return [cparam.name for cparam in self.cparams_list]

    def _make_sliders(self, indexing_list_indep_vars, ordered_params, ordering_of_params_name_and_value, dep_var_mgf, fig, ax):
        """ for all dimensions > 2, a slider is made """
        self.csliders = []

        if len(ordered_params) <= 2:
            print("no sliders generated, len(ordered_params) <= 2 ")
            return

        # continue here with plotting sliders for the parameters that are not visible in the color plot

        for i, param in enumerate(ordered_params[2:]):

            fig.subplots_adjust(left=0.25, bottom=0.05 + 0.05 * i + 0.2)
            mpl_slider_ax = plt.axes([0.1, 0.05 + 0.05 * i, 0.65, 0.03])

            index_expr = indexing_list_indep_vars[self.get_index_of(param.name)] # either an integer or a slice

            init_val = None
            mpl_slider_kwargs = {}

            if isinstance(index_expr, int) or isinstance(index_expr, np.int64):
                init_val = param.np_arr[int(index_expr)]
                mpl_slider_kwargs["valinit"] = init_val
            elif isinstance(index_expr, slice):
                if isinstance(index_expr.start, int):
                    init_val = param.np_arr[index_expr.start]
                    mpl_slider_kwargs["valinit"] = init_val
                else:
                    print("index_expr.start is not an int : ", index_expr)
            else:
                print("err: index_expr neither an int nor a slice : ", index_expr)
                exit(1)

            # slider = Slider(mpl_slider_ax, param.name, np.min(param.np_arr), np.max(param.np_arr), **mpl_slider_kwargs)

            cslider = CSlider(mpl_slider_ax, param.name, np.min(param.np_arr), np.max(param.np_arr), **mpl_slider_kwargs)

            cslider.on_changed(CParameterSpace.update_func, args_opt=(cslider, self, param, ordering_of_params_name_and_value, dep_var_mgf, fig, ax))

            self.csliders.append(cslider)

            print("making slider of " + param.name + ", with init val: ", init_val)

    # TODO
    @staticmethod
    def update_func(val, cslider, cps, param, ordering_of_params_name_and_value, dep_var_mgf, fig, ax):
        """ """
        if not (param.np_arr == val).any(): # if val is not exactly on a data point
            nearest_idx = find_nearest_idx(param.np_arr, val)
            nearest_val = param.np_arr[nearest_idx]
            # print("resetting slider for ", param.name, " from ", val, " to ", nearest_val)
            cslider.set_val(nearest_val)

        # update the pcolor chart
        updated_ordering_of_params_name_and_value = []

        contains_it = False
        for i, (asked_pname, asked_pvalue) in enumerate(ordering_of_params_name_and_value):
            new_tuple = [asked_pname, asked_pvalue]
            if asked_pname == param.name:
                new_tuple[1] = val
                contains_it = True

            updated_ordering_of_params_name_and_value.append(tuple(new_tuple))

        if contains_it == False: # in case it's not contained yet, it must be added
            updated_ordering_of_params_name_and_value.append(
                (param.name, val))

        indexing_list_indep_vars, ordered_params = cps._get_indexing_list_and_ordered_params(updated_ordering_of_params_name_and_value)

        X, Y, Z = shape_arrays_for_pcolor_plotting(cps, indexing_list_indep_vars, ordered_params, dep_var_mgf)
        c = ax.pcolor(X, Y, Z# , shading="nearest"
        )
        # cbar = fig.colorbar(c, ax=ax)
        # cbar.ax.set_ylabel(z_label, rotation=-90, va="bottom")

        # update the slider to show the actual value of the grid point, not the continuous slider value
        index = indexing_list_indep_vars[cps.get_index_of(param.name)]
        assert isinstance(index, int) or isinstance(index, np.int64)
        grid_value = param.np_arr[index]

        fig.canvas.draw_idle()
        # print("updating slider of " + param.name + ", ", val, "updated_ordering_of_params_name_and_value: ", updated_ordering_of_params_name_and_value)


    def plot(self, dep_var_mgf, ordering_of_params_name_and_value=[],
             fig=None, ax=None, z_label=""):
        """
        Args:
            ordering_of_params_name_and_value: list of tuples (param name, default value)
                                the frist two independent parameters appear on x and y axes of the color plot
                                the others (if specified) appear as sliders in the specified order.
        """

        if ax is None:
            ax = plt.gca()

        if fig is None:
            fig = plt.gcf()

        if self.get_dimension() == 1:
            ax.plot(dep_var_mgf[:],  # just one dimension, i.e. take all independent values of that dimension
                    # if there is only one parameter, it must have index 0
                    self.get_arr(0),
                    "k-")
        elif self.get_dimension() >= 2.:
            # make color plot with self.get_dimension() - 2 sliders below to vary the other parameters
            # this contains in the end an expression like [:, :, :, 2, :] (where ":" is equivalent to slice(None))

            indexing_list_indep_vars, ordered_params = self._get_indexing_list_and_ordered_params(ordering_of_params_name_and_value)

            # if there are more than 2 dimensions (free parameters), plot sliders for the values of the other dimensions
            self._make_sliders(indexing_list_indep_vars, ordered_params, ordering_of_params_name_and_value, dep_var_mgf, fig, ax)

            # plot X, Y, Z data, where X, Y, Z must have the same np.shape() tuple (2d tuple!)
            X, Y, Z = shape_arrays_for_pcolor_plotting(self, indexing_list_indep_vars, ordered_params, dep_var_mgf)

            assert np.shape(X) == np.shape(Y) == np.shape(Z) and len(np.shape(X)) == 2
            c = ax.pcolor(X, Y, Z, # shading="auto"
            )
            cbar = fig.colorbar(c, ax=ax)
            cbar.ax.set_ylabel(z_label, rotation=-90, va="bottom")

            ax.set_xlabel(ordered_params[0].get_label_str())
            ax.set_ylabel(ordered_params[1].get_label_str())

    def _get_indexing_list_and_ordered_params(self, ordering_of_params_name_and_value):
        """
        used in preparation for plotting with pcolor and sliders
        Args:
            ordering_of_params_name_and_value: list of tuples (param name, value); if no specific value put None for value
        Returns:
            indexing_list_indep_vars,  : a list that gets unpacked to be the indexing for the arrays provided by meshgrid
            ordered_params  :
            """

        # import pdb; pdb.set_trace()  # noqa BREAKPOINT
        # always supply what you want to have plotted
        assert len(ordering_of_params_name_and_value) >= 2

        asked_ordering_names = [param_name for param_name, value in ordering_of_params_name_and_value]
        asked_ordering_values = [value for param_name, value in ordering_of_params_name_and_value]

        # check if the param names are correct
        assert (False not in [param_name in self.get_param_names()
                              for param_name, value in ordering_of_params_name_and_value])

        ordered_param_names = []
        indexing_list_indep_vars = [None] * self.get_dimension()

        # the first two are always explicitly given
        for j, (asked_pname, asked_def_pvalue) in enumerate(ordering_of_params_name_and_value[:2]):
            ordered_param_names.append(asked_pname)
            indexing_list_indep_vars[self.get_index_of(asked_pname)] = slice(None)


        if len(ordering_of_params_name_and_value) > 2:
            for j, (asked_pname, asked_def_pvalue) in enumerate(ordering_of_params_name_and_value[2:]):
                ordered_param_names.append(asked_pname)

                # the following ones that are given have an initial value of interest
                if asked_def_pvalue is not None:
                    # get the closest calculated point to the asked default value

                    nearest_idx = find_nearest_idx(self.get_arr(asked_pname), asked_def_pvalue)
                    nearest_val = self.get_arr(asked_pname)[nearest_idx]
                    indexing_list_indep_vars[self.get_index_of(asked_pname)] = nearest_idx

                    # print("finding nearest value to", asked_pname, ": provided value : ", asked_def_pvalue,
                    #       ", nearest idx: ", nearest_idx, ", nearest value : ", nearest_val)
                else:
                    # no default value supplied -> just get the 0th index
                    indexing_list_indep_vars[self.get_index_of(asked_pname)] = 0

        # complete the ordering of not explicitly given params in second pass
        for cparam in self.cparams_list:
            if cparam.name not in ordered_param_names:
                ordered_param_names.append(cparam.name)
                indexing_list_indep_vars[self.get_index_of(cparam.name)] = 0

        ordered_params = [self.get_param_by_name(param_name) for param_name in ordered_param_names]

        return indexing_list_indep_vars, ordered_params
