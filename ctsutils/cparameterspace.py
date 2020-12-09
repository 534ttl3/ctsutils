import numpy as np
import matplotlib.pyplot as plt


def get_values_from_meshgrid(mg, np_bool_array):
    """ """
    _ = np.where(np_bool_array)
    return list((map(lambda n: (mg[n][np_bool_array]).item(), range(len(mg)))))


def get_indices_from_np_where_quer(np_where_query):
    """ """
    return tuple(map(lambda el: np.int(el), np.where(Ts == np.max(Ts))))

class CParam:
    def __init__(self, name, np_arr):
        """ """
        self.name = name
        self.np_arr = np_arr

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

    def calc_integral(self, dep_var_meshgridifed,
                      param_to_integrate_over_name):
        """
        Args:
            dep_var_meshgridifed: these are the actual y values
            param_to_integrate_over: name of param to integrate over -> x values
        """

        return np.trapz(dep_var_meshgridifed,
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

    def plot(self, dep_var_meshgridifed, ordering_of_params_names=(),
             fig=None, ax=None, z_label=""):
        """
        Args:
            ordering_of_params_names: the frist two independent parameters appear on x and y axes of the color plot
                                the others (if specified) appear as sliders in the specified order.
        """

        if ax is None:
            ax = plt.gca()

        if fig is None:
            fig = plt.gcf()

        if self.get_dimension() == 1:
            ax.plot(dep_var_meshgridifed[:],  # just one dimension, i.e. take all independent values of that dimension
                    # if there is only one parameter, it must have index 0
                    self.get_arr(0),
                    "k-")
        elif self.get_dimension() >= 2.:
            # make color plot with self.get_dimension() - 2 sliders below to vary the other parameters
            # this contains in the end an expression like [:, :, :, 2, :] (where ":" is equivalent to slice(None))
            indexing_list_indep_vars = []

            # check if the param names are correct
            assert (False not in [param_name in self.get_param_names()
                                  for param_name in list(ordering_of_params_names)])

            params_cp = self.cparams_list.copy()

            ordered_param_names = []

            for param in self.cparams_list:
                if param.name in list(ordering_of_params_names):
                    indexing_list_indep_vars.append(slice(None))
                    ordered_param_names.append(param.name)
                else:
                    # TODO: make the index of the fixed values selectable from the start
                    indexing_list_indep_vars.append(0)


            # complete the ordering of not explicitly given params in second pass
            for param in self.cparams_list:
                if param.name not in list(ordered_param_names):
                    ordered_param_names.append(param.name)

            X = self.get_mgf_arr(ordered_param_names[0])[
                tuple(indexing_list_indep_vars)]
            Y = self.get_mgf_arr(ordered_param_names[1])[
                tuple(indexing_list_indep_vars)]
            Z = None

            if (np.shape(dep_var_meshgridifed) == np.shape(X) and
                np.shape(dep_var_meshgridifed) == np.shape(Y)):
                Z = dep_var_meshgridifed
                # delivered in the appropriate shape to be plotted (e.g. after integrating out one dimension, i.e.
                # function with signature (N^d x N^d x ...) -> N^(d-1) evaluated and passed to this plotting function
                # in the right shape)
                print("plot: Z is already in the right shape")
            else:
                Z = dep_var_meshgridifed[tuple(indexing_list_indep_vars)] # function with signature (N^d x N^d x ...) -> N^d evaluated

            # plot X, Y, Z data, where X, Y, Z must have the same np.shape() tuple (2d tuple!)
            assert np.shape(X) == np.shape(Y) == np.shape(Z) and len(np.shape(X)) == 2
            c = ax.pcolor(X, Y, Z, shading="auto")
            cbar = fig.colorbar(c, ax=ax)
            cbar.ax.set_ylabel(z_label, rotation=-90, va="bottom")

            ax.set_xlabel(ordered_param_names[0])
            ax.set_ylabel(ordered_param_names[1])
