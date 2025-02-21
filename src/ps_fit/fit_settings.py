import numpy as np

class FitSettings:
    """A class that handles converting between parameter values and an array that can be fed into a minimizer"""
    def __init__(self, fixed_qu, fixed_position, fixed_blur, fixed_bg, num_datas):
        self.fixed_qu = fixed_qu
        if fixed_position is None:
            self.fixed_position = None
        else:
            self.fixed_position = np.array(fixed_position)
            assert(self.fixed_position.shape[0] == 2)
            if self.fixed_position.ndim == 2:
                assert(self.fixed_position.shape[1] == num_datas)
        self.fixed_blur = fixed_blur
        self.fixed_bg = fixed_bg
        self.num_datas = num_datas

    def param_to_value(self, params, param, data_index=0):
        """
        Gives the array index of param.
        # Arguments
        * params: list of values
        * param: either "q", "u", "x", "y", "sigma", or "bg"
        * data_index: indicates the index of the data set for which the parameter will be returned.
        # Returns
        * The value of the parameter
        """
        index = self.param_to_index(param, data_index)
        if index is None:
            if param == "q":
                return self.fixed_qu[0]
            if param == "u":
                return self.fixed_qu[1]
            if param == "x":
                if self.fixed_position.ndim == 1:
                    return self.fixed_position[0]
                else:
                    return self.fixed_position[data_index][0]
            if param == "y":
                if self.fixed_position.ndim == 1:
                    return self.fixed_position[1]
                else:
                    return self.fixed_position[data_index][1]
            if param == "bg":
                return self.fixed_bg[0]
            if param == "sigma":
                return self.fixed_blur[1]
            raise Exception(f"Could not recognize parameter {param}")
        else:
            return params[index]

    def param_to_index(self, param, data_index=0):
        """
        Gives the array index of param.
        # Arguments
        * param: either "q", "u", "x", "y", "sigma", or "bg"
        * data_index: indicates the index of the data set for which the parameter will be returned.
        Required for some parameters.
        # Returns
        * The index of the parameter, if it exists. Otherwise None
        """
        index = 0

        if param == "q":
           if self.fixed_qu is None: return index
           else: return None
        if param == "u":
           if self.fixed_qu is None: return index+1
           else: return None
        if self.fixed_qu is None: index += 2

        if param == "x":
           if self.fixed_position is None: return index + 2*data_index
           else: return None
        if param == "y":
           if self.fixed_position is None: return index+1 + 2*data_index
           else: return None
        if self.fixed_position is None: index += 2*self.num_datas

        if param == "sigma":
           if self.fixed_blur is None: return index
           else: return None
        if self.fixed_blur is None: index += 1

        if param == "bg":
            if self.fixed_bg is None: return index
            else: return None
        if self.fixed_bg is None: index += 1
        
        raise Exception(f"Could not recognize parameter {param}")


    def index_to_param(self, index):
        """
        Returns the parameter name and the index of the data set it belongs to. If the parameter is
        global, the index will be None
        """
        if self.fixed_qu is None:
            if index == 0: return "q", None
            elif index == 1: return "u", None
            else: index -= 2
        for data_index in range(self.num_datas):
            if self.fixed_position is None:
                if index == 0: return "x", data_index
                elif index == 1: return "y", data_index
                else: index -= 2
        if self.fixed_blur is None:
            if index == 0: return "sigma", None
            else: index -= 1
        if self.fixed_bg is None:
            if index == 0: return "bg", None
            else: index -= 1

        raise Exception(f"Could not recognize index {index}")

    def length(self):
        length = 0
        if self.fixed_qu is None:
            length += 2
        if self.fixed_position is None:
            length += 2*self.num_datas
        if self.fixed_blur is None:
            length += 1
        if self.fixed_bg is None:
            length += 1
        return length