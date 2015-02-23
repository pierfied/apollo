import numpy as np

class SciLoader(object):
    """
    Class to load and manipulate RAJA loop samples data.

    Data is loaded from a CSV file to a numpy record array.  Other methods in
    this class are helpers to extract X and y vectors from this array.
    """
    def __init__(self, filename, dtype):
        self.__labels = {}
        self.__dtype = dtype

        with open(filename, 'r') as data_file:
            self.__data = np.loadtxt(
                    data_file, 
                    dtype=np.dtype(self.__dtype), 
                    delimiter=',')

    def get_data(self):
        """
        Get full numpy record array of loaded data.

        Returns:
            A numpy array containing all loaded data. The array will have field
            names specified by the dtype used to load the data.
        """
        return self.__data

    def get_x(dtype):
        x_data = np.empty(((len(data))), dtype=np.dtype(dtype))

        for i in range(len(data)):
            for name in x_data.dtype.names:
                if name == 'best':
                    if labels(data, i) == get_label(data, data[i]['loop'], data[i]['range size']):
                        x_data[i]['best'] = 1
                    else:
                        x_data[i]['best'] = 0
                else:
                    x_data[i][name] = data[i][name]

        return x_data

    def get_y(data, dtype):
        y_data = np.empty((len(data)), dtype=np.dtype(dtype))

        for i in range(len(data)):
            mylabel = get_label(data, data[i]['loop'], data[i]['range size'])

            for name in y_data.dtype.names:
                if name == 'outer':
                    y_data[i][name] = mylabel[0]
                elif name == 'inner':
                    y_data[i][name] = mylabel[1]
                else:
                    y_data[i][name] = data[i][name]

        return y_data

    def get_label(data, loop, size):
        if not (loop, size) in _labels:
            data = data[np.logical_and(data['loop'] == loop, data['range size'] == size)]
            label_index = np.argmin(data['time'])
            _labels[(loop, size)] = (data['outer'][label_index], data['inner'][label_index])

        return _labels[(loop, size)]

    def labels(data, index):
        return (data['outer'][index], data['inner'][index])
