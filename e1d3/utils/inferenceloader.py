import numpy as np

try:
    from utils.data_io import DataIO
except ImportError:
    print('reading from `data_io.py` in current directory')
    from data_io import DataIO
except:
    print('`data_io.py` not found')


class DatasetInference3d:
    """
    Helper class for full volume inference during testing
    """

    def __init__(self, config):
        config_data = config['data']
        self.num_classes = config_data.get('num_classes')
        self.channels = config_data.get('channels', None)
        self.data_io_obj = DataIO(config, 'test')
        self.patients_list = self.data_io_obj.patients_list()
        self.num_patients = len(self.patients_list)

        config_net = config['network']
        self.data_shape = config_net.get('data_shape', None)
        self.label_shape = config_net.get('label_shape', None)

        config_test = config['test']
        self.batch_size = config_test.get('batch_size')
        self.segment_overlap = config_test.get('segment_overlap', None)
        assert (self.segment_overlap >= 0) and (self.segment_overlap < 1)
        # the segment stride is the `label_shape` scaled by `(1-segment_overlap)`

    def get_patients_list(self):
        """return list of all patients (for main function)"""
        return self.patients_list

    def get_patient(self, patient_id):
        """"""
        data, weight, info = self.data_io_obj.load_patient_with_info(patient_id, with_label=False)
        return data, info, weight

    def calculate_number_of_steps(self, weight):
        """also update center coordinates, does not update initial weight matrix"""
        self.center_coords = self.__calculate_segments(weight, self.segment_overlap)
        self.batches_test = int(np.ceil(len(self.center_coords) / self.batch_size))

    def __len__(self):
        """return number of iterations for current patient"""
        return self.batches_test

    def generator_function(self, data):
        """
        generate batch of (data segment, center voxel coordinates) pairs on every call
        """
        for i in range(self.batches_test):
            batch_data, batch_coords, batch_padding = [], [], []
            batch_indices = self.center_coords[i * self.batch_size: min((i + 1) * self.batch_size,
                                                                        len(self.center_coords))]

            for center in batch_indices:
                segment_data, indices_list, padding_list = self.__extract_segment_from_volume(data,
                                                                                              self.data_shape,
                                                                                              self.label_shape,
                                                                                              center)

                batch_data.append(segment_data)
                batch_coords.append(indices_list)
                batch_padding.append(padding_list)
            yield np.stack(batch_data, axis=0), batch_coords, batch_padding

    def __calculate_segments(self, mask_volume, segment_overlap):
        """
        calculate center indices for each patch to extract.
        """
        coordinates_list = []
        for i in range(len(mask_volume.shape)):
            # Note: better safe than sorry :)
            start_range = int(np.floor(self.label_shape[i] / 2))
            stride = int(self.label_shape[i] * (1 - segment_overlap))
            assert stride > 0
            end_range = stride * np.ceil(mask_volume.shape[i] / stride)
            coordinates_i = np.arange(start_range, end_range, stride)
            # using int (or flooring) would ensure that no voxel gets missed
            # rather some voxels may get double counted
            coordinates_list += [coordinates_i]

        coordinates_grid = [(x, y, z) for x in coordinates_list[0] \
                            for y in coordinates_list[1] \
                            for z in coordinates_list[2]]
        return coordinates_grid  # list of (X, Y, Z)

    def __extract_segment_from_volume(self, volume, segment_shape, label_shape, center):
        """
        This extraction module is defined for inputs in `channels-first` format
        `segment_shape` is required to extract the segment for input to the network.
        `label_shape` is required to calculate indices where prediction from the network
        is assigned to in the label map.
        """
        offset = 1  # I will be extracting data samples only
        indices_data = np.zeros((len(segment_shape) + 1, 2), dtype=np.uint16)
        paddings_data = np.zeros((len(segment_shape) + 1, 2), dtype=np.int16)
        indices_label = np.zeros((len(label_shape) + 1, 2), dtype=np.uint16)
        paddings_label = np.zeros((len(label_shape) + 1, 2), dtype=np.int16)

        # specifying fields for channel (first) dimension in data.
        indices_data[0] = [0, volume.shape[0]]
        paddings_data[0] = [0, 0]  # no padding along channel axis
        indices_label[0] = [0, self.num_classes]
        paddings_label[0] = [0, 0]  # no padding along channel axis

        # calculate left and right bounds for indices, and corresponding padding
        for i in range(len(segment_shape)):  # (H, W, D)
            indices_data[i + offset][0] = int(max(0, center[i] - np.floor(segment_shape[i] / 2)))
            indices_data[i + offset][1] = int(min(volume.shape[i + offset], center[i] - np.floor(segment_shape[i] / 2) +
                                                  segment_shape[i]))

            indices_label[i + offset][0] = int(max(0, center[i] - np.floor(label_shape[i] / 2)))
            indices_label[i + offset][1] = int(min(volume.shape[i + offset], center[i] - np.floor(label_shape[i] / 2) +
                                                   label_shape[i]))

            paddings_data[i + offset][0] = int(np.abs(min(0, center[i] - np.floor(segment_shape[i] / 2))))
            paddings_data[i + offset][1] = int(max(volume.shape[i + offset],
                                                   np.ceil(center[i] - np.floor(segment_shape[i] / 2) +
                                                           segment_shape[i])) - volume.shape[i + offset])

            paddings_label[i + offset][0] = int(np.abs(min(0, center[i] - np.floor(label_shape[i] / 2))))
            paddings_label[i + offset][1] = int(max(volume.shape[i + offset],
                                                    np.ceil(center[i] - np.floor(label_shape[i] / 2) +
                                                            label_shape[i])) - volume.shape[i + offset])

        # converting to explicit list of coordinates for np.ix_()
        indices_list_data = [range(indices_data[i][0], indices_data[i][1]) for i in range(indices_data.shape[0])]

        volume_to_return = volume[np.ix_(*indices_list_data)]
        volume_to_return = np.pad(volume_to_return, pad_width=paddings_data, mode='constant', constant_values=0)

        # coordinates and paddings data for label segment.
        indices_list_label = [range(indices_label[i][0], indices_label[i][1]) for i in range(indices_label.shape[0])]
        label_undo_shape = [self.num_classes] + label_shape
        paddings_list_label = [range(paddings_label[i][0], label_undo_shape[i] - paddings_label[i][1])
                               for i in range(paddings_label.shape[0])]

        return volume_to_return, indices_list_label, paddings_list_label

    def save_volume(self, volume, affine, patient, volume_type):
        """
        call to 'save_volume' function of 'DataIO' class
        """
        self.data_io_obj.save_volume(volume, affine, patient, volume_type)
