import numpy as np
import torch

try:
    from utils.data_io import DataIO
except ImportError:
    print('reading from `data_io.py` in current directory')
    from data_io import DataIO
except:
    print('`data_io.py` not found')


class DatasetMMEP3d(torch.utils.data.Dataset):
    """
    Batch Generator Class
    Step1: Load all Data in one big array
    Step2: Extract required segments from each class
    Step3: Exhaust the dataset (segments) once every epoch
    """

    def __init__(self, config, train_or_validate, augment=None):
        """
        Args:
            data_array: (np.array)
            label_array: (np.array)
            config: (dict) parsed config_file
            train_or_validate: (str)
        """
        threads_multi = config['train'].get('workers_multithreading', 1)
        # Parameters for Training/Validation
        if not train_or_validate in ['train', 'validate']:
            raise Exception('Incorrect Generator mode selected:{}'.format(
                train_or_validate))
        config_train_or_validate = config[train_or_validate]
        self.train_or_validate = train_or_validate
        # Network parameters
        config_net = config['network']
        self.data_shape = config_net.get('data_shape', None)  # should be list of len 3 (never > 3)
        self.label_shape = config_net.get('label_shape', None)
        assert len(self.data_shape) == len(self.label_shape)
        assert len(self.data_shape) == 3

        # Data specific parameters
        config_data = config['data']
        self.class_labels = config_data.get('class_labels', None)
        self.data_io_obj = DataIO(config, train_or_validate)

        # Training/Validation parameters
        self.segments_per_epoch = config_train_or_validate.get('segments_per_epoch', None)
        self.batch_size = config_train_or_validate.get('batch_size', 128)
        self.batches_per_epoch = np.ceil(self.segments_per_epoch / self.batch_size)
        # ceiling implies one extra smaller mini-batch
        print('Each epoch consists of `{}` batches during {}'.format(
            int(self.batches_per_epoch), train_or_validate.upper()))

        self.patients_list = self.data_io_obj.patients_list()
        self.augment = augment

    def __len__(self):
        """return # of segments to iterate over per epoch (for training/validation loop)"""
        return self.segments_per_epoch

    def on_epoch_begin(self):
        """Execute this function before every epoch"""

    def __getitem__(self, index):
        """
        generates a (data, label) pair on every call.
        The pairs are collated into a batch by a `torch.utils.data.DataLoader`
        """
        # randomly choose a patient
        patient_id = self.patients_list[index % len(self.patients_list)]  # np.random.choice(self.patients_list) #
        # read patient volumes
        data, weight, label = self.data_io_obj.load_patient_npy(patient_id)
        data, label = self.__get_random_sampled_data_label(data, weight, label)
        # perform distortion
        if self.augment and (torch.randn(1) > 0):
            # augmentation function returns a pytorch tensor by default
            # augmentation function requires all inputs to be 4D
            data, label = self.augment(**{
                'data': data.copy(),
                'label': np.expand_dims(label, 0)
            })
            return data, label.squeeze(0).long()
        else:
            return torch.from_numpy(data), torch.from_numpy(label)

    def get_patient_volumes(self, index):
        patient_id = self.patients_list[index]
        return self.data_io_obj.load_patient_npy(patient_id)

    def __get_random_sampled_data_label(self, data, weight, label):
        """"""
        # get sampling mask from 'FLAIR' volume, with help from weight mask.
        sampling_mask = self.__get_sampling_mask(weight)  # np.uint8(label>0))
        # select a center voxel randomly from the sampling mask.
        valid_centers = np.flatnonzero(sampling_mask)
        if len(valid_centers) > 0:
            selected_center = np.random.choice(valid_centers, 1, replace=False)
            center = np.asarray(np.unravel_index(selected_center[0], label.shape), dtype=np.uint16).T
        else:  # for no candidate voxel (sample from center)
            center = [k // 2 for k in weight.shape]
        # extract data and label volume around the selected center
        data_segment, label_segment = self.__extract_data_label_segment(data, label, center)
        return data_segment, label_segment

    def __get_sampling_mask(self, weight_mask=None):
        """
        Sampling mask to ensure extracted segment stays inside the input region
        and covers the desired regions only.
        """
        sampling_mask = np.zeros_like(weight_mask)
        sampling_mask[
        int(np.floor(self.label_shape[0] / 2)):int(sampling_mask.shape[0] - np.floor(self.label_shape[0] / 2)),
        int(np.floor(self.label_shape[1] / 2)):int(sampling_mask.shape[1] - np.floor(self.label_shape[1] / 2)),
        int(np.floor(self.label_shape[2] / 2)):int(sampling_mask.shape[2] - np.floor(self.label_shape[2] / 2))
        ] = 1
        return sampling_mask * weight_mask

    def __extract_data_label_segment(self, data_array, label_array, center):
        """
        return a (data, label) pair of desired shape, centered around the
        provided `center` pixel.
        *NOTE*: weight maps not used yet. Returned label should be of type uint8.
        I tried uint16 and it yielded an error of `unsupported data type`.
        Format of input arguments is as follows:
        data_array: (C, H, W, D)
        label_array: (H, W, D)
        center = (H, W, D)
        """
        data = self.__extract_segment_from_volume(data_array, self.data_shape, center, has_channels=True)
        label = self.__extract_segment_from_volume(label_array, self.label_shape, center, has_channels=False)
        return data, np.uint8(label)

    @staticmethod
    def __extract_segment_from_volume(volume, segment_shape, center, has_channels=False):
        """
        This module extracts a segment centered around `center` and of
        `segment_shape` from `volume`.
        if center is provided such that segment-to-extract has regions outside
        the input volume, appropriate zero-padding will be done to compensate.
        """
        offset = 1 if has_channels else 0  # offset to ignore indexing channel dimension
        assert all([0 <= center[i] < volume.shape[i + offset] for i in range(len(center))])
        # empty arrays to fill later.
        indices_data = np.zeros((len(segment_shape) + offset, 2), dtype=np.uint16)
        paddings_data = np.zeros((len(segment_shape) + offset, 2), dtype=np.uint16).tolist()

        if has_channels:
            indices_data[0] = [0, volume.shape[0]]  # take all along channel axis
            paddings_data[0] = [0, 0]  # no padding along channel axis
        # calculate left and right bounds for indices, and corresponding padding
        # I usually comment out PADDING for code testing.
        for i in range(len(segment_shape)):
            indices_data[i + offset, 0] = int(max(0, center[i] - np.floor(segment_shape[i] / 2)))
            indices_data[i + offset, 1] = int(min(volume.shape[i + offset], center[i] - np.floor(segment_shape[i] / 2)
                                                  + segment_shape[i]))
            paddings_data[i + offset][0] = int(np.abs(min(0, center[i] - np.floor(segment_shape[i] / 2))))
            paddings_data[i + offset][1] = int(max(volume.shape[i + offset],
                                                   np.ceil(center[i] - np.floor(segment_shape[i] / 2) +
                                                           segment_shape[i])) - volume.shape[i + offset])
        indices_list_data = [range(indices_data[i][0], indices_data[i][1]) for i in range(indices_data.shape[0])]
        volume_to_return = volume[np.ix_(*indices_list_data)]
        volume_to_return = np.pad(volume_to_return, pad_width=paddings_data, mode='constant', constant_values=0)
        return volume_to_return
