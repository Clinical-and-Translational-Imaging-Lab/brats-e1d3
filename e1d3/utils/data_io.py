import os
import numpy as np
import nibabel as nib


class DataIO:
    """
    Main class for handling loading/saving of data/label volumes.
    """

    def __init__(self, config, train_validate_test):
        """
        config is a dictionary obtained by parsing a configuration file.
        Args:
            config: 'dictionary' containing configuration parameters parsed
                    from a configuration file.
            train_validate_test: 'string', one of 'train', 'validate' and 'test'.
        """
        config_data = config['data']
        config_network = config['network']
        if train_validate_test not in ['train', 'validate', 'test']:
            raise Exception('Incorrect DataIO mode selected:{}'.format(train_validate_test))
        switch_dict_dataset = {
            'train': config_data.get('data_directory_train'),
            'validate': config_data.get('data_directory_validate'),
            'test': config_data.get('data_directory_test')
        }
        self.data_directory = switch_dict_dataset.get(train_validate_test, None)
        assert (os.path.isdir(self.data_directory))
        self.channels = config_data.get('channels')
        self.weight_mask_channel = config_data.get('weight_mask_channel', 'flair_mask')
        self.seg_file_suffix = config_data.get('seg_file_suffix')
        if train_validate_test == 'test':  # for testing routine only
            self.save_directory = config_data.get('save_directory_test', None)
            self.model_run_time = config_network.get('model_load_config', None)[0]

    def patients_list(self):
        """
        Helper for geting list of patients/folders in 'data_directory' folder.
        Returns:
            'list' containing names of patients.
        """
        patients_list = os.listdir(self.data_directory)
        patients_list = [name for name in patients_list if os.path.isdir(os.path.join(self.data_directory, name))]
        return sorted(patients_list)

    def load_patient(self, patient_id):
        """
        Loads all volumes (data, label, weight) of one patient.
        Primary use of this routine is in training/validation.
        Args:
            patient_id: Patient Name.
        Returns:
            'np.array' of data volumes (all modes), 'np.array' of weight volume,
            'np.array' of label volume.
        """
        data = []
        for mode in self.channels:
            volume = self.load_volume(patient_id, mode, with_info=False)  # Load Modality
            data.append(volume)
        data = np.stack(data, axis=0)  # One 4D volume (Channels, D, H, W)
        weight = self.load_volume(patient_id, self.weight_mask_channel)
        label = self.load_volume(patient_id, self.seg_file_suffix)  # Load Label
        return data, weight, np.uint8(label)

    def load_patient_npy(self, patient_id):
        """
        Loads all volumes (data, label, weight) of one patient.
        Primary use of this routine is in training/validation.
        Args:
            patient_id: Patient Name.
        Returns:
            'np.array' of data volumes (all modes), 'np.array' of weight volume,
            'np.array' of label volume.
        """
        patient_filepath = os.path.join(self.data_directory, patient_id, patient_id + '_{}.npy')
        data = np.load(patient_filepath.format('data'), mmap_mode='r')
        weight = np.load(patient_filepath.format('weight'), mmap_mode='r')
        label = np.load(patient_filepath.format('label'), mmap_mode='r')
        return data, weight, label

    def load_patient_with_info(self, patient_id, with_label=False):
        """
        Loads all volumes (data, label, weight) of one patient.
        (Same as 'load_patient()', plus returns 'affine' and 'header').
        Primary use of this routine is in testing.
        Args:
            patient_id: Patient Name.
            with_label: flag for returning label volume, default:'False'.
        Returns:
            'np.array' of data volume (all modes), 'np.array' of weight volume,
            'list' of '[affines, headers]' of each data volume,
            'np.array' of label volume (optional)
        """
        data, affines, headers = [], [], []
        for mode in self.channels:
            volume, affine, header = self.load_volume(patient_id, mode, with_info=True)  # Load Modality
            data.append(volume)
            affines.append(affine)
            headers.append(header)
        data = np.stack(data, axis=0)  # One 4D volume (Channels, D, H, W)
        weight = self.load_volume(patient_id, self.weight_mask_channel)
        if not with_label:
            return data, weight, [affines, headers]
        label = self.load_volume(patient_id, self.seg_file_suffix)  # Load Label
        return data, weight, [affines, headers], np.uint16(label)

    def load_volume(self, patient, mode, with_info=False):
        """
        Loads single '.nii.gz' volume (data/label/...) of one patient.
        e.g. file_name = 'Brats18_2013_3_1\\Brats18_2013_3_1_flair.nii.gz'
        *NOTE*: loading weight maps is not considered yet.
        Args:
            patient: Patient name.
            mode: Suffix for patient volume file name.
            with_info: flag for returning 'affine' and 'header' of volume (default:'False')
        Returns:
            'np.array' of image volume, 'np.array' of image affine (optional),
            format-specific image header object
        """
        file_name = '{0}_{1}.nii.gz'.format(patient, mode)  # hardcoded format, file name should follow
        file_path = os.path.join(self.data_directory, patient, file_name)
        image = nib.load(file_path)
        image_array = image.get_data().astype(np.float32)
        image.uncache()  # release cache memory
        if with_info:
            return image_array, image.affine, image.header
        else:
            return image_array

    def save_volume(self, volume, affine, patient, volume_type):
        """
        Saves volume at specified directory, with 'affine' provided.
        Directory is created if it does not exist.
        Args:
            volume: Volume to save.
            affine: Image affine matrix.
            patient: Patient name.
            volume_type: one of 'seg' (segmentation map) and 'prob' (probability map).
        Returns:
            'None'.
        """
        assert volume_type.lower() in ['seg', 'prob']
        # create save directory folder if it does not exist
        save_path = os.path.join(self.save_directory, self.model_run_time, patient)
        if not os.path.exists(save_path):
            print(f"Path {save_path} does not exist. Creating...")
            os.makedirs(save_path)
        file_name = '{0}_{1}.nii.gz'.format(patient, volume_type.lower())
        file_path = os.path.join(save_path, file_name)
        nib.save(nib.Nifti1Image(volume, affine), file_path)
