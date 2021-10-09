import os

import numpy as np
import nibabel as nib

from utils.data_io import DataIO


class DataProcessNumpy(object):
    """
    Convert dataset into (cropped) `.npy` volumes for training/validation.
    """

    def __init__(self, data_directory, save_directory):
        if save_directory is not None:
            save_directory = data_directory

        for folder in [data_directory, save_directory]:
            assert os.path.exists(folder), f"Path `{folder}` does not exist"
            assert os.path.exists(folder), f"Path `{folder}` does not exist"

        self.data_directory = data_directory
        self.save_directory = save_directory
        self.channels = ['flair_norm', 't1_norm', 't1ce_norm', 't2_norm']
        self.weight_mask_channel = 'mask'
        self.seg_file_suffix = 'seg'

        self.data_io = DataIO

    @staticmethod
    def __get_non_zero_bounding_box(volume):
        """
        Return bounding box of non-zero region of input `volume`
        """
        ind_tuple = volume.nonzero()
        ind_limits = np.zeros((len(volume.shape), 2), dtype=np.uint16)
        for i in range(len(volume.shape)):
            ind_limits[i, 0], ind_limits[i, 1] = min(ind_tuple[i]), max(ind_tuple[i]) + 1
        ind_limits_list = [range(ind_limits[i][0], ind_limits[i][1]) for i in range(ind_limits.shape[0])]
        return ind_limits_list

    @staticmethod
    def __extract_volume(volume, ind_limits_list):
        """
        Returns a rectangular crop of the input `volume`
        """
        return volume[np.ix_(*ind_limits_list)]

    def process(self):
        """"""
        patients = self.patients_list()
        for p in patients:
            print('Patient:', p)
            data, weight, label = self.load_patient(p)
            ind_limits_list = self.__get_non_zero_bounding_box(weight)
            # extract the non-zero bounding box from data, weight, and label.
            data = self.__extract_volume(data, [range(0, 4)] + ind_limits_list)
            weight = self.__extract_volume(weight, ind_limits_list)
            label = self.__extract_volume(label, ind_limits_list)
            # save npy
            for volume, suffix in zip([data, weight, label], ['data', 'weight', 'label']):
                self.save_npy(volume, p, suffix)
                print('\tSaved:', suffix)

    def patients_list(self):
        """
        Helper for getting list of patients/folders in 'data_directory' folder.
        Returns:
            'list' containing names of patients.
        """
        patients_list = os.listdir(self.data_directory)
        patients_list = [name for name in patients_list if 'brats' in name.lower()]
        return patients_list

    def load_patient(self, patient_id):
        data = []
        for mode in self.channels:
            volume, affine, header = self.load_volume(patient_id, mode, dtype=np.float32, with_info=True)  # Modality
            data.append(volume)
        data = np.stack(data, axis=0)  # One 4D volume (Channels, D, H, W)
        weight = self.load_volume(patient_id, self.weight_mask_channel, dtype=np.bool_)  # Mask (binary)
        label = self.load_volume(patient_id, self.seg_file_suffix, dtype=np.uint8)  # Label
        return data, weight, label

    def load_volume(self, patient, mode, dtype=np.float32, with_info=False):
        file_name = '{0}_{1}.nii.gz'.format(patient, mode)
        file_path = os.path.join(self.data_directory, patient, file_name)
        image = nib.load(file_path)
        image_array = image.get_data().astype(dtype)
        image.uncache()  # release cache memory
        if with_info:
            return image_array, image.affine, image.header
        else:
            return image_array

    def save_npy(self, volume, patient, suffix):
        save_path = os.path.join(self.save_directory, patient)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = '{0}_{1}.npy'.format(patient, suffix)
        file_path = os.path.join(save_path, file_name)
        np.save(file_path, volume)


if __name__ == '__main__':
    """
    Run as:
    python data_crop_npy.py --src_folder <path_to_src_folder> --dst_folder <path_to_dst_folder>
    """
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess BraTS data.')
    parser.add_argument('--src_folder', type=str, required=True, help='path to dataset')
    parser.add_argument('--dst_folder', type=str, required=False, default=None, help='path to dataset')
    args = parser.parse_args()

    processor = DataProcessNumpy(data_directory=args.src_folder, save_directory=args.dst_folder)
    processor.process()
