import os

import numpy as np
import nibabel as nib


def preprocess_in_directory(directory):
    """ Normalize & save (in the same directory) data and non-zero masks. """
    assert os.path.exists(args.folder), f"Path `{args.folder}` does not exist"
    patients = os.listdir(directory)

    for p in patients:
        files = os.listdir(os.path.join(directory, p))
        for mod in ['flair', 't1', 't1ce', 't2']:
            img, affine = load_volume(os.path.join(directory, p, p + '_' + mod + '.nii.gz'))
            img, mask = mean_var_norm(img)
            save_volume(img, affine, directory, p, mod + '_norm')
            if mod == 'flair':
                save_volume(mask, affine, directory, p, 'mask')


def mean_var_norm(volume):
    mask = volume > 0
    pixels = volume[mask]
    mean = np.nanmean(pixels)
    std = np.nanstd(pixels)
    volume = (volume - mean) / std
    volume *= mask
    return volume, np.bool_(mask)


def load_volume(load_file_name):
    image = nib.load(load_file_name)
    image_array = image.get_data().astype(np.float32)
    image.uncache()
    return image_array, image.affine


def save_volume(volume, affine, directory, patient, suffix):
    file_name = '{0}_{1}.nii.gz'.format(patient, suffix)
    file_path = os.path.join(directory, patient, file_name)
    nib.save(nib.Nifti1Image(volume, affine), file_path)


if __name__ == '__main__':
    """
    Run as:
    python data_preprocess.py --data_dir <path_to_data_folder>
    """
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess BraTS data.')
    parser.add_argument('--data_dir', type=str, required=True, help='path to dataset directory.')
    args = parser.parse_args()

    preprocess_in_directory(directory=args.data_dir)
