# E<sub>1</sub>D<sub>3</sub> U-Net for Brain Tumor Segmentation

This repository contains official source code for the method proposed in: [E<sub>1</sub>D<sub>3</sub> U-Net for Brain Tumor
Segmentation: Submission to the RSNA-ASNR-MICCAI BraTS 2021 Challenge](
https://arxiv.org/abs/2110.02519).

## Data Preparation:

Download the BraTS dataset.
The structure of the dataset should be as follows:

    dataset/
        - BraTS2021_AAAAA/
            BraTS2021_AAAAA_flair.nii.gz
            BraTS2021_AAAAA_t1.nii.gz
            BraTS2021_AAAAA_t1ce.nii.gz
            BraTS2021_AAAAA_t2.nii.gz
            BraTS2021_AAAAA_seg.nii.gz
        - BraTS2021_AAAAB/
            BraTS2021_AAAAB_flair.nii.gz
            BraTS2021_AAAAB_t1.nii.gz
            BraTS2021_AAAAB_t1ce.nii.gz
            BraTS2021_AAAAB_t2.nii.gz
            BraTS2021_AAAAB_seg.nii.gz
        ...
(The above uses nomenclature from BraTS 2021 dataset for the sake of demonstration.
The dataset directory should have a _folder_ for each subject, where each folder has files with nomenclature `{foldername}_{modality}.nii.gz`.
See [`data_io.py`](https://github.com/Clinical-and-Translational-Imaging-Lab/brats-e1d3/blob/main/e1d3/utils/data_io.py))


Preprocess the dataset as follows:
```shell
python data_preprocess.py --data_dir "path_to_dataset"
```

**[For training datasets only]:** Crop and save data as `.npy` files via:
```shell
python data_crop_npy.py --src_folder "path_to_src_folder" --dst_folder "path_to_dst_folder"
```

Perform train/val split on the training dataset as needed.
Place the newly generated splits in different folders, as the network's training session differentiates
between training and validation splits according to where it is placed (defined in `config.yaml` file).


## Neural Network Training/Testing
Training/Testing sessions can be executed in two ways: `anaconda` and `docker`.
Training/Testing sessions are guided by the configuration defined in the `config.yaml` file.

### 1. Anaconda

If you have [Anaconda](https://docs.anaconda.com/anaconda/install/) set-up on your system, install
[PyTorch](https://pytorch.org/) (tested with `1.8.1`, CUDA `11.1`).
Install the rest of the dependencies with:
```shell
pip install -r requirements.txt
```

Move to the `e1d3` folder to execute the following commands.

**Training:**
```shell
python train.py --config config.yaml --gpu 0
```

**Testing:**
```shell
python test.py --config config.yaml --gpu 0
```

### 2. Docker

If you have [Docker](https://docs.docker.com/get-docker/) set-up with [GPU support](
https://github.com/NVIDIA/nvidia-docker), build a docker image as follows:
```shell
docker build -t brats_e1d3 .
```
To execute the container for training/testing, provide paths in *absolute* format.

**Training:**
```shell
docker run --rm --gpus all -v "train_data_path":"train_data_path" -v "val_data_path":"val_data_path" -v "model_save_path":"model_save_path" -v "config_path":"config_path" brats_e1d3 --train --config "config_path/config.yaml" --gpu 0
```

**Testing:**
```shell
docker run --rm --gpus all -v "test_data_path":"test_data_path" -v "model_load_path":"model_load_path" -v "config_path":"config_path" brats_e1d3 --test --config "config_path/config.yaml" --gpu 0
```

**[Note]:** The paths set internally in the docker container should match those provided in `config.yaml`, as those will
only be visible to the training/testing session.

<!-- Citation-->
## Citation

If you found any part of this work useful, please cite as follows:
```bibtex
@misc{bukhari2021e1d3,
    title={E1D3 U-Net for Brain Tumor Segmentation: Submission to the RSNA-ASNR-MICCAI BraTS 2021 Challenge},
    author={Syed Talha Bukhari and Hassan Mohy-ud-Din},
    year={2021},
    eprint={2110.02519},
    archivePrefix={arXiv},
    primaryClass={eess.IV},
    url={https://arxiv.org/abs/2110.02519},
}
```
