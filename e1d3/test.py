import os
import random
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from utils.enc1_dec3 import PrototypeArchitecture3d
from utils.inferenceloader import DatasetInference3d
from utils.session_logger import show_progress
from utils.parse_yaml import parse_yaml_config

seed = 40
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
assert torch.cuda.is_available(), "No visible CUDA device."
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


class TestSession:
    """class for managing testing sessions"""

    def __init__(self, config=None, config_file=None):
        print('--------------------------------')
        print('> Initializing Testing Session <')
        print('--------------------------------')
        if config is None:
            assert config_file is not None, f"`config_file` is needed if `config` not provided."
            assert os.path.exists(config_file), f"config file {config_file} not found."
            config = parse_yaml_config(config_file)

        config_data = config['data']
        self.num_classes = config_data.get('num_classes')

        config_net = config['network']
        self.label_shape = config_net.get('label_shape')

        config_test = config['test']
        self.tta = config_test.get('test_time_aug')
        self.save_segmentation = config_test.get('save_segmentation')

        model_load_directory = config_net.get('model_load_directory')
        model_load_config = config_net.get('model_load_config')
        model_checkpoint_str = 'epoch_{epoch:02d}_val_loss_{val_loss:.2f}.pt'.format(
            epoch=int(model_load_config[1]), val_loss=float(model_load_config[2]))
        model_file = os.path.join(model_load_directory, model_load_config[0], model_checkpoint_str)
        self.model = PrototypeArchitecture3d(config=config).cuda()
        self.load_model(model_file)

        self.inference_loader_obj = DatasetInference3d(config)

    def inference_over_all_patients(self):
        """"""
        all_patients = self.inference_loader_obj.get_patients_list()
        print(f"Patients in Directory:\n{str(all_patients)}")
        t0 = datetime.datetime.now()  # start time for session
        print('\n> Starting Clock: {}\n'.format(t0))

        for patient_id in all_patients:
            print('\n----------- Testing Patient: [{}] ------------\n'.format(patient_id))
            t0_p = datetime.datetime.now()  # start time for patient
            print('\n> Starting Clock: {}\n'.format(t0_p))

            data, info, weight = self.inference_loader_obj.get_patient(patient_id)
            wt_prob_map, tc_prob_map, en_prob_map = self.inference_loop(data, weight)

            # test-time augmentation (optional):
            if self.tta:
                for key in TestTimeAugmentation.flips.keys():
                    print('------- TTA dim:', key, '-------')
                    data_tta = TestTimeAugmentation.flip_op(data.copy(), choice=key)
                    tta_out = self.inference_loop(data_tta, weight)

                    wt_prob_map += TestTimeAugmentation.flip_op(tta_out[0], choice=key)
                    tc_prob_map += TestTimeAugmentation.flip_op(tta_out[1], choice=key)
                    en_prob_map += TestTimeAugmentation.flip_op(tta_out[2], choice=key)

                num_infs = float(len(TestTimeAugmentation.flips.keys()) + 1)

                wt_prob_map /= num_infs
                tc_prob_map /= num_infs
                en_prob_map /= num_infs

            if self.save_segmentation:
                print(f"Saving Probability Map for patient: {patient_id}")

            (wt_segm_map, tc_segm_map, en_segm_map) = (np.uint8(np.argmax(wt_prob_map, axis=0) * weight),
                                                       np.uint8(np.argmax(tc_prob_map, axis=0) * weight),
                                                       np.uint8(np.argmax(en_prob_map, axis=0) * weight))
            segm_map = self.label_fusion(wt_segm_map, tc_segm_map, en_segm_map)
            segm_map *= weight.astype(segm_map.dtype)
            print('Segmentation Info', '| Shape:', segm_map.shape,
                  '| Labels:', np.unique(segm_map),
                  '| Data-type:', segm_map.dtype)

            if self.save_segmentation:  # save combined segmentation
                print(f"Saving Segmentation Map for patient: {patient_id}")
                self.inference_loader_obj.save_volume(segm_map, info[0][0], patient_id, volume_type='seg')

            stats = torch.cuda.memory_stats()
            peak_bytes_requirement = stats["allocated_bytes.all.peak"]
            print('Peak Memory Usage:', peak_bytes_requirement, '(Bytes)')
            torch.cuda.reset_peak_memory_stats()

            tn_p = datetime.datetime.now()  # end patient time
            print('\n> Ending Clock: {}\n'.format(tn_p))
            print('--- Time Taken: {} ---\n'.format(tn_p - t0_p))

        print('Testing Finished Successfully!')
        tn = datetime.datetime.now()
        print('\n> Ending Clock: {}\n'.format(tn))
        print('--- Total Testing Time: {} ---\n'.format(tn - t0))

    def inference_loop(self, data, weight):
        """"""
        input_shape = data.shape  # original input shape (before non-zero cropping)
        # read non-zero bounding box of data
        ind_limits_list = self.get_non_zero_bounding_box(data)
        # extract the non-zero bounding box from data, weight, and label.
        data = self.extract_volume(data, ind_limits_list)
        weight = self.extract_volume(weight, ind_limits_list[1:])
        weight[:] = 1

        self.inference_loader_obj.calculate_number_of_steps(weight)
        batches = len(self.inference_loader_obj)
        generator_function = self.inference_loader_obj.generator_function(data)

        # maps for extracted non-zero regions:
        wt_probability_map_ = torch.zeros(self.num_classes, *data.shape[1:]).cuda()
        tc_probability_map_ = torch.zeros(self.num_classes, *data.shape[1:]).cuda()
        en_probability_map_ = torch.zeros(self.num_classes, *data.shape[1:]).cuda()
        weight_map_for_probabilities = torch.zeros_like(wt_probability_map_).cuda()

        # maps for actual input volume:
        wt_probability_map = torch.stack((torch.ones(*input_shape[1:]), torch.zeros(*input_shape[1:])), dim=0).cuda()
        tc_probability_map = torch.stack((torch.ones(*input_shape[1:]), torch.zeros(*input_shape[1:])), dim=0).cuda()
        en_probability_map = torch.stack((torch.ones(*input_shape[1:]), torch.zeros(*input_shape[1:])), dim=0).cuda()
        weight_map_update = torch.ones(self.num_classes, *self.label_shape).cuda()

        print('Total Batches:', batches)
        with torch.no_grad():
            for loop in range(batches):
                show_progress(loop + 1, batches)

                # batch extraction:
                (batch_data,
                 batch_coords,
                 batch_padding) = next(generator_function)

                with torch.no_grad():
                    # prediction:
                    (output_tensor_wt,
                     output_tensor_tc,
                     output_tensor_en) = self.model(torch.from_numpy(batch_data).cuda(non_blocking=True))

                probabilities_wt = F.softmax(output_tensor_wt, dim=1)
                probabilities_tc = F.softmax(output_tensor_tc, dim=1)
                probabilities_en = F.softmax(output_tensor_en, dim=1)

                for subloop in range(len(batch_coords)):
                    coordinates = batch_coords[subloop]
                    undo_padding = batch_padding[subloop]

                    prob_segment_wt = probabilities_wt[subloop, ...]
                    prob_segment_tc = probabilities_tc[subloop, ...]
                    prob_segment_en = probabilities_en[subloop, ...]
                    w_undo_padding = [range(0, self.num_classes)] + undo_padding[1:]

                    wt_probability_map_[np.ix_(*coordinates)] += prob_segment_wt[np.ix_(*undo_padding)] * \
                                                                 weight_map_update[np.ix_(*w_undo_padding)]
                    tc_probability_map_[np.ix_(*coordinates)] += prob_segment_tc[np.ix_(*undo_padding)] * \
                                                                 weight_map_update[np.ix_(*w_undo_padding)]
                    en_probability_map_[np.ix_(*coordinates)] += prob_segment_en[np.ix_(*undo_padding)] * \
                                                                 weight_map_update[np.ix_(*w_undo_padding)]

                    weight_map_for_probabilities[np.ix_(*coordinates)] += weight_map_update[np.ix_(*w_undo_padding)]

        ind_limits_list = [range(0, self.num_classes)] + ind_limits_list[1:]

        wt_probability_map[np.ix_(*ind_limits_list)] = wt_probability_map_ * 1. / weight_map_for_probabilities
        tc_probability_map[np.ix_(*ind_limits_list)] = tc_probability_map_ * 1. / weight_map_for_probabilities
        en_probability_map[np.ix_(*ind_limits_list)] = en_probability_map_ * 1. / weight_map_for_probabilities

        # pmap checks:
        for pmap_x in [wt_probability_map, tc_probability_map, en_probability_map]:
            assert not pmap_x.isnan().any(), "[prob maps] check for NaN(s) failed"
            assert not pmap_x.isinf().any(), "[prob maps] check for Inf(s) failed"
            assert torch.logical_and(pmap_x >= 0, pmap_x <= 1).all(), "[prob maps] not in range [0, 1]"

        return wt_probability_map.cpu().numpy(), tc_probability_map.cpu().numpy(), en_probability_map.cpu().numpy()

    def label_fusion(self, wt_map, tc_map, en_map):
        struct = ndimage.generate_binary_structure(3, 2)

        wt_map = ndimage.morphology.binary_closing(wt_map, structure=struct)
        tc_map = ndimage.morphology.binary_closing(tc_map, structure=struct)
        en_map = ndimage.morphology.binary_closing(en_map, structure=struct)

        wt_map = self.get_largest_two_component(wt_map.copy(), structure=struct, print_info=False, threshold=2000)
        wt_mask = (wt_map + tc_map + en_map) > 0
        wt_mask = ndimage.morphology.binary_closing(wt_mask, structure=struct)
        wt_mask = self.get_largest_two_component(wt_mask, structure=struct, print_info=False, threshold=2000)
        wt_map = wt_map * wt_mask

        tc_en_mask = (tc_map + en_map) > 0
        tc_en_mask = tc_en_mask * wt_mask
        tc_en_mask = ndimage.morphology.binary_closing(tc_en_mask, structure=struct)
        tc_en_mask = self.remove_external_core(wt_map, tc_en_mask, s=struct)
        wt_map = (wt_map + tc_en_mask) > 0
        tc_map = tc_en_mask
        en_map = tc_map * en_map

        vox_3 = np.asarray(en_map > 0, np.float32).sum()
        if 0 < vox_3 < 30:
            en_map = np.zeros_like(tc_map)

        segmentation_map = wt_map * 2
        segmentation_map[tc_map > 0] = 1
        segmentation_map[en_map > 0] = 4

        return segmentation_map.astype(np.uint8)

    @staticmethod
    def get_largest_two_component(img, structure, print_info=False, threshold=None):
        labeled_array, numpatches = ndimage.label(img, structure)  # labeling
        sizes = ndimage.sum(img, labeled_array, range(1, numpatches + 1))
        sizes_list = [sizes[i] for i in range(len(sizes))]
        sizes_list.sort()

        if print_info:
            print('component size', sizes_list)
        if len(sizes) == 1:
            out_img = img
        else:
            if threshold:
                out_img = np.zeros_like(img)
                for temp_size in sizes_list:
                    if temp_size > threshold:
                        temp_lab = np.where(sizes == temp_size)[0] + 1
                        temp_cmp = labeled_array == temp_lab
                        out_img = (out_img + temp_cmp) > 0
                return out_img
            else:
                max_size1 = sizes_list[-1]
                max_size2 = sizes_list[-2]
                max_label1 = np.where(sizes == max_size1)[0] + 1
                max_label2 = np.where(sizes == max_size2)[0] + 1
                component1 = labeled_array == max_label1
                component2 = labeled_array == max_label2
                if max_size2 * 10 > max_size1:
                    component1 = (component1 + component2) > 0
                out_img = component1
        return out_img

    @staticmethod
    def remove_external_core(lab_main, lab_ext, s):
        """
        remove the core region that is outside of whole tumor
        """
        labeled_array, numpatches = ndimage.label(lab_ext, s)  # labeling
        sizes = ndimage.sum(lab_ext, labeled_array, range(1, numpatches + 1))
        sizes_list = [sizes[i] for i in range(len(sizes))]
        new_lab_ext = np.zeros_like(lab_ext)
        for i in range(len(sizes)):
            sizei = sizes_list[i]
            labeli = np.where(sizes == sizei)[0] + 1
            componenti = labeled_array == labeli
            overlap = componenti * lab_main
            if (overlap.sum() + 0.0) / sizei >= 0.5:
                new_lab_ext = np.maximum(new_lab_ext, componenti)
        return new_lab_ext

    @staticmethod
    def get_non_zero_bounding_box(volume):
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
    def extract_volume(volume, ind_limits_list):
        """
        Returns a rectangular crop of the input `volume`
        """
        return volume[np.ix_(*ind_limits_list)]

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))  # providing a dictionary object
        print('Loaded Model file:', path)
        self.model.eval()  # setting dropout/batch-norm/etc layers to evaluation mode


class TestTimeAugmentation:
    flips = {
        'axial': (1,),
        'coronal': (2,),
        'sagittal': (3,),
        'axial-coronal': (1, 2),
        'coronal-sagittal': (2, 3),
        'axial-sagittal': (1, 3),
        'axial-coronal-sagittal': (1, 2, 3),
    }

    @classmethod
    def flip_op(cls, volume, choice, noise_variance=False):
        assert choice in cls.flips.keys()
        axis = cls.flips.get(choice)
        volume = np.flip(volume, axis)
        if noise_variance:
            volume += np.random.randn(*volume.shape) * noise_variance
        return volume


if __name__ == '__main__':
    """
    Run as:
    python test.py --config config.yaml --gpu 0
    """
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess BraTS data.')
    parser.add_argument('--config', type=str, required=True, help='path to `.yaml` config file')
    parser.add_argument('--gpu', type=int, required=False, default=0, help='CUDA device id')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sess = TestSession(config_file=args.config)
    sess.inference_over_all_patients()
