import numpy as np

try:
    import torchio
except ImportError:
    raise Exception('Install TorchIO via: pip install torchio')


class DataAugmentation():
    def __init__(self, aug_list=None):
        p = 0.5  # probability that a certain transform is applied
        flip_prob = 0.5  # probability
        scaling = 0.1  # percentage
        rot = 10  # degrees
        shift = 0.5  # mm
        el_ctrl_pts = 5
        el_max_disp = 4.5
        self.gamma_range = (0.8, 1.2)

        aug_fns = {
            'flip': torchio.transforms.RandomFlip(
                axes=(0, 1, 2),
                flip_probability=flip_prob,
                p=p,
            ),
            'affine': torchio.transforms.RandomAffine(
                scales=(1. - scaling, 1. + scaling, 1. - scaling, 1. + scaling, 1. - scaling, 1. + scaling),
                degrees=(-rot, +rot, -rot, +rot, -rot, +rot),
                translation=(-shift, +shift, -shift, +shift, -shift, +shift),
                isotropic=False,
                center='image',
                default_pad_value=0.,
                image_interpolation='bspline',
                p=p,
            ),
            'elastic': torchio.transforms.RandomElasticDeformation(
                num_control_points=el_ctrl_pts,
                max_displacement=el_max_disp,
                locked_borders=2,
                image_interpolation='bspline',
                p=p,
            ),
            'gamma': torchio.transforms.Lambda(
                function=self.gamma_correction,
                types_to_apply=[torchio.INTENSITY],
                p=p,
            ),
        }

        if aug_list is None:
            aug_list = [i for i in aug_fns]
        else:
            assert len(aug_list) != 0, "Empty list of augmentations provided"
            aug_list = [i.lower() for i in aug_list]
        print('Using Augmentations:', *aug_list)

        aug_fns_list = [aug_fns[i] for i in aug_list if i not in ['gamma', 'noise']]
        # do either noise or gamma, but never both:
        if ('gamma' in aug_list) and ('noise' in aug_list):
            one_of = torchio.transforms.OneOf({
                aug_fns['gamma']: 0.5,
                aug_fns['noise']: 0.5,
            })
            aug_fns_list.append(one_of)

        elif ('gamma' in aug_list) or ('noise' in aug_list):
            if 'gamma' in aug_list:
                aug_fns_list.append(aug_fns['gamma'])
            elif 'noise' in aug_list:
                aug_fns_list.append(aug_fns['noise'])
            else:
                raise Exception('Something is not right!')
        else:
            print('Neither `gamma` nor `noise` used in augmentation')

        self.composed_fns = torchio.transforms.Compose(aug_fns_list)
        del aug_fns

    def __call__(self, data, label):
        input_dict = torchio.Subject(
            data=torchio.ScalarImage(tensor=data),
            label=torchio.LabelMap(tensor=label),
        )
        output_dict = self.composed_fns(input_dict)
        return output_dict['data'].data, output_dict['label'].data

    def get_transform(self):
        return self.composed_fns

    def gamma_correction(self, tensor):
        """ gamma correction code, as implemented in No New Net (Isensee et al.) """
        gamma = np.random.uniform(*self.gamma_range)
        intensity_range = (tensor.max() - tensor.min()).abs()
        return (((tensor - tensor.min()) / (intensity_range + 1e-7)) ** gamma) * intensity_range + tensor.min()


if __name__ == '__main__':
    import time

    data = np.random.randn(4, 240, 240, 155)
    label = np.uint16(np.random.randn(1, 240, 240, 155) > 0)
    print(data.shape, label.shape)
    print(len(np.unique(data)), np.unique(label))

    data_aug_fn = DataAugmentation(aug_list=None)
    t1 = time.perf_counter()
    new_data, new_label = data_aug_fn(data, label)
    t2 = time.perf_counter()
    print(new_data.shape, new_label.shape)
    print(len(np.unique(new_data)), np.unique(new_label))
    print('> Time Taken to Distort: %.2f seconds' % (t2 - t1))
