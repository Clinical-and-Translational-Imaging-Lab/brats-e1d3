import os
import random
import time
import datetime

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.enc1_dec3 import PrototypeArchitecture3d
from utils.losses import XEntropyPlusDiceLoss
from utils.metrics import MetricsPt
from utils.dataloader import DatasetMMEP3d
from utils.data_augment import DataAugmentation
from utils.session_logger import show_progress, log_configuration
from utils.parse_yaml import parse_yaml_config

seed = 40
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
assert torch.cuda.is_available(), "No visible CUDA device."
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


class TrainSession:
    """class for managing training sessions"""

    def __init__(self, config=None, config_file=None):
        print('---------------------------------')
        print('> Initializing Training Session <')
        print('---------------------------------')
        if config is None:
            assert config_file is not None, f"`config_file` is needed if `config` not provided."
            assert os.path.exists(config_file), f"config file {config_file} not found."
            config = parse_yaml_config(config_file)

        config_data = config['data']
        num_classes = config_data.get('num_classes')
        num_channels = len(config_data.get('channels'))

        config_net = config['network']
        models_folder = config_net.get('model_save_directory')
        segment_size = config_net.get('data_shape')

        config_train = config['train']
        self.total_epochs = config_train.get('epochs', None)
        num_workers = config_train.get('workers_multithreading')
        initial_learning_rate = config_train.get('initial_learning_rate')
        lr_decay_rate = config_train.get('poly_decay_rate')
        train_batch_size = config_train.get('batch_size')
        train_augment = config_train.get('augmentation')
        train_augment_list = config_train.get('augmentations_to_do', [])

        config_val = config['validate']
        val_batch_size = config_val.get('batch_size')
        val_augment = config_val.get('augmentation')
        val_augment_list = config_train.get('augmentations_to_do', [])

        run_date_time = time.strftime('%Y-%m-%d_%H.%M.%S')  # create folder in models folder
        print('> Session runtime:', run_date_time)
        model_checkpoint_format = 'epoch_{epoch:02d}_val_loss_{val_loss:.2f}.pt'
        self.model_checkpoint_filepath = os.path.join(models_folder, run_date_time, model_checkpoint_format)

        tensorboard_log_path = os.path.join(models_folder, run_date_time)
        # log_configuration(tensorboard_log_path, config_file)
        # log_configuration(tensorboard_log_path, 'utils/enc1_dec3.py')

        self.metrics_list_train = ['loss', 'dice_wt', 'dice_tc', 'dice_en']
        self.metrics_list_val = ['loss', 'dice_wt', 'dice_tc', 'dice_en']
        # initialize metrics:
        self.train_metrics = dict((met, torch.zeros(1, dtype=torch.float32).cpu().requires_grad_(False))
                                  for met in self.metrics_list_train)
        self.val_metrics = dict((met, torch.zeros(1, dtype=torch.float32).cpu().requires_grad_(False))
                                for met in self.metrics_list_val)

        #####################################
        # network model
        self.model = PrototypeArchitecture3d(config).cuda()

        # initialization:
        def init_weights(m):
            if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear) \
                    or isinstance(m, torch.nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(m.weight)  # He-Weights Init
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.1)  # Constant 0.1

        self.model.apply(init_weights)

        #####################################
        # augmentation object:
        self.train_augfn = DataAugmentation(aug_list=train_augment_list) if train_augment else None
        self.val_augfn = DataAugmentation(aug_list=val_augment_list) if val_augment else None

        # train generator:
        self.train_gen_params = {
            'batch_size': train_batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'pin_memory': True,
        }
        self.train_dataset = DatasetMMEP3d(config, 'train', augment=self.train_augfn)
        self.train_iterations = self.train_dataset.batches_per_epoch

        # val generator:
        self.val_gen_params = {
            'batch_size': val_batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'pin_memory': True,
        }
        self.val_dataset = DatasetMMEP3d(config, 'validate', augment=self.val_augfn)
        self.val_iterations = self.val_dataset.batches_per_epoch

        #####################################
        self.loss_fn = XEntropyPlusDiceLoss(num_classes=num_classes).cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=initial_learning_rate, momentum=0.99,
                                         weight_decay=1e-6, dampening=0, nesterov=True)

        def lr_lambda(epoch):
            return (1 - epoch / self.total_epochs) ** lr_decay_rate
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda, last_epoch=-1, verbose=True)

        self.metrics_obj = MetricsPt()

        #####################################
        # tensorboard writer:
        self.writer = SummaryWriter(log_dir=tensorboard_log_path)  # read from config

        # log graph to tensorboard:
        with torch.no_grad():
            random_tensor = torch.zeros(train_batch_size, num_channels, *segment_size).cuda()
            self.writer.add_graph(self.model, random_tensor)
            del random_tensor

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))  # providing a dictionary object
        print('Loaded Model file:', path)
        self.model.eval()  # setting dropout/batchnorm/etc layers to evaluation mode

    def initialize_generator(self, train_or_val):
        if train_or_val == 'train':
            self.train_generator = torch.utils.data.DataLoader(self.train_dataset, **self.train_gen_params)
        elif train_or_val == 'val':
            self.val_generator = torch.utils.data.DataLoader(self.val_dataset, **self.val_gen_params)
        else:
            raise Exception('Incorrect mode provided: {}'.format(train_or_val))

    def delete_generator(self, train_or_val):
        if train_or_val == 'train':  # train generator
            del self.train_generator
        elif train_or_val == 'val':  # val generator
            del self.val_generator
        else:
            raise Exception('Incorrect mode provided: {}'.format(train_or_val))

    def initialize_metrics(self, train_or_val):
        with torch.no_grad():
            if train_or_val.lower() == 'train':
                for met in self.train_metrics:
                    self.train_metrics[met].zero_()
            elif train_or_val.lower() == 'val':
                for met in self.val_metrics:
                    self.val_metrics[met].zero_()
            else:
                raise Exception('Incorrect mode provided: {}'.format(train_or_val))

    def loop_over_epochs(self):
        """
        Epoch = Train -> Validate
        Training and Validation in alternating fashion
        Model is saved and Learning rate is updated at the end of epoch
        """
        for epoch in range(self.total_epochs):
            print('\n' + ('-' * 20))
            print('> Epoch {}/{}'.format(epoch + 1, self.total_epochs))
            print(('-' * 20) + '\n')

            # read and log learning rate:
            self.log_to_tensorboard({
                f"LR/param_group_{i}": pg['lr'] for i, pg in enumerate(self.optimizer.param_groups)
            }, epoch)

            torch.cuda.synchronize()
            t0 = datetime.datetime.now()

            train_dict = self.train_epoch(epoch + 1)
            val_dict = self.val_epoch(epoch + 1)

            torch.cuda.synchronize()
            tn = datetime.datetime.now()

            print('--- Epoch Time: {} ---'.format(tn - t0))

            validation_loss = val_dict['loss/val']
            self.save_model(epoch + 1, validation_loss)
            self.scheduler.step()

        self.writer.flush()
        self.writer.close()
        print('Training Finished!')

    def train_epoch(self, epoch):
        """"""
        self.model.train()  # model in train mode
        self.initialize_generator('train')
        self.initialize_metrics('train')
        self.train_dataset.on_epoch_begin()  # does nothing right now

        # training iterations:
        i = 0
        for data_tensor, label_tensor in self.train_generator:
            i += 1

            # Binarize Labels:
            with torch.no_grad():
                data_tensor = data_tensor.cuda(non_blocking=True)
                label_tensor_wt = self.binarize_labels(label_tensor.clone(), 'WT').long().cuda(non_blocking=True)
                label_tensor_tc = self.binarize_labels(label_tensor.clone(), 'TC').long().cuda(non_blocking=True)
                label_tensor_en = self.binarize_labels(label_tensor.clone(), 'EN').long().cuda(non_blocking=True)
                del label_tensor

            self.optimizer.zero_grad()

            # predict
            with torch.set_grad_enabled(True):
                (output_tensor_wt,
                 output_tensor_tc,
                 output_tensor_en) = self.model(data_tensor)
                loss = (self.loss_fn(output_tensor_wt, label_tensor_wt) +
                        self.loss_fn(output_tensor_tc, label_tensor_tc) +
                        self.loss_fn(output_tensor_en, label_tensor_en)) / 3.

            # backpropagate
            loss.backward()
            # update
            self.optimizer.step()

            # evaluate
            with torch.no_grad():
                dice_wt = self.metrics_obj.dice_score(torch.argmax(output_tensor_wt, dim=1), label_tensor_wt)
                dice_tc = self.metrics_obj.dice_score(torch.argmax(output_tensor_tc, dim=1), label_tensor_tc)
                dice_en = self.metrics_obj.dice_score(torch.argmax(output_tensor_en, dim=1), label_tensor_en)

                metric_dict = {
                    'loss': loss.detach().cpu(),
                    'dice_wt': dice_wt.cpu(),
                    'dice_tc': dice_tc.cpu(),
                    'dice_en': dice_en.cpu(),
                }
                for met in self.train_metrics:
                    self.train_metrics[met] += metric_dict[met]

            # Print progress:
            show_progress(i, self.train_iterations, 'Loss:%.4f' % (self.train_metrics['loss'].item() / i))

        per_epoch_dict = self.per_epoch_metrics(self.train_metrics, i, 'train')
        self.log_to_tensorboard(per_epoch_dict, epoch)
        self.delete_generator('train')
        return per_epoch_dict

    def val_epoch(self, epoch):
        """"""
        self.model.eval()  # model in evaluation mode
        self.initialize_generator('val')
        self.initialize_metrics('val')
        self.val_dataset.on_epoch_begin()

        # validation iterations:
        i = 0
        self.optimizer.zero_grad()  # just to be sure!
        with torch.no_grad():
            for data_tensor, label_tensor in self.val_generator:
                i += 1

                data_tensor = data_tensor.cuda(non_blocking=True)
                label_tensor_wt = self.binarize_labels(label_tensor.clone(), 'WT').long().cuda(non_blocking=True)
                label_tensor_tc = self.binarize_labels(label_tensor.clone(), 'TC').long().cuda(non_blocking=True)
                label_tensor_en = self.binarize_labels(label_tensor.clone(), 'EN').long().cuda(non_blocking=True)
                del label_tensor

                # predict
                (output_tensor_wt,
                 output_tensor_tc,
                 output_tensor_en) = self.model(data_tensor)

                loss = (self.loss_fn(output_tensor_wt, label_tensor_wt) +
                        self.loss_fn(output_tensor_tc, label_tensor_tc) +
                        self.loss_fn(output_tensor_en, label_tensor_en)) / 3.

                dice_wt = self.metrics_obj.dice_score(torch.argmax(output_tensor_wt, dim=1), label_tensor_wt)
                dice_tc = self.metrics_obj.dice_score(torch.argmax(output_tensor_tc, dim=1), label_tensor_tc)
                dice_en = self.metrics_obj.dice_score(torch.argmax(output_tensor_en, dim=1), label_tensor_en)

                metric_dict = {
                    'loss': loss.detach().cpu(),
                    'dice_wt': dice_wt.cpu(),
                    'dice_tc': dice_tc.cpu(),
                    'dice_en': dice_en.cpu(),
                }

                for met in self.val_metrics:
                    self.val_metrics[met] += metric_dict[met]

                # Print progress:
                show_progress(i, self.val_iterations, 'Loss:%.4f' % (self.val_metrics['loss'].item() / i))

        per_epoch_dict = self.per_epoch_metrics(self.val_metrics, i, 'val')
        self.log_to_tensorboard(per_epoch_dict, epoch)
        self.delete_generator('val')
        return per_epoch_dict

    @staticmethod
    def binarize_labels(volume, tumor_mode):
        """  Labels are converted for binary segmentation """
        if tumor_mode.upper() == 'WT':
            volume[volume != 0] = 1
        elif tumor_mode.upper() == 'TC':
            volume[volume == 2] = 0
            volume[volume != 0] = 1
        elif tumor_mode.upper() == 'EN':
            volume[volume != 4] = 0
            volume[volume == 4] = 1
        else:
            raise Exception(f"Incorrect Tumor Mode provided: {tumor_mode}")
        return volume

    @staticmethod
    def per_epoch_metrics(metrics_dict, iterations, train_or_val):
        """displays and returns per_epoch mean of provided metrics"""
        per_epoch_dict = {}
        for key in metrics_dict:
            per_epoch_dict[key + '/' + train_or_val] = metrics_dict[key].item() / (1. * iterations)
        print_list = ['{}:{}'.format(key, '%.4f' % per_epoch_dict[key]) for key in per_epoch_dict]
        print(' | '.join(print_list))
        return per_epoch_dict

    def save_model(self, epoch, epoch_loss_val):
        """"""
        save_path = self.model_checkpoint_filepath.format(epoch=epoch, val_loss=epoch_loss_val)
        torch.save(self.model.state_dict(), save_path)
        print('> Model Saved: {}'.format(save_path))

    def log_to_tensorboard(self, scalar_dict, epoch):
        """routine to log per-epoch metrics to tensorboard"""
        for key in scalar_dict:
            self.writer.add_scalar(key, scalar_dict[key], epoch)


if __name__ == '__main__':
    """
    Run as:
    python train.py --config config.yaml --gpu 0
    """
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess BraTS data.')
    parser.add_argument('--config', type=str, required=True, help='.yaml config file')
    parser.add_argument('--gpu', type=int, required=False, default=0, help='CUDA device id')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sess = TrainSession(config_file=args.config)
    sess.loop_over_epochs()
