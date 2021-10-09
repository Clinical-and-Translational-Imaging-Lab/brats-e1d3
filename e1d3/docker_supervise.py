import os
import argparse

from train import TrainSession
from test import TestSession
from utils.parse_yaml import parse_yaml_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E1D3 U-Net for Brain Tumor Segmentation.')
    parser.add_argument('--train', action='store_true', default=False, help='train flag')
    parser.add_argument('--test', action='store_true', default=False, help='test flag')
    parser.add_argument('--config', type=str, required=True, help='.yaml config file')
    parser.add_argument('--gpu', type=int, required=False, default=0, help='CUDA device id')
    args = parser.parse_args()

    assert args.train or args.test
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    config = parse_yaml_config(args.config)

    if args.test:
        assert not args.train
        sess = TestSession(config=config)
        sess.inference_over_all_patients()

    if args.train:
        assert not args.test
        sess = TrainSession(config=config)
        sess.loop_over_epochs()
