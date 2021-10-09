import numpy as np
import torch


class MetricsPt(object):
    def __init__(self):
        """Training/Validation Metrics"""
        self._EPS = 1e-7

    def accuracy(self, y_pred, y_true):
        """
        y_pred: (B, H, W, D)    # from argmax
        y_true: (B, H, W, D)    # no channel dimension
        """
        return torch.sum(y_pred == y_true).float() / (1. * y_true.nelement())

    def dice_score(self, y_pred_bin, y_true_bin):
        """
        y_pred_bin: (B, H, W, D)    # binary volume
        y_true_bin: (B, H, W, D)    # binary volume
        """
        return (2.0 * torch.sum(y_true_bin * y_pred_bin) + self._EPS) / \
               (torch.sum(y_true_bin) + torch.sum(y_pred_bin) + self._EPS)

    def get_dice_per_region(self, y_pred, y_true):
        """
        y_pred: (B, H, W, D)    # from argmax
        y_true: (B, H, W, D)    # no channel dimension
        """
        y_pred_wt = torch.gt(y_pred, 0).float()
        y_true_wt = torch.gt(y_true, 0).float()
        dice_wt = self.dice_score(y_pred_wt, y_true_wt)

        y_pred_tc = (torch.eq(y_pred, 1) | torch.eq(y_pred, 3)).float()
        y_true_tc = (torch.eq(y_true, 1) | torch.eq(y_true, 3)).float()
        dice_tc = self.dice_score(y_pred_tc, y_true_tc)

        y_pred_en = torch.eq(y_pred, 3).float()
        y_true_en = torch.eq(y_true, 3).float()
        dice_en = self.dice_score(y_pred_en, y_true_en)

        return dice_wt, dice_tc, dice_en


class MetricsNp(object):
    def __init__(self):
        """Test Time Metrics"""
        self._EPSILON = 1e-7

    def __call__(self, data, label):
        """"""
        return self.get_dice_per_region(data, label)

    def dice_score_binary_np(self, data, label):
        """
        f1-score <or> dice score: 2TP / (2TP + FP + FN)
        Args:
            label: 'np.array' binary volume.
            data: 'np.array' binary volume.
        Returns:
            dice_score (scalar)
        """
        numerator = 2 * np.sum(label * data)
        denominator = np.sum(label + data)
        score = (numerator + self._EPSILON) / (denominator + self._EPSILON)
        if np.isnan(score).any() or np.isinf(score).any():
            return 'NA'
        return score

    def get_dice_per_region(self, data, label):  # data = Prediction, label = Ground Truth
        """
        Provides region-wise Dice scores of a multi-class prediction simultaneously
        """
        assert data.shape == label.shape  # Shape check
        print('> Calculating Metrics ...')
        unique_labels = np.unique(label)
        num_classes = len(unique_labels)
        print('Total Classes:\t', num_classes)
        print('Class Labels:\t', unique_labels)

        # Whole Tumor:
        wt_data = np.float32(data > 0)
        wt_label = np.float32(label > 0)
        wt_score = self.dice_score_binary_np(wt_data, wt_label)

        # Tumor Core:
        tc_data = np.float32((data == 1) | (data == 4))
        tc_label = np.float32((label == 1) | (label == 4))
        tc_score = self.dice_score_binary_np(tc_data, tc_label)

        # Enhancing Core:
        en_data = np.float32(data == 4)
        en_label = np.float32(label == 4)
        en_score = self.dice_score_binary_np(en_data, en_label)

        return wt_score, tc_score, en_score


if __name__ == '__main__':
    print('No *elaborate* testing routine implemented')
    metrics_obj = MetricsPt()
    x = torch.zeros(240, 240, 155)
    y = torch.zeros(240, 240, 155)
    print(metrics_obj.dice_score(x, y))
