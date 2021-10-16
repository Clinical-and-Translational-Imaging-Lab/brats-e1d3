import torch
from torch import nn
import torch.nn.functional as F


class XEntropyPlusDiceLoss(nn.Module):
    """ Cross-Entropy + Dice Loss (Categorical), includes softmax internally """

    def __init__(self, num_classes=5, reduction_dims=(0, 2, 3, 4)):
        super(XEntropyPlusDiceLoss, self).__init__()
        self.dice_loss = DiceLoss(num_classes=num_classes, reduction_dims=reduction_dims)
        self.ce_loss = XEntropyLoss()
        print(f"Instantiated: {self.__class__.__name__}")

    def forward(self, y_pred, y_true):
        """
        y_pred: (B, C, D, H, W), without softmax
        y_true: (B, D, H, W), dtype='long'
        """
        dice_loss = self.dice_loss(y_pred, y_true)
        ce_loss = self.ce_loss(y_pred, y_true)
        return dice_loss + ce_loss


class XEntropyLoss(nn.Module):
    """ Cross-Entropy (Categorical), includes softmax internally """

    def __init__(self):
        super(XEntropyLoss, self).__init__()
        self.crossentropy = nn.modules.loss.CrossEntropyLoss(reduction='mean')

    def forward(self, y_pred, y_true):
        """
        y_pred: (B, C, D, H, W), without softmax
        y_true: (B, D, H, W), dtype='long'
        """
        return self.crossentropy(y_pred, y_true)


class DiceLoss(nn.Module):
    """ - (dice score), includes softmax """

    def __init__(self, num_classes=5, reduction_dims=(0, 2, 3, 4)):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self._REDUCTION_DIMS = reduction_dims
        self._EPS = 1e-7

    def forward(self, y_pred, y_true):
        """
        y_pred: (B, C, D, H, W), without softmax
        y_true: (B, D, H, W), dtype='long'
        """
        y_true = F.one_hot(y_true, num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        y_pred = F.softmax(y_pred, dim=1)  # activate prediction

        numerator = 2.0 * torch.sum(y_true * y_pred, dim=self._REDUCTION_DIMS)
        denominator = torch.sum(y_true, dim=self._REDUCTION_DIMS) + torch.sum(y_pred, dim=self._REDUCTION_DIMS)
        return - torch.mean((numerator + self._EPS) / (denominator + self._EPS))
