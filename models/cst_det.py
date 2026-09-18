import torch
from torch import nn
from release import RELEASE_NOTICE

class SpatiotemporalFeatures(nn.Module):

    def __init__(self):
        raise NotImplementedError(RELEASE_NOTICE)

    def forward(self, eeg):
        raise NotImplementedError(RELEASE_NOTICE)

class LearnableSmoothing(nn.Module):

    def __init__(self, kernel_size=5):
        raise NotImplementedError(RELEASE_NOTICE)

    def forward(self, scores):
        raise NotImplementedError(RELEASE_NOTICE)

class CSTDet(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = SpatiotemporalFeatures()
        self.smoothing = LearnableSmoothing(kernel_size=5)

    def forward(self, eeg):
        scores = self.features(eeg)
        return torch.sigmoid(self.smoothing(scores))
