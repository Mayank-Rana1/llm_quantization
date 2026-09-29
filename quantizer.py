import torch

class TernaryQuantizer:
    @staticmethod
    def quantize(weights, threshold=0.05):
        return torch.where(
            weights > threshold,
            torch.ones_like(weights),
            torch.where(
                weights < -threshold,
                -torch.ones_like(weights),
                torch.zeros_like(weights)
            )
        )
