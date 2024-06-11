import torch
import torch.nn as nn

class L1L2Loss(nn.Module):
    def __init__(self, args):
        super(L1L2Loss, self).__init__()

        self.max_depth = args.max_depth
        self.min_depth = args.min_depth
        self.t_valid = self.min_depth

    def forward(self, output, gt):
        """
        pred , gt = torch.Size([B, 1, H, W])
        """

        pred = output['pred']
        gt = torch.clamp(gt, min=self.min_depth, max=self.max_depth)
        pred = torch.clamp(pred, min=self.min_depth, max=self.max_depth)

        mask = (gt > self.t_valid).type_as(pred).detach()

        d = torch.pow(pred - gt, 2) * mask + torch.abs(pred - gt) * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.mean()

        return loss
