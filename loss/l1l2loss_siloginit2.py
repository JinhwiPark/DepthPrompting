import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

    def forward(self, output, gt):

        pred = output['pred']
        init = output['pred_init']

        gt = torch.clamp(gt, min=self.min_depth, max=self.max_depth)
        pred = torch.clamp(pred, min=self.min_depth, max=self.max_depth)
        init = torch.clamp(init, min=self.min_depth, max=self.max_depth)

        mask = (gt > self.min_depth).type_as(pred).detach()

        valid_mask = gt > self.min_depth
        
        if self.max_depth is not None:            
            valid_mask = torch.logical_and(gt > self.min_depth, gt <= self.max_depth)
        g = torch.log(init[valid_mask]) - torch.log(gt[valid_mask])
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)

        d1 = torch.sum(torch.pow(pred - gt, 2) * mask, dim=[1, 2, 3])
        d2 = torch.sum(torch.abs(pred - gt) * mask, dim=[1, 2, 3])
        d3 = torch.sqrt(Dg)

        num_valid = torch.sum(mask, dim=[1, 2, 3])
        loss = (d1+d2) * 0.1 / (num_valid + 1e-8) + d3

        return loss.mean()
