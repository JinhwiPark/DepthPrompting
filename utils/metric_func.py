import math
import numpy as np
import torch
import torch.nn.functional as F
import warnings

def evaluate_all_metric(sample, output, t_valid=0.0001):
    with torch.no_grad():
        pred = output.detach()
        gt = sample['gt'].detach()

        pred_inv = 1.0 / (pred + 1e-8)
        gt_inv = 1.0 / (gt + 1e-8)

        mask = gt > t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        pred_inv = pred_inv[mask]
        gt_inv = gt_inv[mask]

        pred_inv[pred <= t_valid] = 0.0
        gt_inv[gt <= t_valid] = 0.0

        diff = pred - gt
        diff_abs = torch.abs(diff)
        diff_sqr = torch.pow(diff, 2)

        rmse = diff_sqr.sum() / (num_valid + 1e-8)
        rmse = torch.sqrt(rmse)

        mae = diff_abs.sum() / (num_valid + 1e-8)

        diff_inv = pred_inv - gt_inv
        diff_inv_abs = torch.abs(diff_inv)
        diff_inv_sqr = torch.pow(diff_inv, 2)

        irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
        irmse = torch.sqrt(irmse)

        imae = diff_inv_abs.sum() / (num_valid + 1e-8)

        rel = diff_abs / (gt + 1e-8)
        rel = rel.sum() / (num_valid + 1e-8)

        r1 = gt / (pred + 1e-8)
        r2 = pred / (gt + 1e-8)
        ratio = torch.max(r1, r2)

        del_1 = (ratio < 1.25).type_as(ratio)
        del_2 = (ratio < 1.25**2).type_as(ratio)
        del_3 = (ratio < 1.25**3).type_as(ratio)

        del_1 = del_1.sum() / (num_valid + 1e-8)
        del_2 = del_2.sum() / (num_valid + 1e-8)
        del_3 = del_3.sum() / (num_valid + 1e-8)

        result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3]

    return result

def rmse_eval(sample, output, t_valid=1e-3):

    with torch.no_grad():
        pred = output.detach()
        gt = sample['gt'].detach()

        mask = gt > t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        diff = pred - gt
        diff_sqr = torch.pow(diff, 2)

        rmse = diff_sqr.sum() / (num_valid + 1e-8)
        rmse = torch.sqrt(rmse)

    return rmse

def mae_eval(sample, output, t_valid=1e-3):

    with torch.no_grad():
        pred = output.detach()
        gt = sample['gt'].detach()

        mask = gt > t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        diff = pred - gt
        diff_abs = torch.abs(diff)

        rmse = diff_abs.sum() / (num_valid + 1e-8)

    return rmse


def eval_metric(sample, output, t_valid=1e-3):
    with torch.no_grad():
        
        pred = output.detach()
        gt = sample['gt'].detach()

        mask = gt > t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        diff = pred - gt

        diff_sqr = torch.pow(diff, 2)
        rmse = diff_sqr.sum() / (num_valid + 1e-8)
        rmse = torch.sqrt(rmse)
        diff_abs = torch.abs(diff)
        mae = diff_abs.sum() / (num_valid + 1e-8)
        rel_mat = torch.div(diff_abs, gt)
        abs_rel = torch.sum(rel_mat) / num_valid
        y_over_z = torch.div(gt, pred)
        z_over_y = torch.div(pred, gt)
        max_ratio = max_of_two(y_over_z, z_over_y)
        delta_102 = torch.sum(max_ratio < 1.02)/(num_valid + 1e-8)
        delta_105 = torch.sum(max_ratio < 1.05)/(num_valid + 1e-8)
        delta_110 = torch.sum(max_ratio < 1.10)/(num_valid + 1e-8)
        delta_125_1 = torch.sum(max_ratio < 1.25)/(num_valid + 1e-8)
        delta_125_2 = torch.sum(max_ratio < 1.25**2)/(num_valid + 1e-8)
        delta_125_3 = torch.sum(max_ratio < 1.25**3)/(num_valid + 1e-8)

    return rmse, mae, abs_rel, delta_102, delta_105, delta_110, delta_125_1, delta_125_2, delta_125_3

def evaluate_error(gt_depth, pred_depth):
    depth_mask = gt_depth>0.0001
    batch_size = gt_depth.size(0)
    error = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
             'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
             'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0,
             }
    _pred_depth = pred_depth[depth_mask]
    _gt_depth   = gt_depth[depth_mask]
    n_valid_element = _gt_depth.size(0)

    if n_valid_element > 0:
        diff_mat = torch.abs(_gt_depth-_pred_depth)
        rel_mat = torch.div(diff_mat, _gt_depth)
        error['MSE'] = torch.sum(torch.pow(diff_mat, 2))/n_valid_element
        error['RMSE'] = math.sqrt(error['MSE'])
        error['MAE'] = torch.sum(diff_mat)/n_valid_element
        error['ABS_REL'] = torch.sum(rel_mat)/n_valid_element
        y_over_z = torch.div(_gt_depth, _pred_depth)
        z_over_y = torch.div(_pred_depth, _gt_depth)
        max_ratio = max_of_two(y_over_z, z_over_y)
        error['DELTA1.02'] = torch.sum(max_ratio < 1.02).numpy()/float(n_valid_element)
        error['DELTA1.05'] = torch.sum(max_ratio < 1.05).numpy()/float(n_valid_element)
        error['DELTA1.10'] = torch.sum(max_ratio < 1.10).numpy()/float(n_valid_element)
        error['DELTA1.25'] = torch.sum(max_ratio < 1.25).numpy()/float(n_valid_element)
        error['DELTA1.25^2'] = torch.sum(max_ratio < 1.25**2).numpy()/float(n_valid_element)
        error['DELTA1.25^3'] = torch.sum(max_ratio < 1.25**3).numpy()/float(n_valid_element)
    return error

def max_of_two(y_over_z, z_over_y):
    return torch.max(y_over_z, z_over_y)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def eval_metric2(sample, output, args, rescale=None):
    args.min_depth = 1e-3
    with torch.no_grad():
        pred = output.detach()
        gt = sample['gt'].detach()

        mask = eval_mask(gt,args, garg_crop=args.garg_crop, eigen_crop=args.eigen_crop)
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        diff = pred - gt
        diff_sqr = torch.pow(diff, 2)

        diff_abs = torch.abs(diff)
        rel = diff_abs / (gt)
        rel = rel.sum() / (num_valid)

        rmse = diff_sqr.sum() / (num_valid)
        rmse = torch.sqrt(rmse)
        
        diff_abs = torch.abs(diff)
        mae = diff_abs.sum() / (num_valid + 1e-8)

        y_over_z = torch.div(gt, pred)
        z_over_y = torch.div(pred, gt)
        max_ratio = max_of_two(y_over_z, z_over_y)
        delta_125_1 = torch.sum(max_ratio < 1.25)/(num_valid + 1e-8)
    return rmse, mae, delta_125_1

def eval_mask(depth_gt, args, garg_crop=False, eigen_crop=False):
        depth_gt = depth_gt.cpu().numpy()
        depth_gt = np.squeeze(depth_gt)
        valid_mask = np.logical_and(depth_gt > args.min_depth, depth_gt < args.max_depth)
        
        if garg_crop or eigen_crop:
            gt_height, gt_width = depth_gt.shape
            eval_mask = np.zeros(valid_mask.shape)

            if garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif eigen_crop:
                eval_mask[45:471, 41:601] = 1 
        else:
            eval_mask = 1

        valid_mask = np.logical_and(valid_mask, eval_mask)
        valid_mask = np.expand_dims(valid_mask, axis=(0,1))
        return torch.from_numpy(valid_mask)
