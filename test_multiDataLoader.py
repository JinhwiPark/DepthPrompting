import os
import random
import time
import warnings
from collections import OrderedDict
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from utils.metric_func import *
from utils.util_func import *

from tqdm import tqdm
from config import args as args_config
from model_list import import_model

args = args_config
best_rmse = 10.0

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
    
    if args.data_name == 'NYU':
        from data.nyu import NYU as NYU_Dataset
        args.patch_height, args.patch_width = 240, 320
        args.max_depth = 10.0
        args.split_json = './data/data_split/nyu.json'
        target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
        val_datasets = [NYU_Dataset(args, 'test', num_sample_test=v) for v in target_vals]
        print('Dataset is NYU')
        num_sparse_dep = args.num_sample        
    elif args.data_name == 'KITTIDC':
        from data.kittidc import KITTIDC as KITTI_dataset
        args.patch_height, args.patch_width = 240, 1216
        args.max_depth = 80.0
        args.split_json = './data/data_split/kitti_dc.json'
        target_vals = convert_str_to_num(args.kitti_val_lidars, 'int')
        val_datasets = [KITTI_dataset(args, 'test', num_lidars_test=v) for v in target_vals]
        print('Dataset is KITTI')
        num_sparse_dep = args.lidar_lines
    elif args.data_name == 'VOID':
        from data.void import VOID
        args.nyu_val_samples = str(args.void_sparsity)
        target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
        dataset = VOID(args, 'test')
        val_datasets = [dataset]
        num_sparse_dep = args.num_sample
    elif args.data_name == 'SUNRGBD':
        from data.sun_rgbd import SUN_RGBD
        args.split_json = './data/data_split/allsplit.mat'
        args.max_depth = 10.0
        target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
        val_datasets = [SUN_RGBD(args, 'test', num_sample_test=v) for v in target_vals]
        num_sparse_dep = args.num_sample
    elif args.data_name == 'IPAD':
        from data.ipad import iPad as IPAD_dataset
        args.max_depth = 10.0
        target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
        val_datasets = [IPAD_dataset(args, 'test', num_sample_test=v) for v in target_vals]
        print('[IPAD Dataset] Split: {} | MaxDepth: {} | H,W: {},{} | Num Sample: {}'.format(args.split_json, args.max_depth, args.patch_height, args.patch_width, args.num_sample))
        num_sparse_dep = args.num_sample
    elif args.data_name == 'NUSCENE':
        from data.nuscene import NUSCENE
        args.max_depth = 80.0
        dataset = NUSCENE(args, 'test')
        target_vals = convert_str_to_num(args.kitti_val_lidars, 'int')
        val_datasets = [dataset]
        num_sparse_dep = args.num_sample
    else:
        print("Please Choice Dataset !!")
        raise NotImplementedError
    model = import_model(args)
    args.num_sparse_dep = num_sparse_dep

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')        

    if args.pretrain is not None:
        print("Pretrain Paramter Path:", args.pretrain)
        checkpoint = torch.load(args.pretrain)
        try:
            loaded_state_dict = checkpoint['state_dict']
        except:
            loaded_state_dict = checkpoint
        new_state_dict = OrderedDict()
        for n, v in loaded_state_dict.items():
            name = n.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model = model.cuda()
        print('Load pretrained weight')

    print('MaxDepth: {} | H,W: {},{}'.format(args.max_depth, args.patch_height, args.patch_width))

    if args.visualization:
        print("Save directory: ", args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        os.system("chmod -R 777 {}".format(args.save_dir))
        from utils import visualize
        visual = visualize.visualize(args)
    else:
        visual = None

    test_loaders = [torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=False, drop_last=False) for val_dataset in val_datasets]

    avg_rmse = AverageMeter('avg_rmse', ':6.4f')
    avg_mae = AverageMeter('avg_mae', ':6.4f')
    
    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args)):
        print(key, ':',  getattr(args, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')
    
    for target_val, val_loader in zip(target_vals, test_loaders):
        val_rmse, val_mae = test(val_loader, model, args, visual, target_val)
        avg_rmse.update(val_rmse)
        avg_mae.update(val_mae)
    print("Test for various Sampels/Lidars:",target_vals)
    for rmse_,mae_ in zip(avg_rmse.list,avg_mae.list):
        print('{:.4f}/{:.4f}'.format(rmse_,mae_),end=" ")
    print("\n [Average RMSE/MAE] ==> {:2.4f}/{:2.4f}\n".format(avg_rmse.avg,avg_mae.avg))
    
def test(test_loader, model, args, visual, target_sample):
    rmse = AverageMeter('RMSE', ':.4f')
    mae = AverageMeter('MAE', ':.4f')
    model.eval()
    pbar = tqdm(total=len(test_loader))

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            sample = {key: val.to('cuda') for key, val in sample.items() if val is not None}
            output = model(sample)

            if target_sample==0: 
                rmse_result, mae_result, abs_rel_result = eval_metric2(sample, output['pred_init'], args)
            else: rmse_result, mae_result, abs_rel_result = eval_metric2(sample, output['pred'], args)

            rmse.update(rmse_result, sample['gt'].size(0))
            mae.update(mae_result, sample['gt'].size(0))

            if args.visualization:
                visual.data_put(sample, output)
                path_ = os.path.join(args.save_dir,'sample_{:04d}'.format(target_sample))
                os.makedirs(path_, exist_ok=True)
                if args.data_name ==  'IPAD':
                    visual.save_all_nyu_gt_sparse_rgb_errormap(idx=i, path_to_save=path_)    
                elif args.data_name == 'NUSCENE':
                    visual.save_all_kitti_gt_sparse_rgb_errormap(idx=i, path_to_save=path_)
                    visual.depth(type='pred', idx=i, path_to_save=path_)
                    visual.depth(type='sparse', idx=i, path_to_save=path_)
                    visual.RGB(idx=i, path_to_save=path_)

            if args.use_raw_depth_as_input:
                error_str = '{} | #:{} | '.format('Test', 'raw')
            else:
                error_str = '{} | #:{:3d} | '.format('Test', int(target_sample))

            pbar.set_description(error_str)
            pbar.update(test_loader.batch_size)

        if args.use_raw_depth_as_input:
            error_str_new = '[{}] #:{} | RMSE/MAE: {:.4f}/{:.4f}'.format('Test', 'raw', rmse.avg, mae.avg)
        else:
            error_str_new = '[{}] #:{:3d} | RMSE/MAE: {:.4f}/{:.4f}'.format('Test', int(target_sample), rmse.avg, mae.avg)
            

        pbar.set_description(error_str_new)
        pbar.close()

    return rmse.avg, mae.avg

if __name__ == '__main__':
    main()
