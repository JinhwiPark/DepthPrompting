import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', 'y', '1'):
        return True
    elif v.lower() in ('false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_name', type=str, choices=('NYU', 'KITTIDC', 'IPAD', 'NUSCENE', 'VOID', 'SUNRGBD'), help='dataset name')
parser.add_argument('--dir_data', default='YOUR_DATA_FOLDER', help='path to dataset')
parser.add_argument('--save_dir', help='path to dataset')
parser.add_argument('--split_json', help='path to json file')
parser.add_argument('--model_name', type=str, help='model name')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=16, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=12, type=int, metavar='N', help='mini-batch size, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

#Learning rate scheduler
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--decay', type=str, default='10,15,20,25', help='learning rate decay schedule')
parser.add_argument('--gamma', type=str, default='1.0,0.5,0.05,0.001', help='learning rate multiplicative factors')
parser.add_argument('--betas', default=(0.9, 0.999), type=tuple, metavar='M', help='ADAM beta')
parser.add_argument('--epsilon', default=1e-8, type=float, metavar='M', help='ADAM epsilon for numerical stability')
parser.add_argument('--wd', '--weight-decay', default=0, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')

parser.add_argument('--resume', type=str, default=None, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,  help='distributed backend')
parser.add_argument('--seed', default=None, type=int,  help='seed for initializing training. ')
parser.add_argument('--multiprocessing-distributed', action='store_true',  help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the ' 
                         'fastest way to use PyTorch for either single node or'
                         'multi node data parallel training')

# propagation
parser.add_argument('--prop_time', type=int, default=12, help='number of propagation')
parser.add_argument('--prop_kernel', type=int, default=7,help='propagation kernel size')
parser.add_argument('--conf_prop', action='store_true',default=False)

parser.add_argument("--augment", type=str2bool, nargs='?', default=True, help="data augmentation")
parser.add_argument('--top_crop', type=int, default=0, help='top crop size for KITTI dataset')
parser.add_argument('--save', type=str, help='file name to save')
parser.add_argument('--gpus', type=str, help='visible GPUs')
parser.add_argument('--min_depth', type=float, default=1e-3, help='Minimum depth')
parser.add_argument('--max_depth', type=float, default=10.0, help='maximum depth')
parser.add_argument('--visualization', '-v', action='store_true',default=False)
parser.add_argument('--pretrain', type=str, default=None, help='ckpt path')

parser.add_argument('--lidar_lines', default=None, help='the extracted lidar lines [KITTI-DC]')
parser.add_argument('--num_sample', help='number of sparse samples [NYU-DC]')
parser.add_argument('--nyu_val_samples', type=str, default='500,200,100,32,8,4,1')
parser.add_argument('--kitti_val_lidars', type=str, default='64,32,16,8,4,2,1')

# Test for orininal size
parser.add_argument('--_480640', action='store_true', default=False)
parser.add_argument('--_kittiori', action='store_true', default=False)

# Crop
parser.add_argument('--garg_crop', action='store_true', default=False)
parser.add_argument('--eigen_crop', action='store_true', default=False)


parser.add_argument('--patch_height', type=int)
parser.add_argument('--patch_width', type=int)

parser.add_argument('--no_res_pre', action='store_true', default=False)
parser.add_argument('--init_scailing', action='store_true', default=False)

parser.add_argument('--minidataset', action='store_true', default=False)
parser.add_argument('--minidataset_fewshot', action='store_true', default=False)
parser.add_argument('--minidataset_fewshot_number', type=int, default=10)

parser.add_argument('--sigloss', action='store_true', default=False)

parser.add_argument('--sample_seed', type=int, default=77)
parser.add_argument('--loss', default='L1L2', type=str)

parser.add_argument('--resume_optim_sched', action='store_true', default=False)
parser.add_argument('--sp_max', type=int, default=500)
parser.add_argument('--change_backbone', action='store_true', default=False)

parser.add_argument('--sparsity', type=str)
parser.add_argument('--pattern', type=str, choices=('grid', 'random', 'original'))
parser.add_argument('--range', type=str, choices=('over30', 'under30', 'over3', 'under3','original'))
parser.add_argument('--exp_name', type=str, choices=('SparsityChange', 'PatternChange', 'RangeChange-Far',
                                                'SparsityChange-Rev', 'PatternChange-Rev', 'RangeChange-Near',
                                                'Debug', 'Ablation','multisample','PatternChangeFig2',
                                                'TestGrid','TestGrid2','TestFar','TestNear','TestSparse'))
parser.add_argument('--time_check', action='store_true', default=False)
parser.add_argument('--conf_select', action='store_true', default=False)
parser.add_argument('--main_test_model_comp', action='store_true', default=False)

parser.add_argument('--no_net', action='store_true', default=False)
parser.add_argument('--few_shot_result', action='store_true', default=False)
parser.add_argument('--affinity_visualization', action='store_true', default=False)

parser.add_argument('--kitti_spr_weight', type=str)


### chanhwi Add ###
parser.add_argument('--network',
                    type=str,
                    default='resnet34',
                    choices=('resnet18', 'resnet34'),
                    help='network name')
parser.add_argument('--from_scratch',
                    action='store_true',
                    default=False,
                    help='train from scratch')
parser.add_argument('--preserve_input',
                    action='store_true',
                    default=False,
                    help='preserve input points by replacement')
parser.add_argument('--affinity',
                    type=str,
                    default='TGASS',
                    choices=('AS', 'ASS', 'TC', 'TGASS'),
                    help='affinity type (dynamic pos-neg, dynamic pos, '
                         'static pos-neg, static pos, none')
parser.add_argument('--affinity_gamma',
                    type=float,
                    default=0.5,
                    help='affinity gamma initial multiplier '
                         '(gamma = affinity_gamma * number of neighbors')
parser.add_argument('--legacy',
                    action='store_true',
                    default=False,
                    help='legacy code support for pre-trained models')

# For visualization
parser.add_argument('--color_mode',
                    default='jet',
                    help='color mode')
parser.add_argument('--dilation_size',
                    default=8,
                    type=int,
                    help='dilation_size for sparse depth')

# VOID Dataset          
parser.add_argument('--void_sparsity',
                    default='500',
                    type=str,
                    help='void_sparsity')

# SUN_RGBD Dataset
parser.add_argument('--sensor_mode',
                    default='kv1',
                    type=str,
                    choices=('kv1', 'kv2', 'xtion', 'realsense'),
                    help='sensor_mode for SUN-RGBD')
parser.add_argument('--use_raw_depth_as_input',
                    default=False,
                    action='store_true',
                    help='Using raw depth as input for SUN-RGBD')
parser.add_argument('--sun_rgbd_each_sensor',
                    default=False,
                    action='store_true',
                    help='Ud')

args = parser.parse_args()