import os
import shutil
import torch

def convert_str_to_num(val, t):
    val = val.replace('\'', '')
    val = val.replace('\"', '')

    if t == 'int':
        val = [int(v) for v in val.split(',')]
    elif t == 'float':
        val = [float(v) for v in val.split(',')]
    else:
        raise NotImplementedError

    return val

class LRFactor:
    def __init__(self, args):
        
        decay = convert_str_to_num(args.decay, 'int')
        gamma = convert_str_to_num(args.gamma, 'float')

        self.decay = decay
        self.gamma = gamma

        assert len(self.decay) == len(self.gamma)
    def get_factor(self, epoch):
        for (d, g) in zip(self.decay, self.gamma):
            if epoch < d:
                return g
        return self.gamma[-1]

def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*","data","experiments","wandb","cifar","ipad_arkit_data")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))

def save_checkpoint(state, is_best, filename, save_dir):
    torch.save(state, save_dir + '/' + filename)
    if is_best:
        shutil.copyfile(save_dir + '/' + filename, save_dir + '/' + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.list=[]

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.list.append(val)


    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)