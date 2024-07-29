# [CVPR2024] Depth Prompting for Sensor-Agnostic Depth Estimation

### [<a href="https://arxiv.org/abs/2405.11867">Arxiv</a>] [<a href="https://github.com/JinhwiPark/DepthPrompting">Code</a>] [<a href="https://www.youtube.com/watch?v=TBXJPC6yeFY">Video</a>] [[Poster](https://drive.google.com/file/d/1gO4XqBlqTe2Lx3ZMrpeFwDsPfrTkSOSh/view?usp=drive_link)] [[Slide](https://drive.google.com/file/d/1iGptBZhA1VOnTw22rJWpoXO4q8qO8hGO/view?usp=drive_link)]

## The source code contains
 - Our implementation of the depth prompting module 
 - Depth prompting network for depth completion
 - Train code for NYU, KITTI dataset
 - Evaluation code for NYU, KITTI, VOID, SUN RGBD, IPAD, nuScenes

## Requirements
 - python==3.8.18
 - torch==1.9.0+cu111
 - torchvision==0.10.0+cu111
 - h5py
 - tqdm
 - scipy
 - matplotlib
 - nuscenes-devkit
 - imageio
 - pillow==9.5.0
```
 pip install opencv-python
 apt-get update
 apt-get -y install libgl1-mesa-glx -y
 apt-get -y install libglib2.0-0 -y
 pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

## Data Preparation

### Pretrained Weight and Data Split for NYU & KITTI
Download the weight files and json files at [URL](https://drive.google.com/drive/folders/1fzh0btV3Gd1qeY6Yr9qzACRo5HcSer4I?usp=sharing)

### NYU Depth V2 data Preparation
Please download the preprocessed NYU Depth V2 dataset in HDF5 formats provided by Fangchang Ma.
```bash
mkdir data; cd data
wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
mv nyudepthv2 nyudepth_hdf5
```
After that, you will get a data structure as follows:
```
nyudepthv2
├── train
│    ├── basement_0001a
│    │    ├── 00001.h5
│    │    └── ...
│    ├── basement_0001b
│    │    ├── 00001.h5
│    │    └── ...
│    └── ...
└── val
    └── official
        ├── 00001.h5
        └── ...
```

### KITTI data Preparation
Please download the KITTI DC dataset at the [KITTI DC Website](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).

For color images, KITTI Raw dataset is also needed, which is available at the [KITTI Raw Website](http://www.cvlibs.net/datasets/kitti/raw_data.php).

Please follow the official instructions (cf., devkit/readme.txt in each dataset) for preparation.

After downloading datasets, you should first copy color images, poses, and calibrations from the KITTI Raw to the KITTI DC dataset.
```
cd src/utils
python prepare_KITTI_DC.py --path_root_dc PATH_TO_KITTI_DC --path_root_raw PATH_TO_KITTI_RAW
```
After that, you will get a data structure as follows:
```
├── depth_selection
│    ├── test_depth_completion_anonymous
│    │    ├── image
│    │    ├── intrinsics
│    │    └── velodyne_raw
│    ├── test_depth_prediction_anonymous
│    │    ├── image
│    │    └── intrinsics
│    └── val_selection_cropped
│        ├── groundtruth_depth
│        ├── image
│        ├── intrinsics
│        └── velodyne_raw
├── train
│    ├── 2011_09_26_drive_0001_sync
│    │    ├── image_02
│    │    │     └── data
│    │    ├── image_03
│    │    │     └── data
│    │    ├── oxts
│    │    │     └── data
│    │    └── proj_depth
│    │        ├── groundtruth
│    │        └── velodyne_raw
│    └── ...
└── val
    ├── 2011_09_26_drive_0002_sync
    └── ...
```

### VOID data Preparation
Please download the VOID dataset at google drive.
```bash
https://drive.google.com/open?id=1kZ6ALxCzhQP8Tq1enMyNhjclVNzG8ODA
https://drive.google.com/open?id=1ys5EwYK6i8yvLcln6Av6GwxOhMGb068m
https://drive.google.com/open?id=1bTM5eh9wQ4U8p2ANOGbhZqTvDOddFnlI
```
which will give you three files void_150.zip, void_500.zip, void_1500.zip

After unzip, create the following data structure
```
VOID dataset
├── void_150-0
├── void_150-1
├── ...
├── void_500-0
├── void_500-1
├── ...
├── void_1500-0
├── void_1500-1
└── ...
```

### SUN RGBD data Preparation
Please download the SUN RGBD dataset at the [SUN RGBD Website](https://rgbd.cs.princeton.edu/)

After unzip, create the following data structure
```
SUNRGBD 
├── kv1
├── kv2
├── realsense
└── xtion
```

### IPAD data Preparation
Save the images and depths taken with the iPad in NPZ format (see this address for more information: [HNDR Website](https://github.com/princeton-computational-imaging/HNDR/tree/main/!DepthBundleApp))

After the above procedure, create a data structure that looks like this
```
ipad dataset
├── selected
│    ├── bundle-2023-09-25_15-48-09 
│			   └── frame_bundle.npz
│    ├── bundle-2023-09-25_15-48-13
│ 			    └── frame_bundle.npz
│    ├── bundle-2023-09-25_15-48-15
│    ├── ...
```

### nuScenes data Preparation
Please download the NUSCENES dataset at the [NUSCENES Website](https://www.nuscenes.org/nuscenes)

After unzip, create the following data structure
```
nuScenes dataset
├── maps
├── samples
├── sweeps
├── v1.0-mini
└── .v1.0-mini.txt
```

### [NYU Depth V2] Training & Testing
Please download the pretrained weights for any monocular depth model. 

We offer the pretrained weight files for [DepthFormer](https://drive.google.com/drive/folders/1kBb-70yvXjIQqGU28RNaHC2aZT1aTy7H). 

```shell
# Train
python main_DP.py --data_name NYU --dir_data {Dataset Directory} --pretrain ./pretrained/OURS/depthformer_nyu.pth --num_sample random --batch_size 32 --model_name depth_prompt_main --save OURS-NYU --patch_height 240 --patch_width 320 --prop_kernel 9 --prop_time 18 --init_scailing  --loss L1L2_SILogloss_init2 --gpus 0,1,2,3

# Test
python test_multiDataLoader.py --data_name NYU --dir_data {Dataset Directory} --model_name depth_prompt_main --pretrain ./pretrained/OURS/Depthprompting_depthformer_nyu.tar --prop_kernel 9 --conf_prop --prop_time 18 --patch_height 240 --patch_width 320 --nyu_val_samples 500,200,100,5,1  --init_scailing --gpu 0
```

### [KITTI Depth Completion] Training & Testing
```shell
# Train
python main_DP.py --data_name KITTIDC --dir_data {Dataset Directory} --pretrain ./pretrained/OURS/depthformer_kitti.pth --top_crop 100 --lidar_lines random_lidar --batch_size 1 --model_name depth_prompt_main --save OURS-KITTI --patch_height 240 --patch_width 1216 --prop_kernel 9 --prop_time 18 --conf_prop --init_scailing --loss L1L2_SILogloss_init --gpus 0

# Test
python test_multiDataLoader.py --data_name KITTIDC --dir_data {Dataset Directory} --model_name depth_prompt_main --pretrain ./pretrained/OURS/Depthprompting_depthformer_kitti.tar  --top_crop 0 --kitti_val_lidars 64,32,16,8,4,2,1,0 --prop_kernel 9 --prop_time 18 --conf_prop --init_scailing --gpu 1
```

### [VOID] Testing
```shell
# Test 
 python test_multiDataLoader.py --data_name VOID --void_sparsity {choose from 150, 500, 1500} --dir_data {Dataset Directory} --gpus 0 --model_name depth_prompt_main --pretrain {Pretrained weight}  --prop_kernel 9 --conf_prop --prop_time 18 --patch_height 240 --patch_width 320 --init_scailing
```

### [SUN RGBD] Testing
```shell
# Test
python test_multiDataLoader.py --data_name SUNRGBD --dir_data {Dataset Directory} --gpus 0 --model_name depth_prompt_main --pretrain {Pretrained weight}  --prop_kernel 9 --conf_prop --prop_time 18 --patch_height 240 --patch_width 320 --init_scailing --use_raw_depth_as_input
```

### [IPAD] Testing (Only visualization)
```shell
# Test
python test_multiDataLoader.py --data_name IPAD --dir_data {Dataset Directory} --save_dir {Save Directory} --gpu 0 --model_name depth_prompt_main --pretrain {Pretrained weight} --nyu_val_samples 0,1,5 --patch_height 228 --patch_width 304 --conf_select -v
```

### [nuScenes] Testing (Only visualization)
```shell
# Test
python test_multiDataLoader.py --data_name NUSCENE --dir_data {Dataset Directory} --save_dir {Save Directory} --gpu 0 --model_name depth_prompt_main --pretrain {Pretrained weight} --top_crop 0 --kitti_val_lidars 64 --prop_kernel 9 --prop_time 18 --conf_prop --init_scailing -v 
```

### Acknowledgement
This code is based on the original implementations: 
[CSPN](https://github.com/XinJCheng/CSPN)([paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Xinjing_Cheng_Depth_Estimation_via_ECCV_2018_paper.html)), 
[NLSPN](https://github.com/zzangjinsun/NLSPN_ECCV20)([paper](https://arxiv.org/abs/2007.10042)),
[GraphCSPN](https://github.com/xinliu20/GraphCSPN_ECCV2022)([paper](https://arxiv.org/abs/2210.10758)),
[HNDR](https://github.com/princeton-computational-imaging/HNDR/tree/main/!DepthBundleApp)([paper](https://light.princeton.edu/wp-content/uploads/2022/04/hndr.pdf))
[DepthFormer](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox)([paper](https://arxiv.org/pdf/2203.14211))


