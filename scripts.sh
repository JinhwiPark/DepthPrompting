# NYU - Ours
# Train
python main.py --data_name NYU --dir_data '/workspace/data/DepthCompletion/nyudepthv2/' --pretrain ./pretrained/OURS/depthformer_nyu.pth --num_sample random --batch_size 32 --model_name depth_prompt_main --save OURS-NYU --patch_height 240 --patch_width 320 --prop_kernel 9 --prop_time 18 --init_scailing  --loss L1L2_SILogloss_init2 --gpus 0,1,2,3
# Test
python test_multiDataLoader.py --data_name NYU --dir_data '/workspace/data/DepthCompletion/nyudepthv2/' --model_name depth_prompt_main --pretrain ./pretrained/OURS/Depthprompting_depthformer_nyu.tar --prop_kernel 9 --conf_prop --prop_time 18 --patch_height 240 --patch_width 320 --nyu_val_samples 500,200,100,5,1  --init_scailing --gpu 0

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# KITTI - Ours
# Train
python main.py --data_name KITTIDC --dir_data '/workspace/data/DepthCompletion/kitti_DC/' --pretrain ./pretrained/OURS/depthformer_kitti.pth --top_crop 100 --lidar_lines random_lidar --batch_size 8 --model_name depth_prompt_main --save OURS-KITTI --patch_height 240 --patch_width 1216 --prop_kernel 9 --prop_time 18 --conf_prop --init_scailing --loss L1L2_SILogloss_init2 --gpus 4,5,6,7
# Test
python test_multiDataLoader.py --data_name KITTIDC --dir_data '/workspace/data/DepthCompletion/kitti_DC/' --model_name depth_prompt_main --pretrain ./pretrained/OURS/Depthprompting_depthformer_kitti.tar  --top_crop 0 --kitti_val_lidars 64,32,16,8,4,2,1,0 --prop_kernel 9 --prop_time 18 --conf_prop --init_scailing --gpu 1

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# VOID - Ours
# Test
python test_multiDataLoader.py --data_name VOID --void_sparsity 150 --dir_data /workspace/logs/void_data --gpus 1 --model_name depth_prompt_main --pretrain ./pretrained/OURS/Depthprompting_depthformer_nyu.tar --prop_kernel 9 --conf_prop --prop_time 18 --patch_height 240 --patch_width 320 --init_scailing

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# SUNRGBD - Ours
# Test
python test_multiDataLoader.py --data_name SUNRGBD --dir_data /workspace/data/SUNRGBD --gpus 1 --model_name depth_prompt_main --pretrain ./pretrained/OURS/Depthprompting_depthformer_nyu.tar --prop_kernel 9 --conf_prop --prop_time 18 --patch_height 240 --patch_width 320 --init_scailing --use_raw_depth_as_input

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# IPAD - Ours
# Test
python test_multiDataLoader.py --data_name IPAD --dir_data /workspace/logs/ipad_dataset/selected/ --save_dir /workspace/logs/visualization/IPAD-Comparison/_cvpr_submission --model_name depth_prompt_main --pretrain ./pretrained/OURS/Depthprompting_depthformer_nyu.tar --gpu 0 --nyu_val_samples 0,1,5 --patch_height 228 --patch_width 304 --conf_select -v

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# nuScenes - Ours
# Test
python test_multiDataLoader.py --data_name NUSCENE --dir_data /workspace/data/nuscene --save_dir /workspace/logs/visualization/NUSCENE-Comparison/_cvpr_submission --gpu 0 --model_name depth_prompt_main --pretrain ./pretrained/OURS/Depthprompting_depthformer_kitti.tar  --top_crop 0 --kitti_val_lidars 64 --prop_kernel 9 --prop_time 18 --conf_prop --init_scailing -v 
