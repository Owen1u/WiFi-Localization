###
 # @Descripttion: 
 # @version: 
 # @Contributor: Minjun Lu
 # @Source: Original
 # @LastEditTime: 2023-11-03 00:27:05
### 
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_port=29935 --use_env train.py --cfg /server19/lmj/github/wifi_localization/config/config.yaml