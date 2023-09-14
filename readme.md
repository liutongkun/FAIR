This is the official implementation for the paper 'Frequency-aware Image Restoration for Industrial Visual Anomaly Detection'. https://arxiv.org/abs/2309.07068
If you have any question, you could contact ltk98633@stu.xjtu.edu.cn

#Preparation
You need to download:
the MVTec AD dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
the VisA dataset: https://github.com/amazon-science/spot-diff
the DTD dataset (optional): https://www.robots.ox.ac.uk/~vgg/data/dtd/

#Training
Modify all the paths to your own paths.

--gpu_id
0
--obj_id
-1
--lr
0.0001
--bs
8
--epochs
800
--data_path
/home/b211-3090ti/Anomaly-Dataset/mvtec_ad/  
--anomaly_source_path
/home/b211-3090ti/Anomaly-Dataset/dtd/images
--log_path
/home/b211-3090ti/FAIR/checkpoints_mvtecad/
--checkpoint_path
/home/b211-3090ti/FAIR/checkpoints_mvtecad/
--visualize

#Testing
Modify all the paths to your own paths. If you want to visualize the reconstruction and segmentation results, you could additionally add '--saveimages'

--gpu_id
0
--base_model_name
FAIR_0.0001_800_bs8
--data_path
/home/b211-3090ti/Anomaly-Dataset/mvtec_ad/
--checkpoint_path
/home/b211-3090ti/FAIR/checkpoints_mvtecad/