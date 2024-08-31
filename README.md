# FAIR
This is the official implementation for the paper 'Frequency-aware Image Restoration for Industrial Visual Anomaly Detection'.  https://arxiv.org/abs/2309.07068
If you have any questions, you could contact ltk98633@stu.xjtu.edu.cn
![OverallFinal](https://github.com/liutongkun/FAIR/assets/59155313/86b7068a-0c01-4740-807c-272efbf3ad00)

## Preparation
The method is evaluated on:<br>
the MVTec AD dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad<br> 
the VisA dataset: https://github.com/amazon-science/spot-diff<br>

Code of FAIRm (using morphological gradient) can be found in https://github.com/liutongkun/EdgRec
## Training
#20240828:
The dataloader_bhpfnewaug module introduces NSA as an aditional synthesized anomaly, achieving a 99.1% image-level AUROC on the MVTec AD dataset.
### MVTec AD
The original code uses the DTD dataset to create synthesized anomalies, so you first need to download it <br>:
the DTD dataset (optional): https://www.robots.ox.ac.uk/~vgg/data/dtd/<br>
Then 

```python train.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 800 --data_path /home/b211-3090ti/Anomaly-Dataset/mvtec_ad/ --anomaly_source_path /home/b211-3090ti/Anomaly-Dataset/dtd/images --log_path /home/b211-3090ti/FAIR/checkpoints_mvtecad/ --checkpoint_path /home/b211-3090ti/FAIR/checkpoints_mvtecad/ --visualize ``` 

Change all the involved paths to your own paths  

### VisA
activate line 16 and line 76 in data_loaderbhpfnoDTD.py 

```self.images = sorted(glob.glob(root_dir+"/*/*.JPG"))```
```mask_file_name = file_name.split(".")[0]+".png" ``` 

### Without extra data
It's also feasible to train it without extra data, just activate line 3 in train.py: 

```from data_loaderbhpfnoDTD import MVTecTrainDataset``` 

Then 

```python train.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 800 --data_path /home/b211-3090ti/Anomaly-Dataset/mvtec_ad/ --log_path /home/b211-3090ti/FAIR/checkpoints_mvtecad/ --checkpoint_path /home/b211-3090ti/FAIR/checkpoints_mvtecad/ --visualize ```

## Testing
``` python test.py --gpu_id 0 --base_model_name FAIR_0.0001_800_bs8 --data_path /home/b211-3090ti/Anomaly-Dataset/mvtec_ad/ --checkpoint_path /home/b211-3090ti/FAIR/checkpoints_mvtecad/``` 

Change all the involved paths to your own paths 

If you want to visualize the results, add 

```--saveimages``` 

## Pre-trained models
MVTec AD

https://drive.google.com/file/d/1uwodfQZSXNq_TToio0r8gflUCg1EwSJv/view?usp=sharing

https://pan.baidu.com/s/1y8eocVy8FKf88nCkyd_4_A   jc5j 

VisA

https://drive.google.com/file/d/1DKYOqZAE-wbjRamk8xBhKB1Q2qswYxFR/view?usp=sharing

https://pan.baidu.com/s/1uS6etphpGiVDKTGPYjDdCg   qecd

## Acknowledgment
We use the codes from https://github.com/VitjanZ/DRAEM, https://github.com/taikiinoue45/RIAD, and https://www.mvtec.com/company/research/datasets/mvtec-3d-ad, https://github.com/hmsch/natural-synthetic-anomalies

A big thanks to their great work




