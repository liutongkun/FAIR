import torch
import torch.nn.functional as F
from data_loaderbhpf import MVTecTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unetskip import ReconstructiveSubNetwork
import os
import cv2
from utils import savefig
from msgms import MSGMSLoss
import kornia
from generic_util import trapezoid
from pro_curve_util import compute_pro

def see_img(data,dir,i,type): #B,C,H,W
    data=data.permute(0,2,3,1)
    data = data[0, :, :, :]
    data = data.cpu().numpy()
    data = data* 200
    data=data.astype('uint8')
    cv2.imwrite(dir+'/'+f'{type}{i}.png',data)

def see_img_heatmap(data,segresult,dir,i,type):  #B,C,H,W
    y2max = 255
    y2min = 0
    x2max = segresult.max()
    x2min = segresult.min()
    segresult = np.round((y2max - y2min) * (segresult - x2min) / (x2max - x2min) + y2min)
    segresult = segresult.astype(np.uint8)
    heatmap = cv2.applyColorMap(segresult, colormap=cv2.COLORMAP_JET)
    alpha = 0.15
    alpha2 = 0.3
    data=data.permute(0,2,3,1)
    data = data[0, :, :, :]
    data = data.cpu().numpy()
    data = data* 200
    data=data.astype('uint8')
    overlay = data.copy()
    data = cv2.addWeighted(heatmap, alpha2, overlay, 1 - alpha, 0, overlay)
    cv2.imwrite(dir+'/'+f'{type}{i}.png',data)


def mean_smoothing(amaps, kernel_size: int = 21) :

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)

def test(obj_names, mvtec_path, checkpoint_path, base_model_name,saveimages):
    obj_auroc_pixel_list = []
    obj_pro_list= []
    obj_auroc_image_list = []
    for obj_name in obj_names:
        img_dim = 256
        run_name = base_model_name+"_"+obj_name+'_'

        model = ReconstructiveSubNetwork(in_channels=1, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
        model.cuda()
        model.eval()



        dataset = MVTecTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=6)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        #calculate pro
        pro_gt=[]
        pro_out=[]
        anomaly_score_gt = []
        anomaly_score_prediction = []
        msgms = MSGMSLoss().cuda()
        with torch.no_grad():
            i=0
            if not os.path.exists(f'{savepath}/{obj_name}'):
                os.makedirs(f'{savepath}/{obj_name}')
            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                gray_gray = sample_batched["imagegray"].cuda()
                is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
                anomaly_score_gt.append(is_normal)
                true_mask = sample_batched["mask"]
                true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
                gray_rec = model(gray_gray)

                rec_lab = kornia.color.rgb_to_lab(gray_rec)
                ori_lab = kornia.color.rgb_to_lab(gray_batch)
                colordif = (ori_lab-rec_lab)*(ori_lab-rec_lab)
                colorresult = colordif[:,1,:,:]+colordif[:,2,:,:]
                colorresult = colorresult[None, :, :, :]*0.0003

                out_map = msgms(gray_rec, gray_batch, as_loss=False)+colorresult
                out_mask_averaged = mean_smoothing(out_map, 21)
                out_mask_averaged = out_mask_averaged.detach().cpu().numpy()

                if saveimages:
                    segresult=out_mask_averaged[0,0,:,:]
                    truemaskresult=true_mask[0,0,:,:]
                    see_img(gray_rec,f'{savepath}/{obj_name}/',i,'rec')
                    see_img(gray_batch,f'{savepath}/{obj_name}/',i,'orig')
                    see_img_heatmap(gray_batch,segresult,f'{savepath}/{obj_name}/',i,'hetamap')
                    savefig(gray_batch,segresult,truemaskresult,f'{savepath}/{obj_name}/'+f'segresult{i}.png',gray_rec)
                    i=i+1

                image_score = np.max(out_mask_averaged)
                anomaly_score_prediction.append(image_score)
                flat_true_mask = true_mask_cv.flatten()
                flat_out_mask = out_mask_averaged.flatten()
                total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
                total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
                mask_cnt += 1

                #for pro
                truegt=true_mask_cv[:,:,0]
                outresult=out_mask_averaged[0,0,:,:]
                pro_gt.append(truegt)
                pro_out.append(outresult)

        all_fprs, all_pros = compute_pro(
            anomaly_maps=pro_out,
            ground_truth_maps=pro_gt,
            num_thresholds=5000)

        au_pro = trapezoid(all_fprs, all_pros, x_max=0.3)
        au_pro /= 0.3
        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_pro_list.append(au_pro)
        print(obj_name)
        print("AUC Image:  " +str(auroc))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("PRO:  " +str(au_pro))

        print("==============================")

    print(run_name)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("PRO mean:  " + str(np.mean(obj_pro_list)))

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--saveimages', action='store_true')

    args = parser.parse_args()
    savepath = args.checkpoint_path
    #MVTec AD
    obj_list = [
                'screw',
                'bottle',
                'carpet',
                'leather',
                'pill',
                'transistor',
                'tile',
                'cable',
                'zipper',
                'toothbrush',
                'metal_nut',
                'hazelnut',
                'wood',
                'grid',
                'capsule'
                ]
    #VisA
    '''
    obj_list = ['capsules',
                'candle',
                'cashew',
                'chewinggum',
                'fryum',
                'macaroni2',
                'macaroni1',
                'pcb1',
                'pcb2',
                'pcb3',
                'pcb4',
                'pipe_fryum',
                ]
    '''
    with torch.cuda.device(args.gpu_id):
        test(obj_list,args.data_path, args.checkpoint_path, args.base_model_name,args.saveimages)
