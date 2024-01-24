import os
from torchvision import transforms
from data_utils.data_loader import To_Tensor,  PETandCTNormalize,MRNormalize
from utils import hdf5_reader
from models.HDenseFormer import HDenseFormer_16
from torch.nn import functional as F
from utils import cal_score
import torch
import numpy as np
from torch.cuda.amp import autocast as autocast
import glob
from utils import cal_score
from models.HDenseFormer_2D import HDenseFormer_2D_32

def cal_steps(image_size):
        patch_size = (144,144,144)
        step_size = (72,72,72)

        steps = []

        for dim in range(len(image_size)):
            if image_size[dim] <= patch_size[dim]:
                steps_here = [
                    0,
                ]
            else:
                max_step_value = image_size[dim] - patch_size[dim]
                num_steps = int(np.ceil((max_step_value) / step_size[dim])) + 1
                actual_step_size = max_step_value / (num_steps - 1)
                steps_here = [
                    int(np.round(actual_step_size * i))
                    for i in range(num_steps)
                ]

            steps.append(steps_here)

        return steps

def eval_3d():
    # test data
    data_path = r'E:\Hecktor21\test_3d_seg'
    test_transformer = transforms.Compose([
            PETandCTNormalize(),  #2
            To_Tensor(num_class=2)  #6
        ])
    net=HDenseFormer_16(in_channels=2, n_cls=2,image_size=(144,144,144),transformer_depth=24)
    pathlist = glob.glob(os.path.join(data_path,'*.hdf5'))
    all_JI,all_Dice,all_95=[],[],[]

    for i in range(1,6):
        ck_path='ckpt/Hecktor21/3d_seg/fold{}/'.format(i)
        weight=os.listdir(ck_path)[-1]
        print(weight)
        # get weight
        checkpoint = torch.load(ck_path+weight)
        net.load_state_dict(checkpoint['state_dict'],strict=False)
        net = net.cuda()
        net.eval()
        JI,Dice,h95=[],[],[]
        for path in pathlist:
            image = hdf5_reader(path, 'ct')
            label = hdf5_reader(path, 'seg')
            sample = {'image': image, 'label': label}
            sample = test_transformer(sample)
            
            ori_image = np.asarray(sample['image'])

            new_image = np.expand_dims(ori_image, axis=0)

            aggregated_results = torch.zeros(
                [1, 2] +
                list(new_image.shape[2:]), ).cuda()
            aggregated_nb_of_predictions = torch.zeros(
                [1, 2] +
                list(new_image.shape[2:]), ).cuda()

            steps = cal_steps(ori_image.shape[1:])
            patch_size = (144,144,144)
            for x in steps[0]:
                lb_x = x
                ub_x = x + patch_size[0] if x + patch_size[
                    0] <= ori_image.shape[1] else ori_image.shape[1]
                for y in steps[1]:
                    lb_y = y
                    ub_y = y + patch_size[1] if y + patch_size[
                        1] <= ori_image.shape[2] else ori_image.shape[2]
                    for z in steps[2]:
                        lb_z = z
                        ub_z = z + patch_size[2] if z + patch_size[
                            2] <= ori_image.shape[3] else ori_image.shape[3]

                        image = ori_image[:, lb_x:ub_x, lb_y:ub_y,
                                            lb_z:ub_z]

                        image = np.expand_dims(image, axis=0)

                        data = torch.from_numpy(image).float()
                        data = data.cuda()

                        with autocast(False):
                            predicted_patch = net(data)
                            if isinstance(predicted_patch, tuple):
                                predicted_patch = predicted_patch[0]

                        if isinstance(predicted_patch, list):
                            predicted_patch = predicted_patch[0]

                        predicted_patch = F.softmax(predicted_patch, dim=1)
                        predicted_patch = F.interpolate(
                            predicted_patch,
                            (ub_x - lb_x, ub_y - lb_y, ub_z - lb_z))

                        aggregated_results[:, :, lb_x:ub_x, lb_y:ub_y,
                                            lb_z:
                                            ub_z] += predicted_patch  #* gaussian_importance_map
                        aggregated_nb_of_predictions[:, :, lb_x:ub_x,
                                                        lb_y:ub_y,
                                                        lb_z:ub_z] += 1

            output = aggregated_results / aggregated_nb_of_predictions

            # measure run dice
            output = torch.argmax(torch.softmax(
                output, dim=1), 1).detach().cpu().numpy().squeeze()  #N*H*W
            result = cal_score(output,label) #采用处理前的标签
            JI.append(result['Jaccard'])
            Dice.append(result['Dice'])
            h95.append(result['HausdorffDistance95'])
            
        all_JI.append(np.mean(JI))
        all_Dice.append(np.mean(Dice))
        all_95.append(np.mean(h95))

    print('Dice均值：',np.mean(all_Dice),'标准差：',np.std(all_Dice))
    print('HD95均值：',np.mean(all_95),'标准差：',np.std(all_95))
    print('JI均值：',np.mean(all_JI),'标准差：',np.std(all_JI))

class Normalize_2d(object):
    def __call__(self,sample):
        ct = sample['image']
        seg = sample['label']
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                if np.max(ct[i,j])!=0:
                    ct[i,j] = ct[i,j]/np.max(ct[i,j])
            
        new_sample = {'image':ct, 'label':seg}
        return new_sample

def eval_2d():
    # test data
    data_path = r'E:\PI-CAI22\test_2d_seg'
    test_transformer = transforms.Compose([
                MRNormalize(),
                To_Tensor(num_class=2)
            ])
    net=HDenseFormer_2D_32(in_channels=3, n_cls=2, image_size=(384,384), transformer_depth=24)
    pathlist = glob.glob(os.path.join(data_path,'*.hdf5'))
    all_JI,all_Dice,all_95=[],[],[]

    for i in range(1,6):
        ck_path='ckpt/PI-CAI22/2d_seg/fold{}/'.format(i)
        weight=os.listdir(ck_path)[-1]
        print(weight)
        # get weight
        checkpoint = torch.load(ck_path+weight)
        net.load_state_dict(checkpoint['state_dict'],strict=False)
        net = net.cuda()
        net.eval()
        JI,Dice,h95=[],[],[]
        for path in pathlist:
            image = hdf5_reader(path, 'ct')
            label = hdf5_reader(path, 'seg')
            sample = {'image': image, 'label': label}
            sample = test_transformer(sample)
            
            image = sample['image'].unsqueeze(0).cuda()

            with autocast(False):
                output = net(image)
                if isinstance(output,list):
                    output = output[0]
            # measure run dice
            output = torch.argmax(torch.softmax(
                output, dim=1), 1).detach().cpu().numpy().squeeze()  #N*H*W
            result = cal_score(output,label)
            JI.append(result['Jaccard'])
            Dice.append(result['Dice'])
            h95.append(result['HausdorffDistance95'])
            
        all_JI.append(np.nanmean(JI))
        all_Dice.append(np.nanmean(Dice))
        all_95.append(np.nanmean(h95))
        
    print('Dice均值：',np.mean(all_Dice),'标准差：',np.std(all_Dice))
    print('HD95均值：',np.mean(all_95),'标准差：',np.std(all_95))
    print('JI均值：',np.mean(all_JI),'标准差：',np.std(all_JI))

def eval_single2d():
    # test data
    data_path = r'E:\PI-CAI22\test_2d_seg'
    test_transformer = transforms.Compose([
                MRNormalize(),
                To_Tensor(num_class=2)
            ])
    net=HDenseFormer_2D_32(in_channels=3, n_cls=2, image_size=(384,384), transformer_depth=24)
    pathlist = glob.glob(os.path.join(data_path,'*.hdf5'))

    checkpoint = torch.load(r'ckpt\PI-CAI22\val_run_dice=0.54921.pth')
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    net = net.cuda()
    net.eval()
    JI,Dice,h95=[],[],[]
    for path in pathlist:
        image = hdf5_reader(path, 'ct')
        label = hdf5_reader(path, 'seg')
        sample = {'image': image, 'label': label}
        sample = test_transformer(sample)
        
        image = sample['image'].unsqueeze(0).cuda()

        with autocast(False):
            output = net(image)
            if isinstance(output,list):
                output = output[0]
        # measure run dice
        output = torch.argmax(torch.softmax(
            output, dim=1), 1).detach().cpu().numpy().squeeze()  #N*H*W
        result = cal_score(output,label)
        JI.append(result['Jaccard'])
        Dice.append(result['Dice'])
        h95.append(result['HausdorffDistance95'])
            
    print('Dice均值：',np.nanmean(Dice))
    print('HD95均值：',np.nanmean(h95))
    print('JI均值：',np.nanmean(JI))

eval_single2d()
