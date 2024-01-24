import os 
import torch
from models.HDenseFormer import HDenseFormer_16
from torchvision import transforms
from data_utils.data_loader import DataGenerator, CropResize, To_Tensor, PETandCTNormalize, MRNormalize,Trunc_and_Normalize
from utils import dfs_remove_weight, hdf5_reader
import numpy as np
from torch.cuda.amp import autocast
from torch.nn import functional as F

# def cal_steps(image_size):
#         patch_size = (144,144,144)
#         step_size = (72,72,72)

#         steps = []

#         for dim in range(len(image_size)):
#             if image_size[dim] <= patch_size[dim]:
#                 steps_here = [
#                     0,
#                 ]
#             else:
#                 max_step_value = image_size[dim] - patch_size[dim]
#                 num_steps = int(np.ceil((max_step_value) / step_size[dim])) + 1
#                 actual_step_size = max_step_value / (num_steps - 1)
#                 steps_here = [
#                     int(np.round(actual_step_size * i))
#                     for i in range(num_steps)
#                 ]

#             steps.append(steps_here)

#         return steps

# def get_gaussian(patch_size, sigma_scale=1. / 8):
#     tmp = np.zeros(patch_size)
#     center_coords = [i // 2 for i in patch_size]
#     sigmas = [i * sigma_scale for i in patch_size]
#     tmp[tuple(center_coords)] = 1
#     gaussian_importance_map = gaussian_filter(tmp,
#                                                 sigmas,
#                                                 0,
#                                                 mode='constant',
#                                                 cval=0)
#     gaussian_importance_map = gaussian_importance_map / np.max(
#         gaussian_importance_map) * 1
#     gaussian_importance_map = gaussian_importance_map.astype(np.float32)

#     # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
#     gaussian_importance_map[gaussian_importance_map == 0] = np.min(
#         gaussian_importance_map[gaussian_importance_map != 0])

#     return gaussian_importance_map

# def inference_slidingwindow(save_path, net=None):

#     if net is None:
#         net = self.net

#     net = net.cuda()
#     net.eval()

#     # pathlist = glob.glob(os.path.join(test_path, '*.hdf5'))
#     pathlist=['E:/Hecktor21/train_3d_seg/0.hdf5']

#     test_transformer = transforms.Compose([
#         PETandCTNormalize(),  #2
#         To_Tensor(num_class=2)  #6
#     ])

#     patch_size = (144,144,144)

#     with torch.no_grad():
#         for step, path in enumerate(pathlist):
#             print(path)

#             image = hdf5_reader(path, 'ct')
#             label = hdf5_reader(path, 'seg')
#             sample = {'image': image, 'label': label}

#             # Transform
#             if test_transformer is not None:
#                 sample = test_transformer(sample)

#             ori_image = np.asarray(sample['image'])

#             new_image = np.expand_dims(ori_image, axis=0)

#             aggregated_results = torch.zeros(
#                 [1, 2] +
#                 list(new_image.shape[2:]), ).cuda()
#             aggregated_nb_of_predictions = torch.zeros(
#                 [1, 2] +
#                 list(new_image.shape[2:]), ).cuda()

#             steps = cal_steps(ori_image.shape[1:])

#             for x in steps[0]:
#                 lb_x = x
#                 ub_x = x + patch_size[0] if x + patch_size[
#                     0] <= ori_image.shape[1] else ori_image.shape[1]
#                 for y in steps[1]:
#                     lb_y = y
#                     ub_y = y + patch_size[1] if y + patch_size[
#                         1] <= ori_image.shape[2] else ori_image.shape[2]
#                     for z in steps[2]:
#                         lb_z = z
#                         ub_z = z + patch_size[2] if z + patch_size[
#                             2] <= ori_image.shape[3] else ori_image.shape[3]

#                         image = ori_image[:, lb_x:ub_x, lb_y:ub_y,
#                                             lb_z:ub_z]

#                         image = np.expand_dims(image, axis=0)

#                         data = torch.from_numpy(image).float()
#                         data = data.cuda()

#                         with autocast(False):
#                             predicted_patch = net(data)
#                             if isinstance(predicted_patch, tuple):
#                                 predicted_patch = predicted_patch[0]

#                         if isinstance(predicted_patch, list):
#                             predicted_patch = predicted_patch[0]
#                         predicted_patch = predicted_patch.float()  #N*C

#                         predicted_patch = F.softmax(predicted_patch, dim=1)
#                         predicted_patch = F.interpolate(
#                             predicted_patch,
#                             (ub_x - lb_x, ub_y - lb_y, ub_z - lb_z))
#                         # print(predicted_patch.size()[2:])

#                         gaussian_importance_map = torch.from_numpy(
#                             get_gaussian(
#                                 np.asarray(
#                                     predicted_patch.size()[2:]))).cuda()
#                         aggregated_results[:, :, lb_x:ub_x, lb_y:ub_y,
#                                             lb_z:
#                                             ub_z] += predicted_patch  #* gaussian_importance_map
#                         aggregated_nb_of_predictions[:, :, lb_x:ub_x,
#                                                         lb_y:ub_y,
#                                                         lb_z:ub_z] += 1
#                         # aggregated_nb_of_predictions[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += gaussian_importance_map

#             output = aggregated_results / aggregated_nb_of_predictions

#             # measure run dice
#             # output = output.detach().cpu().numpy().squeeze()
#             output = torch.argmax(torch.softmax(
#                 output, dim=1), 1).detach().cpu().numpy().squeeze()  #N*H*W

#             print(output.shape)

#             np.save(
#                 os.path.join(save_path,
#                                 path.split('/')[-1].split('.')[0] + '.npy'),
#                 output)

#             torch.cuda.empty_cache()

# net=HDenseFormer_16(in_channels=2, n_cls=2,image_size=(144,144,144),transformer_depth=24)
# inference_slidingwindow(save_path='.',net=net)
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np

# mhd_path = 'E:/picai_public_images_fold0/10022/10022_1000022_sag.mha'  # mhd文件需和同名raw文件放在同一个文件夹
# data = sitk.ReadImage(mhd_path)  # 读取mhd文件
# img_data = sitk.GetArrayFromImage(data)  # 获得图像矩阵
# print(img_data.shape)

data=np.load(r'E:\PI-CAI22\test_3d_seg\182.npy')
print(data.shape)
index=np.argmax(np.sum(data,(1,2))) #最大肿瘤区域的索引
plt.imshow(data[index],cmap='gray')
plt.show()