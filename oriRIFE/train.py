import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from model.RIFE import Model
from dataset import *
from VimeoSeptuplet import *                                                                                                                                    
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from model.IFNet_BIVSR_7images import *
from model.VimeoSeptuplet import *
device = torch.device("cuda")

log_path = 'train_log'
intrain_path = 'intrain_log'


def add_BatchImageList(writer, name:str, imagelist: list, step):
    # Imagelist should be a list composed of image tensors with shape BCHW 
    # i.e. It should be N*(BCHW)
    imagelist_tensor = torch.stack(imagelist, dim=1)
    # imagelist_tensor: B*N*C*H*W
    B,N,C,H,W = imagelist_tensor.shape
    batchadd_images(writer=writer, name=name, imagelist_tensor=imagelist_tensor, batch=B, step=step, dataformats='NCHW')



def batchadd_images(writer, name, imagelist_tensor, batch, step, dataformats='NCHW'):
        imagelist_tensor.cpu()
        for i in range(batch):
        # add_images take NCHW
            grid = imagelist_tensor[i].squeeze(0)
            addname = name + '_batch_'+ str(i)
            writer.add_images(addname, grid, step, dataformats='NCHW')



def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, local_rank):
    if local_rank == 0:
        writer = SummaryWriter('train')
        writer_val = SummaryWriter('validate')
    else:
        writer = None
        writer_val = None
    step = 0
    nr_eval = 0
    dataset = VimeoDatasetSep('train')
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = dataset.__len__()
    dataset_val = VimeoDatasetSep('test')
    val_data = DataLoader(dataset_val, batch_size=4, pin_memory=True, num_workers=1)
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            # Changed for Septuplet 0 1 2 3 4 5 6
            b_data_gpu, _,h_data_gpu,w_data_gpu = data_gpu.shape
            display_all = data_gpu.view(b_data_gpu,7,3,h_data_gpu,w_data_gpu) 
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            learning_rate = get_learning_rate(step) * args.world_size / 4
            pred, info = model.update(data_gpu, learning_rate, training=True) # pass timestep if you are training RIFEm
            # preds: B*N*C*H*W
            '''
            return output_allframes, {
            'merged_tea': output_onlyteacher,
            'mask': mask_list,
            #'mask_tea': mask,
            'flow': flow_list[-1][:, :2],
            'flow_tea': flow_teacher_list[-1][:,:,2],
            'Sum_loss_context': Sum_loss_context,
            'Sum_loss_tea': Sum_loss_tea,
            'Sum_loss_distill': Sum_loss_distill
            }'''
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/Sum_loss_context', info['Sum_loss_context'], step)
                writer.add_scalar('loss/Sum_loss_tea', info['Sum_loss_tea'], step)
                writer.add_scalar('loss/Sum_loss_distill', info['Sum_loss_distill'], step)
            if step % 1000 == 1 and local_rank == 0:
                # display_gt = (display_gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                # mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                # pred = (pred[-2].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                # flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                # flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                # add_BatchImageList(writer, "pred in train", pred, step)
                batchadd_images(writer, "pred in train", pred, b_data_gpu, step, dataformats='NCHW')
                batchadd_images(writer, "origin images in train", display_all, b_data_gpu, step, dataformats='NCHW')

                
                # for i in range(5):
                    #imgs = np.concatenate((merged_img[i], pred[i], display_gt[i]), 1)[:, :, ::-1]
                    #writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    #writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                    # writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_context:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['Sum_loss_context']))
            step += 1
        if (epoch + 1) % 10 == 0:  # Adding 1 to make it human-readable (epochs start from 1)
            model_save_path = os.path.join(log_path, f'model_epoch_{epoch+1}.pth')
            # torch.save(model.state_dict(), model_save_path)
            model.save_model(model_save_path, local_rank)    
            print(f'Model saved at epoch {epoch+1}')
        nr_eval += 1
        print(nr_eval)
        if nr_eval % 1 == 0:
            print("Evaluate now\n")
            evaluate(model, val_data, step, local_rank, writer_val)
        if not os.path.exists(intrain_path):
            # os.makedirs(log_path)
            model.save_model(intrain_path, local_rank)    
        # dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    addImageFlag =1

    for i, data in enumerate(val_data):
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        timestep = timestep.to(device, non_blocking=True)
        # Changed for Septuplet 0 1 2 3 4 5 6
        b_data_gpu, _,h_data_gpu,w_data_gpu = data_gpu.shape
        display_all = data_gpu.view(b_data_gpu,7,3,h_data_gpu,w_data_gpu) 

        data_time_interval = time.time() - time_stamp
        time_stamp = time.time()
        # learning_rate = get_learning_rate(step) * args.world_size / 4
        # pred, info = model.update(data_gpu, learning_rate, training=True) # pass timestep if you are training RIFEm        
        gt = []
        for i in range(3):
            # img0 = data_gpu[:, 6*i:6*i+3]
            gt.append(data_gpu[:, 6*i+3:6*i+6])
            # img1 = data_gpu[:, 6*i+6:6*i+9]
        gt = torch.stack(gt,dim=1) # B*N*C*H*W
        with torch.no_grad():
            pred, info = model.update(data_gpu, training=False)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['Sum_loss_context'].cpu().numpy())
        loss_tea_list.append(info['Sum_loss_tea'].cpu().numpy())
        loss_distill_list.append(info['Sum_loss_distill'].cpu().numpy())
        if addImageFlag:
            addImageFlag = 0
            batchadd_images(writer_val, "pred in val", pred, b_data_gpu, nr_eval, dataformats='NCHW')
            batchadd_images(writer_val, "origin images in val", display_all, b_data_gpu, nr_eval, dataformats='NCHW')
        for j in range(gt.shape[0]):
            setpsnr = torch.tensor(0.0)
            for k in range(3):
                epsilon = 1e-8  # Small value to avoid log10(0)
                mse = torch.mean((gt[j][k] - pred[j][k]) ** 2).cpu().data  # Better to use **2 for clarity
                single_psnr = -10 * math.log10(mse + epsilon)
                # single_psnr = -10 * math.log10(torch.mean((gt[j][k] - pred[j][k]) * (gt[j][k] - pred[j][k])).cpu().data)
                setpsnr = setpsnr + single_psnr
            setpsnr = setpsnr / 3
            psnr_list.append(setpsnr)
            print("--\n")
            # psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            # psnr_list_teacher.append(psnr)
        # gt = (gt[][:,-3:].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # pred = (pred[:,-6:-3].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        # flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        # if i == 0 and local_rank == 0:
            # for j in range(10):
                # imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                # writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                # writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
        # eval_time_interval = time.time() - time_stamp
    print(".\n")
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.flush()
    # writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    args = parser.parse_args()
    
    torch.distributed.init_process_group(backend="nccl",world_size=1, rank=0)
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    train(model, args.local_rank)
        
        