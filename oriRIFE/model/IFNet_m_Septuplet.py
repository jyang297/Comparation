import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *

eps = 1e-8


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask
    
class IFNet_m(nn.Module):
    def __init__(self):
        super(IFNet_m, self).__init__()
        self.block0 = IFBlock(6+1, c=240)
        self.block1 = IFBlock(13+4+1, c=150)
        self.block2 = IFBlock(13+4+1, c=90)
        self.block_tea = IFBlock(16+4+1, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale=[4,2,1], timestep=0.5, returnflow=False):
        timestep = (x[:, :1].clone() * 0 + 1) * timestep
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1, timestep), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        
        predictimage = torch.clamp(merged[2] + tmp, 0, 1)
        loss_pred = (((predictimage - gt) **2+eps).mean(1,True)**0.5).mean()
        #loss_pred = (((merged[2] - gt) **2).mean(1,True)**0.5).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()

        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill, loss_tea, loss_pred,



# Backbone of the 7 images for loop
class VSRbackbone(nn.Module):
    def __init__(self):
        super().__init__()
      
        self.Fusion = IFNet_m()


    def forward(self, allframes):

        # using septuplet: Batch*21*224*224
        Sum_loss_context = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_distill = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_tea = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        
        # LSTM LOOP
        # input：
        #   Forward:  img0 and img1 --> forward feature at 0.5
        #   Backward: img1 and img0 --> backward feature at 0.5
        # FirstExtractor: input 3 --> output 6    

        output_allframes = []
        output_onlyteacher = []
        flow_list = []
        flow_teacher_list = []
        mask_list = []
        # Now iterate through septuplet and get three inter frames
        for i in range(3):
            img0 = allframes[:, 6*i:6*i+3]
            gt = allframes[:, 6*i+3:6*i+6]
            img1 = allframes[:, 6*i+6:6*i+9]
            # check the order
            imgs_and_gt = torch.cat([img0,img1,gt],dim=1)
            
            #
            # def forward(self, x, scale=[4,2,1], timestep=0.5, returnflow=False):
            # return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill 
            flow, mask, merged, flow_teacher, merged_teacher, loss_distill, loss_tea, loss_pred = self.Fusion(imgs_and_gt)
            # flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(allframes)
            Sum_loss_distill += loss_distill 
            Sum_loss_context += 1/3 * loss_pred
            Sum_loss_tea +=loss_tea
            output_allframes.append(img0)
            # output_allframes.append(merged[2])
            output_allframes.append(merged)
            flow_list.append(flow)
            flow_teacher_list.append(flow_teacher)
            output_onlyteacher.append(merged_teacher)
            mask_list.append(mask)

            # The way RIFE compute prediction loss and 
            # loss_l1 = (self.lap(merged[2], gt)).mean()
            # loss_tea = (self.lap(merged_teacher, gt)).mean()

        img6 = allframes[:,-3:] 
        output_allframes.append(img6)
        output_allframes_tensors = torch.stack(output_allframes, dim=1)
        pass


        return flow_list, mask_list, output_allframes_tensors, flow_teacher_list, output_onlyteacher, Sum_loss_distill, Sum_loss_context, Sum_loss_tea
            

