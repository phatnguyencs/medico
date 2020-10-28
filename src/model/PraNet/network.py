import os 
import os.path as osp

import torch
import torch.nn as nn 
import torch.nn.functional as F

from model.PraNet.core.PraNet_Res2Net import PraNet

class MedicoNet(nn.Module):
    def __init__(self, cfg):
        super(MedicoNet, self).__init__()
        self.backbone = PraNet(pretrained_backbone=False)

    def set_eval(self):
        self.backbone.eval()
    
    def set_train(self):
        self.backbone.train()

    def forward(self, images):
        return self.backbone(images)

    def to_device(self, device=None):
        if device is None:
            self.backbone.cuda()
        else:
            self.backbone.to(device)
    
    def get_weights(self):
        return self.backbone.parameters()
    
    def get_backbone(self):
        return self.backbone
    
    def save_checkpoint(self, save_path, epoch, best_score, optimizer):
        if '.pt' not in save_path:
            save_path = osp.join(save_path, 'best_model.pt')
        
        dict_to_save = {
            'epoch': epoch, 
            'state_dict': self.backbone.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
        }

        torch.save(dict_to_save, save_path)
        print(f"saved checkpoint to {save_path}")

    def load_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path)
        self.backbone.load_state_dict(ckpt['state_dict'])
        return ckpt

    def get_prediction_from_output(self, output, raw_shape, is_numpy=False):
        res5, res4, res3, res2 = output
        res = res2 # choose res2 to get final result
        
        res = nn.Upsample(size=(raw_shape['height'], raw_shape['width']), mode='bilinear', align_corners=False)(res)
        res = res.sigmoid().squeeze()
        if is_numpy:
            res = res.data.cpu().numpy()
        res = (res-res.min()) / (res.max() - res.min() + 1e-8)
        return res

    def predict_mask(self, image, raw_shape):
        '''
        - Get prediction from of a single image
        Args:
            image: Tensor of batch size 1
            raw_shape: dictionary of {'height': H, 'width': W}
        '''
        self.set_eval()
        with torch.no_grad():
            output = self.backbone(image)
            return self.get_prediction_from_output(output, raw_shape)


