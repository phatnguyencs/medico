import os 
import os.path as osp

import torch
import torch.nn as nn 

from model.ResUnet.core.res_unet_plus import ResUnetPlusPlus

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.backbone = ResUnetPlusPlus(cfg.MODEL.CHANNEL)
        

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



