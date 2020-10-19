import torch
from torch import nn
import numpy as np
import json

class TSA_BCEDiceLoss(nn.Module):
    def __init__(self, cfg, num_steps, cuda=True):
        self.alpha = cfg.TRAIN.TSA.ALPHA
        self.current_step = 0
        self.num_classes = 2
        self.temperature = cfg.TRAIN.TSA.TEMPERATURE
        self.num_steps = num_steps
        self.cuda = cuda
        self.thres_history = []
        super().__init__()

    def step(self):
        self.current_step += 1
    
    def threshold(self, batch_size):
        # alpha_3: a = exp(5*(t/T-1)) 
        alpha_3 = torch.exp(torch.Tensor(batch_size*[self.alpha*(self.current_step/self.num_steps - 1)]))

        # alpha_1: a = 1 - exp(5* -t/T)
        alpha_1 = 1 - torch.exp(torch.Tensor(batch_size*[self.alpha*self.current_step/self.num_steps])) 
        
        thres = alpha_3*(1-1/self.num_classes) + 1/self.num_classes

        self.thres_history.append(thres[0].item())
        
        if (self.cuda):
            thres = thres.cuda()
        
        return thres

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        thres = self.threshold(batch_size)
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)

        # BCE loss
        bce_loss = nn.BCELoss(reduction='none')(pred, target).mean(dim=-1).double() # [B]

        # Dice Loss
        dice_coef = (2.0 * (pred * target).double().sum(dim=-1) + 1) / (
            pred.double().sum(dim=-1) + target.double().sum(dim=-1) + 1
        ) # [B, -1]

        mask = (dice_coef < thres).detach().double()
        # n_under_thres = torch.sum(mask).item()
        # n_over_thres = batch_size - n_under_thres
        # mask[mask == 0] = max(n_over_thres, n_under_thres)/batch_size
        # mask[mask == 1] += min(n_over_thres, n_under_thres)/batch_size
        
        loss = torch.mean((bce_loss + (1 - dice_coef))*mask)
        # print(f"dice_coeff: {dice_coef}")
        # print(f"thres: {thres}")
        # print(f"original loss: {torch.mean((bce_loss + (1 - dice_coef)))}")
        # print(f"tsa loss: {loss}")
        self.step()
        # loss = (1 - dice_coef)
        return loss


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inp, target):
        pred = inp.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        loss = bce_loss + (1 - dice_coef)
        # loss = (1 - dice_coef)
        return loss



# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ValidationTracker(MetricTracker):
    def __init__(self):
        self.all_scores = {
            'dice_coeff': 0.0,
            'IoU': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'F2': 0.0,
        }
        self.count = 0


    def reset(self):
        for k in self.all_scores.keys():
            self.all_scores[k] = 0.0
        self.count = 0
    
    def update(self, score_dict):
        self.count += score_dict['IoU'].shape[0]
        for k in score_dict.keys():
            self.all_scores[k] += np.sum(score_dict[k])
        
    def get_avg(self):
        res = self.all_scores
        for k in res.keys():
            res[k] /= self.count

        return res

    def to_json(self, json_file):
        res = self.get_avg()
        with open(json_file, 'w') as f:
            json.dump(res, f, indent=4)
            

# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def jaccard_index(output, target):
    intersection = torch.sum(output * target).cpu().numpy().tolist() # This will return a single float number
    union = (
        output.long().sum().data.cpu()
        + target.long().sum().data.cpu()
        - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(output, target):
    num_in_target = output.size(0)

    smooth = 1.0

    pred = output.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()

def calculate_f_score(precision, recall, base=2):
    return (1 + base**2)*(precision*recall)/((base**2 * precision) + recall)


def calculate_all_metrics_numpy(preds, gt, cpu=True):
    '''
    Args:
        preds, gt: [batch, C, H, W], here C = 1 for mask
    Return:
        {
            'dice_coeff': np.array[dice-coeff scores]
            'IoU': np.array[IoU scores]
            'precision': np.array[Precision scores]
            'recall': np.array[Recall scores]
            'F2': np.array[F2 scores]
        }
    '''
    batch_size = preds.size(0)
    y_preds = preds.reshape(batch_size, -1)
    y_true = gt.reshape(batch_size, -1)
    smooth = 1.0
    eps = 1e-5
    intersection = torch.sum(y_preds*y_true, dim=1)
    union = torch.sum(y_preds, 1) + torch.sum(y_true,1) - intersection

    iou = intersection/(union + eps)
    dice_coeff = (2.0*intersection + smooth)/(y_preds.sum(1) + y_true.sum(1) + smooth)
    precision = intersection / (y_true.sum(1)  + eps)
    recall = intersection / (y_preds.sum(1) + eps)
    f2 = calculate_f_score(precision, recall, 2)

    all_scores = {
        'dice_coeff': dice_coeff,
        'IoU': iou,
        'precision': precision,
        'recall': recall,
        'F2': f2
    }

    return all_scores

def calculate_all_metrics(preds, gt, cpu=True):
    '''
    Args:
        preds, gt: [batch, C, H, W], here C = 1 for mask
    Return:
        {
            'dice_coeff': np.array[dice-coeff scores]
            'IoU': np.array[IoU scores]
            'precision': np.array[Precision scores]
            'recall': np.array[Recall scores]
            'F2': np.array[F2 scores]
        }
    '''

    batch_size = preds.size(0)
    y_preds = preds.view(batch_size, -1)
    y_true = gt.view(batch_size, -1)
    smooth = 1.0
    eps = 1e-5
    intersection = torch.sum(y_preds*y_true, dim=1)
    union = torch.sum(y_preds, 1) + torch.sum(y_true,1) - intersection

    iou = intersection/(union + eps)
    dice_coeff = (2.0*intersection + smooth)/(y_preds.sum(1) + y_true.sum(1) + smooth)
    precision = intersection / (y_true.sum(1)  + eps)
    recall = intersection / (y_preds.sum(1) + eps)
    f2 = calculate_f_score(precision, recall, 2)

    all_scores = {
        'dice_coeff': dice_coeff,
        'IoU': iou,
        'precision': precision,
        'recall': recall,
        'F2': f2
    }

    if cpu == True:
        for k in all_scores.keys():
            all_scores[k] = all_scores[k].cpu().numpy()

    return all_scores
