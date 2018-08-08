import os
from tqdm import tqdm
# import ipdb
        
import numpy as np
import torch as t
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from config import opt
from data.dataset import KaggleSalt
from models.FCN8s import  FCN8s

def test(**kwargs):
    opt.parse(kwargs)






def train(**kwargs):
    opt.parse(kwargs)

# step1: configure model (defined in models.py)
    model = FCN8s()
    device = t.device('cpu')
    if opt.use_gpu == True:
        device = t.device('cpu')

# step2: data preparation
    train_data = KaggleSalt(root = opt.train_data_root)
    train_dataloader = DataLoader( train_data, opt.batch_size,
                    shuffle = True, num_workers = opt.num_workers )
    
    val_data = KaggleSalt(opt.train_data_root)
    val_dataloader = DataLoader( val_data, opt.batch_size,
                    shuffle = True, num_workers = opt.num_workers )

# step3: criterion and optimizer
    criterion = t.nn.BCEWithLogitsLoss()
    lr = opt.lr
    optimizer = t.optim.RMSprop(model.parameters(), lr=lr, momentum = opt.momentum, weight_decay = opt.weight_decay)

    loss_pre = 1e10

# step4: training
    for epoch in range(opt.max_epoch):
        loss_now = 0
        for ii, (data, label) in tqdm(enumerate(train_dataloader)):

            data = data.to(device)
            label = t.Tensor(label.float())
            label = label.to(device)

            # Forward pass
            heatmap  = model(data)
            #import ipdb
            #ipdb.set_trace()
            loss = criterion(heatmap.reshape(-1), label.reshape(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update and visualize
            loss_now += loss.item()
        
        model.save()
        val_accuracy = val(model, val_dataloader)
        print('epoch:%d:   loss:%f / acc:%f' %(epoch, loss_now, val_accuracy) )

        if loss_now > loss_pre:
            lr = lr * opt.lr_decay
        loss_pre = loss_now


def val(model, dataloader):
    model.eval()

    total_ious = []

    device = t.device('cpu')
    if opt.use_gpu == True:
        device = t.device('cpu')

    for ii, (data, label) in tqdm(enumerate(dataloader)):
        data = data.to(device)
        label = label.to(device)

        output = np.array(model(data))
        heatmap = np.array(output > 0.5)

        iou_acc = (heatmap * label).sum()/(label.sum() + heatmap.sum())

        total_ious.append(iou_acc)

    
    return np.array(total_ious).sum() / len(total_ious)






if __name__ == '__main__':
    import fire
    fire.Fire()
