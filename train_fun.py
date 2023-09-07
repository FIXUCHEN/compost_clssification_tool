import sys
from matplotlib import pyplot as plt
import torch
import numpy as np
import os
def one_step(model, optimizer, loss_fn, x, y,istrain = True,metric_func = None):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if istrain:
        # Zero gradients, perform a backward pass, and update
        # the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if metric_func is not None:
        metric = metric_func(y_pred,y)
    
        return loss.item(),metric
    else:
        return loss.item(),None
    
def one_epoch(epoch_index,model, optimizer, loss_fn, train_loader,metric_func = None,istrain = True,device = 'cpu'):
    if istrain:
        model.train()
        status = 'Train'
    else:
        model.eval()
        status = 'Valid'
    
    losses = []
    metrics = []
    loss_sum = 0
    for i,(x,y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        loss,metric = one_step(model,optimizer,loss_fn,x,y,istrain,metric_func)
        losses.append(loss)
        if metric_func is not None:
            metrics.append(metric)
        loss_sum += loss
            
        sys.stdout.write('\r{} EPOCH[{}]: [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
            status,epoch_index, (i+1) * len(x), len(train_loader.dataset),
            100. * (i+1) / len(train_loader), loss_sum/(i+1)))
        sys.stdout.flush()
    
    print()    
    return losses , metrics


def train(model, optimizer, loss_fn, train_loader, valid_loader, epochs, 
          metric_func = None,device = 'cpu',model_name = 'model',lr_scheduler = None):
    
    os.makedirs('saved_models', exist_ok=True)
    train_losses = []
    valid_losses = []
    train_metrics = []
    valid_metrics = []
    for epoch in range(1, epochs + 1):
        train_loss,train_metric = one_epoch(epoch, model, optimizer, loss_fn, train_loader,metric_func,istrain = True,device = device)
        valid_loss,valid_metric = one_epoch(epoch, model, optimizer, loss_fn, valid_loader,metric_func,istrain = False,device = device)
       
        mean_train_loss = np.mean(train_loss)
        mean_valid_loss = np.mean(valid_loss)
        train_losses.append(mean_train_loss)
        valid_losses.append(mean_valid_loss)
    
        if metric_func is not None:
            mean_train_metric = np.mean(train_metric)
            mean_valid_metric = np.mean(valid_metric)
            train_metrics.append(mean_train_metric)
            valid_metrics.append(mean_valid_metric)

        if min(valid_losses) == mean_valid_loss:
            torch.save(model.state_dict(), 'saved_models/{}.pth'.format(model_name))
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        # plot loss
        
        plt.plot(train_losses, label='Training loss')
        plt.plot(valid_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.savefig('{}_loss.png'.format(model_name))
        plt.clf()
        plt.close()
        if metric_func is not None:
            # plot metric
            plt.plot(train_metrics, label='Training metric')
            plt.plot(valid_metrics, label='Validation metric')
            plt.legend(frameon=False)
            plt.savefig('{}_metric.png'.format(model_name))
        
            plt.clf()
            plt.close()
    