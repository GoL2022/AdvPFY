import torch
import torch.nn as nn
import numpy as np

def pgd_linf(input_img, target, model, atk_itr=8, obj_func=nn.CrossEntropyLoss(), eps=0.2, alpha=1/255, random_start=False, device=None):
    delta = torch.zeros_like(input_img, requires_grad=True)
    if random_start:
        delta.data = (torch.rand(input_img.shape).to(device) - 0.5) * 2 * eps
        
    delta = delta.to(device)
    for i in range(atk_itr):
        model.zero_grad()
        y_pred = model(input_img + delta)
        loss = obj_func(y_pred, target) 
        loss.backward()
        delta.data = (delta + torch.sign(delta.grad.data) * alpha).clamp(-eps,eps)
        delta.data = torch.clamp(input_img + delta, min=0, max=1) - input_img
        loss.zero_()
        delta.grad.zero_()
        
        
    adv_img = torch.clamp(input_img + delta.detach(), min=0, max=1)
    return adv_img.detach()


def vae_purify(data, model, atk_itr=256, eps=0.2, random_iteration=16, alpha=1/255, device=None):
    def purify(input_img, model, atk_itr=8, eps=0.2, alpha=1/255, random_start=False,  device=None):
        delta = torch.zeros_like(input_img, requires_grad=True)
        if random_start:
            delta.data = (torch.rand(input_img.shape).to(device) - 0.5) * 2 * eps

        for i in range(atk_itr):
            model.zero_grad()
            x_reconst, z, y_test, mu, log_var = model(input_img + delta, deterministic=False, classification_only=False)
            recons_loss = torch.sum((x_reconst - input_img - delta) ** 2)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()) 
            loss = recons_loss + kld_loss
            loss.backward()
            delta.data = (delta - torch.sign(delta.grad.data) * alpha).clamp(-eps,eps)
            delta.data = torch.clamp(input_img + delta, min=0, max=1) - input_img
            loss.zero_()
            delta.grad.zero_()

        adv_img = torch.clamp(input_img + delta.detach(), min=0, max=1)
        x_reconst, z, y_test, mu, log_var = model(adv_img, deterministic=True, classification_only=False)
        recons_loss = torch.sum((x_reconst - input_img - delta) ** 2, dim=(1, 2, 3))
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1) 
        loss = recons_loss + kld_loss
        loss_val = loss.cpu().detach().numpy()
        loss.zero_()
        return adv_img.detach(), loss_val
    
    purify_list = []
    loss_list = []
    for i in range(random_iteration):
        bool_random = True
        if i == 0:
            bool_random = False

        purify_data, loss_v = purify(data, model, atk_itr=atk_itr, eps=eps, 
                                     random_start=bool_random, alpha=(i % 2 + 1)/255.0,
                                     device=device)
        
        purify_list.append(purify_data)
        loss_list.append(loss_v)

    # select the run with minimum ELBO loss
    data_len = len(np.argmax(np.array(loss_list), axis=0))
    sel_purify = np.argmin(np.array(loss_list), axis=0) * data_len + np.arange(data_len)
    purify_data = torch.concat(purify_list)[sel_purify]
    return purify_data


def stae_purify(data, model, atk_itr=256, eps=0.2, random_iteration=16, alpha=1/255, device=None):
    def purify(input_img, model, atk_itr=8, eps=0.2, alpha=1/255, random_start=False,  device=None):
        delta = torch.zeros_like(input_img, requires_grad=True)
        if random_start:
            delta.data = (torch.rand(input_img.shape).to(device) - 0.5) * 2 * eps

        for i in range(atk_itr):
            model.zero_grad()
            x_reconst, z, y_test = model(input_img + delta, classification_only=False)
            recons_loss = torch.sum((x_reconst - input_img - delta) ** 2)
            loss = recons_loss
            loss.backward()
            delta.data = (delta - torch.sign(delta.grad.data) * alpha).clamp(-eps,eps)
            delta.data = torch.clamp(input_img + delta, min=0, max=1) - input_img
            loss.zero_()
            delta.grad.zero_()

        adv_img = torch.clamp(input_img + delta.detach(), min=0, max=1)
        x_reconst, z, y_test = model(adv_img, classification_only=False)
        recons_loss = torch.sum((x_reconst - input_img - delta) ** 2, dim=(1, 2, 3))
        loss = recons_loss
        loss_val = loss.cpu().detach().numpy()
        loss.zero_()
        return adv_img.detach(), loss_val
    
    purify_list = []
    loss_list = []
    for i in range(random_iteration):
        bool_random = True
        if i == 0:
            bool_random = False

        purify_data, loss_v = purify(data, model, atk_itr=atk_itr, eps=eps, 
                                     random_start=bool_random, alpha=(i % 2 + 1)/255.0,
                                     device=device)
        
        purify_list.append(purify_data)
        loss_list.append(loss_v)

    # select the run with minimum reconstruction loss
    data_len = len(np.argmax(np.array(loss_list), axis=0))
    sel_purify = np.argmin(np.array(loss_list), axis=0) * data_len + np.arange(data_len)
    purify_data = torch.concat(purify_list)[sel_purify]
    return purify_data