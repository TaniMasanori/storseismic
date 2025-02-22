from transformers import BertConfig, BertForMaskedLM
import transformers
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from radam import RAdam
import sys
import pandas as pd
import itertools

from storseismic.pytorchtools import EarlyStopping # https://github.com/Bjarten/early-stopping-pytorch

def run_pretraining(
    model, 
    optim, 
    loss_fn, 
    train_dataloader, 
    test_dataloader, 
    epochs, 
    device,
    projection,  # <-- new parameter
    tmp_dir=None, 
    patience=20, 
    plot=False, 
    f=None, 
    ax=None
):
    total_time = time.time()
    avg_train_loss = []
    avg_valid_loss = []
    time_per_epoch = []
    if patience is not None:
        checkpoint = os.path.join(tmp_dir, str(os.getpid())+"checkpoint.pt")
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        epoch_time = time.time()
        model.train()
        train_losses = []

        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            optim.zero_grad()
            inputs_embeds = batch['inputs_embeds'].to(device)
            labels = batch['labels'].to(device)
            mask_label = batch['mask_label'].to(device)

            # Use the projection layer
            inputs_256 = projection(inputs_embeds.float())

            outputs = model(inputs_embeds=inputs_256)
            select_matrix = mask_label.clone()
            loss = loss_fn(outputs.logits * select_matrix, labels.float() * select_matrix)

            loss.backward()
            optim.step()

            train_losses.append(loss.item())

        loop_valid = tqdm(test_dataloader, leave=True)
        losses_valid = 0
        with torch.no_grad():
            for i, batch in enumerate(loop_valid):
                # pull all tensor batches required for training

                inputs_embeds = batch['inputs_embeds'].to(device)
                mask_label = batch['mask_label'].to(device)
                labels = batch['labels'].to(device)

                # process

                # "projection" must be in scope; for example, pass it as an argument
                inputs_256 = projection(inputs_embeds.float())

                outputs = model(inputs_embeds=inputs_256)

                select_matrix = mask_label.clone()

                loss = loss_fn(outputs.logits * select_matrix, labels.float() * select_matrix)

                losses_valid += loss.item()

                loop_valid.set_description(f'Validation {epoch}')
                loop_valid.set_postfix(loss=loss.item())

        avg_train_loss.append(np.mean(train_losses))
        avg_valid_loss.append(losses_valid / len(test_dataloader))
        print("Epoch time: {:.2f} s".format(time.time() - epoch_time))
        time_per_epoch.append(time.time() - epoch_time)
        print("Total time elapsed: {:.2f} s".format(time.time() - total_time))
        print("---------------------------------------")
        
        if plot:
            ax.cla()
            ax.plot(np.arange(1, epoch+2), avg_train_loss,'b', label='Training Loss')
            ax.plot(np.arange(1, epoch+2), avg_valid_loss, 'orange', label='Validation Loss')
            ax.legend()
            ax.set_title("Loss Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Avg Loss")
            f.canvas.draw()

        if patience is not None:
            early_stopping(avg_valid_loss[-1], model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    if patience is not None:
        model.load_state_dict(torch.load(checkpoint))

    return model, avg_train_loss, avg_valid_loss, time_per_epoch

def run_denoising(model, optim, loss_fn, train_dataloader, test_dataloader, epochs, device, tmp_dir, patience=None, plot=False, f=None, ax=None):
    total_time = time.time()
    avg_train_loss = []
    avg_valid_loss = []
    time_per_epoch = []
    if patience is not None:
        checkpoint = os.path.join(tmp_dir, str(os.getpid())+"checkpoint.pt")
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint)

    for epoch in range(epochs):
        epoch_time = time.time()
        model.train()
        # setup loop with TQDM and dataloader
        loop_train = tqdm(train_dataloader, leave=True)
        losses_train = 0
        for i, batch in enumerate(loop_train):
            # initialize calculated gradients (from prev step)
            optim.zero_grad()

            # pull all tensor batches required for training

            # Mask outside loop
            inputs_embeds = batch['inputs_embeds'].to(device)
            labels = batch['labels'].to(device)     

            # process

            outputs = model(inputs_embeds=inputs_embeds.float())

            loss = loss_fn(outputs.logits, labels.float())

            outputs.loss = loss
            outputs.loss.backward()

            # update parameters
            optim.step()            

            losses_train += loss.item()

            loop_train.set_description(f'Epoch {epoch}')
            loop_train.set_postfix(loss=loss.item())

        loop_valid = tqdm(test_dataloader, leave=True)
        losses_valid = 0
        with torch.no_grad():
            for i, batch in enumerate(loop_valid):
                # pull all tensor batches required for training

                inputs_embeds = batch['inputs_embeds'].to(device)
                labels = batch['labels'].to(device)

                # process

                outputs = model(inputs_embeds=inputs_embeds.float())

                loss = loss_fn(outputs.logits, labels.float())

                losses_valid += loss.item()

                loop_valid.set_description(f'Validation {epoch}')
                loop_valid.set_postfix(loss=loss.item())

        avg_train_loss.append(losses_train / len(train_dataloader))
        avg_valid_loss.append(losses_valid / len(test_dataloader))
        print("Epoch time: {:.2f} s".format(time.time() - epoch_time))
        time_per_epoch.append(time.time() - epoch_time)
        print("Total time elapsed: {:.2f} s".format(time.time() - total_time))
        print("---------------------------------------")
        
        if plot:
            ax.cla()
            ax.plot(np.arange(1, epoch+2), avg_train_loss,'b', label='Training Loss')
            ax.plot(np.arange(1, epoch+2), avg_valid_loss, 'orange', label='Validation Loss')
            ax.legend()
            ax.set_title("Loss Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Avg Loss")
            f.canvas.draw()

        if patience is not None:
            early_stopping(avg_valid_loss[-1], model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    if patience is not None:
        model.load_state_dict(torch.load(checkpoint))

    return model, avg_train_loss, avg_valid_loss, time_per_epoch

def run_velpred(
    model, 
    optim, 
    loss_fn, 
    train_dataloader, 
    test_dataloader, 
    vel_size, 
    epochs, 
    device, 
    tmp_dir, 
    patience=None, 
    plot=False, 
    f=None, 
    ax=None,
    projection=None
):
    total_time = time.time()
    avg_train_loss = []
    avg_valid_loss = []
    time_per_epoch = []
    if patience is not None:
        checkpoint = os.path.join(tmp_dir, str(os.getpid()) + "checkpoint.pt")
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint)

    for epoch in range(epochs):
        epoch_time = time.time()
        model.train()
        loop_train = tqdm(train_dataloader, leave=True)
        losses_train = 0
        for i, batch in enumerate(loop_train):
            optim.zero_grad()

            inputs_embeds = batch['inputs_embeds'].to(device)
            labels = batch['labels'].to(device)
            
            # Debug prints
            print(f"\nInput shapes before processing:")
            print(f"inputs_embeds shape: {inputs_embeds.shape}")
            print(f"labels shape: {labels.shape}")
            
            # Select the first position from labels (or you could use mean across positions)
            labels = labels[:, 0, :]  # Now shape will be [16, 271]
            print(f"labels shape after selecting first position: {labels.shape}")
            
            if projection is not None:
                inputs_embeds = projection(inputs_embeds.float())
                print(f"inputs_embeds shape after projection: {inputs_embeds.shape}")

            outputs = model(inputs_embeds=inputs_embeds.float())
            print(f"outputs.logits shape: {outputs.logits.shape}")
            
            if outputs.logits.ndim == 3:
                pred = outputs.logits[:, 0, :]
            else:
                pred = outputs.logits
            
            print(f"pred shape: {pred.shape}")
            print(f"final labels shape: {labels.shape}")

            loss = loss_fn(pred, labels.float())

            loss.backward()
            optim.step()            

            losses_train += loss.item()
            loop_train.set_description(f'Epoch {epoch}')
            loop_train.set_postfix(loss=loss.item())

        loop_valid = tqdm(test_dataloader, leave=True)
        losses_valid = 0
        with torch.no_grad():
            for i, batch in enumerate(loop_valid):
                inputs_embeds = batch['inputs_embeds'].to(device)
                labels = batch['labels'].to(device)
                # Select the first position from labels (same as training)
                labels = labels[:, 0, :]
                
                if projection is not None:
                    inputs_embeds = projection(inputs_embeds.float())
                
                outputs = model(inputs_embeds=inputs_embeds.float())
                if outputs.logits.ndim == 3:
                    pred = outputs.logits[:, 0, :]
                else:
                    pred = outputs.logits
                    
                loss = loss_fn(pred, labels.float())
                losses_valid += loss.item()

                loop_valid.set_description(f'Validation {epoch}')
                loop_valid.set_postfix(loss=loss.item())

        avg_train_loss.append(losses_train / len(train_dataloader))
        avg_valid_loss.append(losses_valid / len(test_dataloader))
        print("Epoch time: {:.2f} s".format(time.time() - epoch_time))
        time_per_epoch.append(time.time() - epoch_time)
        print("Total time elapsed: {:.2f} s".format(time.time() - total_time))
        print("---------------------------------------")
        
        if plot:
            ax.cla()
            ax.plot(np.arange(1, epoch+2), avg_train_loss, 'b', label='Training Loss')
            ax.plot(np.arange(1, epoch+2), avg_valid_loss, 'orange', label='Validation Loss')
            ax.legend()
            ax.set_title("Loss Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Avg Loss")
            f.canvas.draw()

        if patience is not None:
            early_stopping(avg_valid_loss[-1], model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    if patience is not None:
        model.load_state_dict(torch.load(checkpoint))
        
    return model, avg_train_loss, avg_valid_loss, time_per_epoch