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
        checkpoint = os.path.join(tmp_dir, str(os.getpid()) + "checkpoint.pt")
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        epoch_time = time.time()
        model.train()
        train_losses = []

        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            optim.zero_grad()

            # Get data and move to device.
            # NOTE: The original expected shape was [B, 271, 20] with tokens along dim 2.
            # However, the input data is now stored with tokens in the second dimension, i.e.
            # the shape is [B, 20, 271]. Thus, no transposition is required.
            inputs_embeds = batch['inputs_embeds'].to(device)  # expected shape: [B, 20, 271]
            labels = batch['labels'].to(device)
            mask_label = batch['mask_label'].to(device)

            # Directly project the one-hot encoded input (last dim of size 271) to 256 dimensions.
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
                # Get validation data.
                inputs_embeds = batch['inputs_embeds'].to(device)  # expected shape: [B, 20, 271]
                mask_label = batch['mask_label'].to(device)
                labels = batch['labels'].to(device)

                # No transposition is needed as tokens are already in the second dimension.
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
            ax.plot(np.arange(1, epoch + 2), avg_train_loss, 'b', label='Training Loss')
            ax.plot(np.arange(1, epoch + 2), avg_valid_loss, 'orange', label='Validation Loss')
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
    projection=None  # Expecting inputs to be of shape [B, token_length, one_hot_dim] e.g., [B, 20, 271]
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

            # Expected input shape: [B, token_length, one_hot_dim] (e.g. [16, 20, 271])
            inputs_embeds = batch['inputs_embeds'].to(device)
            labels = batch['labels'].to(device)
            # If labels contain a token dimension (e.g., [B, 20, vel_size]), select the first token.
            if labels.dim() > 2:
                labels = labels[:, 0, :]

            # Apply the projection if provided to convert from one-hot (271) to embedding (256).
            if projection is not None:
                inputs_embeds = projection(inputs_embeds.float())

            # Forward pass.
            outputs = model(inputs_embeds=inputs_embeds)
            # If the model returns a sequence output (shape: [B, token_length, vel_size]),
            # select the first token's output as the prediction (commonly the [CLS] token).
            if outputs.logits.dim() == 3:
                pred = outputs.logits[:, 0, :]
            else:
                pred = outputs.logits

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
                if labels.dim() > 2:
                    labels = labels[:, 0, :]

                if projection is not None:
                    inputs_embeds = projection(inputs_embeds.float())

                outputs = model(inputs_embeds=inputs_embeds)
                if outputs.logits.dim() == 3:
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