import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from util import *


def get_results(device, model, loader):
    model.eval()

    prec_sum = 0
    recall_sum = 0
    acc_sum = 0

    sm = torch.nn.Softmax(1)
    for step, (input_, label) in enumerate(loader):
        input_, label = input_.to(device), label.to(device)

        y_hat = model.forward(input_)

        # Apply softmax model output for y_test
        y_hat = sm(y_hat)
        y_hat = torch.argmax(y_hat, axis=1)

        # -- Get finally prediction
        y_hat = y_hat.to("cpu").detach().numpy().reshape(-1, 1)
        y_true = label.to("cpu").detach().numpy().reshape(-1, 1)

        # -- Confusion Matrix
        cm = confusion_matrix(y_true, y_hat)

        # -- Get rid of true division error
        np.seterr(invalid='ignore')

        # -- Get score for val set
        prec = np.diag(cm) / np.sum(cm, axis=0)
        prec = np.nan_to_num(prec)
        prec = np.mean(prec) * 100

        recall = np.diag(cm) / np.sum(cm, axis=1)
        recall = np.nan_to_num(recall)
        recall = np.mean(recall) * 100

        acc = np.diag(cm).sum() / cm.sum()
        acc = acc * 100

        prec_sum += prec
        recall_sum += recall
        acc_sum += acc
        break

    model.train()

    step += 1
    return (prec_sum / step, recall_sum / step, acc_sum / step)


def train_model(device, model, lr, epochs, train_loader, val_loader=None, en_schedular=True):
    # Clear log
    Logger.clear_all()

    # Move model to
    model.to(device)

    # Define loss
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)

    # Schedular
    schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs * 100, eta_min=0.001,
                                                                     T_mult=2, verbose=True)

    for ep in range(epochs):

        step_losses = []
        for step, (input_, label) in enumerate(train_loader):
            input_, label = input_.to(device), label.to(device)

            # Forward
            y_hat = model.forward(input_)

            # Loss
            loss = criterion(y_hat, label)
            step_losses.append(loss.item())

            # Get last lr
            last_lr = schedular.get_last_lr()[0]

            Logger.save_step_loss(loss.item())
            Logger.save_lr(last_lr)

            print(f"Epoch = {ep} | step = {step} | lr = {last_lr} | step_loss = {loss.item():.5f}")

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Update
            optimizer.step()

            if en_schedular:
                schedular.step()

        epoch_loss = np.mean(step_losses)

        Logger.save_epoch_loss(epoch_loss)

        train_prec, train_recall, train_acc = (0, 0, 0)

        if val_loader is not None:
            train_prec, train_recall, train_acc = get_results(device, model, val_loader)
            Logger.save_train_info(train_prec, train_recall, train_acc)
            print(f"Epoch = {ep}  | loss = {epoch_loss:.5f} | train_acc = {train_acc:.5f}")

        else:
            print(f"Epoch = {ep} | loss = {epoch_loss} | train_acc = {train_acc}")
