import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from animal_dataset import AnimalDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from model import CNN
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import shutil

def get_args():
    parser = ArgumentParser(description="CNN training")

    parser.add_argument("--root", "-n", type=str, default='animals', help="Root of the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard", help="Logging")
    parser.add_argument("--trained-models", "-tr", type=str, default="trained_models", help="Logging")
    parser.add_argument("--checkpoint", "-chkpt", type=str, default="trained_models/last_cnn.pt", help="checkpoint")

    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('run with gpu')
    else:
        device = torch.device("cpu")
    num_epochs = args.epochs
    batch_size = args.batch_size
    transform = Compose(
        [
            Resize((args.image_size, args.image_size)),
            ToTensor(),
        ]
    )

    train_dataset = AnimalDataset(root=args.root, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    print(train_loader)

    test_dataset = AnimalDataset(root=args.root, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter(args.logging)

    model = CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint["best_acc"]

    else:
        start_epoch = 0
        best_acc = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(train_loader)

        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            #forward
            output = model(images)
            loss = criterion(output, labels)
            progress_bar.set_description("epoch {}/{} iteration {}/{} loss {:.3f}".format(epoch + 1, num_epochs, i + 1, len(train_loader), loss))
            writer.add_scalar("Train/Loss", loss, epoch * len(train_loader) + i)
            #backward
            optimizer.zero_grad() # refresh buffer
            loss.backward() # gradient
            optimizer.step()  # update parameter

        model.eval()
        all_predictions = []
        all_labels = []
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            all_labels.extend(labels)
            # khong tinh gradient trong with torch.no_grad():
            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions,1)
                all_predictions.extend(indices)
                loss = criterion(predictions, labels)

        all_predictions = [prediction.item() for prediction in all_predictions]
        all_labels = [label.item() for label in all_labels]
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), class_names=test_dataset.categories, epoch=epoch)
        accuracy =  accuracy_score(all_labels, all_predictions)
        print("Epoch {}/{} accuracy {:.3f}".format(epoch + 1, num_epochs, accuracy))
        writer.add_scalar("Val/Accuracy", accuracy, epoch)
        checkpoint = {
            "best_acc": best_acc,
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            best_acc = accuracy
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))

        # print("Epoch {}/{}".format(epoch + 1, num_epochs))
        # print(classification_report(all_labels, all_predictions))