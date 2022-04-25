"""main file for training and testing the model"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import time
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm  # for a progress bar for training so it looks cool
from PIL import Image
from torch.utils.data import DataLoader
from SEP_AI_model import Net
from SEP_AI_dataset import VOCDataset
from SEP_AI_utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bounds,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from SEP_AI_loss import Loss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 8  # lower this depending on your gpu capabilities while training i was using a gtx 1660 super 6gb
WEIGHT_DECAY = 0
EPOCHS = 100  # number of times the model will train for
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True  # make false if you dont want to load the saved checkpoint
LOAD_MODEL_FILE = "overfit.pth.tar"  # where the checkpoints are stored
IMG_DIR = "archive/images"  # where the dataset images are stored
LABEL_DIR = "archive/labels"  # where the dataset images are stored


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bounds):
        for t in self.transforms:
            img, bounds = t(img), bounds

        return img, bounds


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def main():
    model = Net(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = Loss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    # dataset for training, small dataset used otherwise it would take forever to train
    train_dataset = VOCDataset(
        "archive/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "archive/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    """uncomment this to test the webcam"""
    # cap = cv2.VideoCapture(0)
    # while True:

    #    re, frame = cap.read()  # reads an image
    #    image = cv2.flip(frame, 1)  # flips image to make it more natural for the viewer
    #    cv2.imshow('Image', image)  # shows the webcam

    #    if cv2.waitKey(10) & 0xFF == ord('q'):  # press q to close webcam
    #        break

    # cap.release()
    # cv2.destroyAllWindows()

    for epoch in range(EPOCHS):
        """uncomment this to test the object recognition"""
        #for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(8):
        #        bounds = cellboxes_to_boxes(model(x))
        #        bounds = non_max_suppression(bounds[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1, 2, 0).to("cpu"), bounds)
        #    import sys
        #    sys.exit()

        #
        pred_boxes, target_boxes = get_bounds(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")
        """uncomment this to train the model"""
        if mean_avg_prec > 0.993:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
