#!/usr/bin/python

import argparse
import copy

import numpy as np
import torch
from rich.progress import track
from sklearn import metrics
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.crnn import CRNN
from src.dataset import Dataset

BLANK = "nada"


def main():
    parser = argparse.ArgumentParser(
        prog="pov-textt", description="Text transcript using CRNN"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Choose a mode to run"
    )

    train_parser = subparsers.add_parser("train", help="Train the CRNN model")
    train_parser.add_argument(
        "-d",
        "--dataset",
        default="dataset",
        type=str,
        help="Path to the dataset directory",
    )
    train_parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "-b",
        "--batch",
        default=64,
        type=int,
        help="Batch size used for training",
    )

    run_parser = subparsers.add_parser(
        "run", help="Runs the CRNN model on given image"
    )
    # TODO

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "run":
        pass


def train(args):
    data = Dataset.parse_datasets(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_h = 64

    train_loader, test_loader, test_original_targets, classes = (
        Dataset.make_dataloader([data[2]], img_h, args.batch, device)
    )

    classes = np.insert(np.array(classes), 0, BLANK, axis=0).tolist()
    print(classes)

    crnn = CRNN(img_h, len(classes), device).to(device)

    optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)

    best_wts = copy.deepcopy(crnn.state_dict())
    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss = train_fn(crnn, train_loader, optimizer, device)

        preds, test_loss = eval_fn(crnn, test_loader, device)
        text_preds = []
        for p in preds:
            text_preds.extend(ctc_decode_predictions(p, classes))

        accuracy = metrics.accuracy_score(test_original_targets, text_preds)
        print(f"epoch {epoch} loss: {train_loss} accuracy: {accuracy}")

        if accuracy > best_acc:
            best_acc = accuracy
            best_wts = copy.deepcopy(crnn.state_dict())

    return best_wts


def train_fn(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
):
    model.train()
    loss_sum = 0

    for data in track(data_loader, description="training"):
        dict_to_device(data, device)

        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / len(data_loader)


def eval_fn(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()
    with torch.no_grad():
        sum_loss = 0
        preds = []

        for data in data_loader:
            dict_to_device(data, device)

            batch_preds, loss = model(**data)
            sum_loss += loss.item()
            preds.append(batch_preds)
        return preds, sum_loss / len(data_loader)


def decode_predictions(
    predictions: Tensor, classes: list[str], pad_token: str = BLANK
) -> list[str]:
    pred = _prep_pred(predictions)

    texts = []
    for item in pred:
        text = ""
        for idx in item:
            token = classes[idx]
            if token != pad_token:
                text += classes[idx]
        texts.append(text)
    return texts


def ctc_decode_predictions(
    predictions: Tensor, classes: list[str], blank: str = BLANK
) -> list[str]:
    pred = _prep_pred(predictions.permute(1, 0, 2))

    texts = []
    for i in range(pred.shape[0]):
        text = ""
        batch_e = pred[i]

        for cl in batch_e:
            text += classes[cl]

        text = text.split(blank)
        text = [c for c in text if c != ""]
        text = [list(set(c))[0] for c in text]
        texts.append("".join(text))

    return texts


def _prep_pred(pred: Tensor):
    pred = torch.softmax(pred, 2)
    pred = torch.argmax(pred, 2)
    return pred.detach().cpu().numpy()


def dict_to_device(dic, dev: torch.device):
    for key, value in dic.items():
        dic[key] = value.to(dev)


if __name__ == "__main__":
    main()
