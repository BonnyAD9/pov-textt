import copy
import itertools
from pathlib import Path

import Levenshtein
import numpy as np
import torch
from rich.progress import track
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.crnn import CRNN
from src.dataset import Dataset

BLANK = "∅"


def train(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = Dataset.parse_datasets(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_h = 64

    train_loader, test_loader, test_original_targets, classes = (
        Dataset.make_dataloader([data[2]], img_h, args.batch, device)
    )

    classes = np.insert(np.array(classes), 0, BLANK, axis=0).tolist()
    print(classes)

    crnn = CRNN(img_h, len(classes), device).to(device)
    max_acc = 0
    if args.model:
        checkpoint = torch.load(args.model, map_location=device)
        crnn.load_state_dict(checkpoint["model_state"])
        max_acc = checkpoint.get("acc", 0)

    print(max_acc)

    optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5
    )

    best_wts = copy.deepcopy(crnn.state_dict())

    for epoch in range(args.epochs):
        train_loss = train_fn(crnn, train_loader, optimizer, device)

        preds, test_loss = eval_fn(crnn, test_loader, device)
        text_preds = []
        for p in preds:
            text_preds.extend(ctc_decode_predictions(p, classes))

        for i in range(min(len(test_original_targets), 5)):
            print(test_original_targets[i], "->", text_preds[i])

        acc = 1 - get_cer(text_preds, test_original_targets)
        print(f"epoch {epoch} loss: {train_loss} acc: {acc}")
        save_model(crnn.state_dict(), classes, acc, output_dir / "last.pt")

        if acc > max_acc:
            max_acc = acc
            best_wts = copy.deepcopy(crnn.state_dict())
            save_model(best_wts, classes, acc, output_dir / "best.pt")

        scheduler.step(test_loss)

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
    predictions: Tensor, classes: list[str]
) -> list[str]:
    pred = _prep_pred(predictions.permute(1, 0, 2))

    texts = []
    blank_idx = 0
    for i in range(pred.shape[0]):
        collapsed = [k for k, _ in itertools.groupby(pred[i])]
        cleaned = [k for k in collapsed if k != blank_idx]

        text = "".join([classes[idx] for idx in cleaned])
        texts.append(text)

    return texts


def _prep_pred(pred: Tensor):
    pred = torch.softmax(pred, 2)
    pred = torch.argmax(pred, 2)
    return pred.detach().cpu().numpy()


def get_cer(preds: list[str], targets: list[str]) -> float:
    total_dist = 0
    total_len = 0

    for pred, target in zip(preds, targets):
        dist = Levenshtein.distance(pred, target)
        total_dist += dist
        total_len += len(target)

    return total_dist / total_len if total_len > 0 else 0.0


def save_model(state, classes, acc, file):
    checkpoint = {
        "model_state": state,
        "classes": classes,
        "acc": acc,
    }
    torch.save(checkpoint, file)


def dict_to_device(dic, dev: torch.device):
    for key, value in dic.items():
        dic[key] = value.to(dev)
