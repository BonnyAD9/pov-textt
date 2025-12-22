from pathlib import Path

import numpy as np
import torch
from PIL import Image

# import albumentations
from sklearn import model_selection, preprocessing
from torch.utils.data import DataLoader, RandomSampler

from .data_point import DataPoint


class Dataset:
    def __init__(self, dir: Path, data: list[DataPoint]):
        self.dir = dir
        self.data = data

    @staticmethod
    def parse_datasets(dataset: str) -> dict[str, "Dataset"]:
        data_path = Path(dataset) / "data"
        if not data_path.exists():
            return {}

        data = {}
        for sub_dir in data_path.iterdir():
            txt_file = list(sub_dir.glob("*.txt"))
            if not txt_file:
                continue
            data[sub_dir.name] = Dataset(sub_dir, DataPoint.parse_datapoints(txt_file[0]))

        return data

    def get_classes(self) -> set[str]:
        classes = set()
        for d in self.data:
            classes |= d.get_classes()
        return classes

    @staticmethod
    def make_dataloader(
        datasets: list["Dataset"], image_h: int, batch: int, device
    ):
        images = []
        orig_targets = []
        unused = {}
        for dataset in datasets:
            # Randomly choose one datapoint that will not be used in the
            # training from each dataset.
            idx = np.random.randint(0, len(dataset.data))
            unused[dataset.dir.name] = dataset.dir / "text_line_orig" / dataset.data[idx].image
            dataset.data[idx] = dataset.data[-1]
            dataset.data.pop()
            
            for d in dataset.data:
                images.append(dataset.dir / "text_line_orig" / d.image)
                orig_targets.append(d.text)

        targets = [[c for c in target] for target in orig_targets]
        flat_targets = [c for clist in targets for c in clist]

        encoder = preprocessing.LabelEncoder()
        encoder.fit(flat_targets)

        enc_targets = [encoder.transform(target) + 1 for target in targets]

        (train_imgs, test_imgs, train_targ, test_targ, _, test_orig_targ) = (
            model_selection.train_test_split(
                images,
                enc_targets,
                orig_targets,
                test_size=0.1,
                random_state=69,
            )
        )

        train_dataset = ClassifyDataset(train_imgs, train_targ)
        test_dataset = ClassifyDataset(test_imgs, test_targ, test_orig_targ)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch,
            collate_fn=lambda x: collate_fn_padd(x, device),
            shuffle=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch,
            collate_fn=lambda x: collate_fn_padd(x, device),
            shuffle=True,
        )

        return train_loader, test_loader, test_orig_targ, encoder.classes_, unused

    @staticmethod
    def join_classes(datasets: list["Dataset"]) -> set[str]:
        classes = set()
        for d in datasets:
            classes |= d.get_classes()
        return classes


def collate_fn_padd(batch, device):
    images = [item["images"] for item in batch]
    targets = [item["targets"] for item in batch]
    orig_targets = [item["orig_targets"] for item in batch]

    stride = 8
    input_lengths = torch.tensor(
        [img.shape[2] // stride for img in images], dtype=torch.long
    )
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    widths = [img.shape[2] for img in images]
    max_width = max(widths)

    batch_size = len(images)
    channels = images[0].shape[0]
    height = images[0].shape[1]

    padded_imgs = torch.zeros(batch_size, channels, height, max_width)

    for i, img in enumerate(images):
        w = img.shape[2]
        padded_imgs[i, :, :, :w] = img
    padded_imgs = padded_imgs

    padded_targets = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0
    )

    return {
        "images": padded_imgs.to(device),
        "targets": padded_targets.to(device),
        "input_lengths": input_lengths.to(device),
        "target_lengths": target_lengths.to(device),
        "orig_targets": orig_targets,
    }


class ClassifyDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets, orig_targ=None, resize_h=64):
        self.images = images
        self.targets = targets
        self.resize_h = resize_h
        self.orig_targ = orig_targ

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = prep_img(self.images[item], self.resize_h)
        target = self.targets[item]
        orig_targets = self.orig_targ[item] if self.orig_targ is not None else None

        return {
            "images": image,
            "targets": torch.tensor(target, dtype=torch.long),
            "orig_targets": orig_targets,
        }


def prep_img(file, res_h):
    image = Image.open(file).convert("RGB")

    if image._size[1] != res_h:
        w = res_h * image._size[0] // image._size[1]
        image = image.resize((w, res_h))

    image = np.array(image)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image /= 255.0
    image = 1 - torch.tensor(image, dtype=torch.float)
    return image
