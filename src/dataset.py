from pathlib import Path

import torch
# import albumentations
from sklearn import preprocessing, model_selection
from torch.utils.data import DataLoader
from .data_point import DataPoint
import numpy as np
from PIL import Image


class Dataset:
    def __init__(self, dir: Path, data: list[DataPoint]):
        self.dir = dir
        self.data = data

    @staticmethod
    def parse_datasets(dataset: str) -> list["Dataset"]:
        data_path = Path(dataset) / "data"
        if not data_path.exists():
            return []

        data = []
        for sub_dir in data_path.iterdir():
            txt_file = list(sub_dir.glob("*.txt"))
            if not txt_file:
                continue
            data.append(
                Dataset(sub_dir, DataPoint.parse_datapoints(txt_file[0]))
            )

        return data

    def get_classes(self) -> set[str]:
        classes = set()
        for d in self.data:
            classes |= d.get_classes()
        return classes

    @staticmethod
    def make_dataloader(datasets: list["Dataset"], image_h: int, batch: int, device):
        images = []
        orig_targets = []
        for dataset in datasets:
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
        test_dataset = ClassifyDataset(test_imgs, test_targ)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch,
            collate_fn = lambda x: collate_fn_padd2(x, device)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = batch,
            collate_fn = lambda x: collate_fn_padd2(x, device)
        )

        return train_loader, test_loader, test_orig_targ, encoder.classes_

    @staticmethod
    def join_classes(datasets: list["Dataset"]) -> set[str]:
        classes = set()
        for d in datasets:
            classes |= d.get_classes()
        return classes

def collate_fn_padd(batch, device):
    ## get sequence lengths
    # lengths = torch.tensor([ t['images'].shape[0] for t in batch ]).to(device)
    ## padd
    images = [ torch.Tensor(t['images']).to(device) for t in batch ]
    images = [torch.swapaxes(i, 0, 2) for i in images]
    images = torch.nn.utils.rnn.pad_sequence(images)
    images = images.permute(1, 3, 2, 0)

    targets = [ torch.Tensor(t['targets']).to(device) for t in batch ]
    targets = torch.nn.utils.rnn.pad_sequence(targets)

    ## compute mask
    # mask = (batch != 0).to(device)
    return { "images": images, "targets": targets }

def collate_fn_padd2(batch, device):
    #  ## get sequence lengths
    # lengths = torch.tensor([ t['images'].shape[0] for t in batch ]).to(device)
    # ## padd
    # batch = [ torch.Tensor(t['images']).to(device) for t in batch ]
    # batch = torch.nn.utils.rnn.pad_sequence(batch)
    # ## compute mask
    # mask = (batch != 0).to(device)
    # return batch, lengths, mask

    batch.sort(key=lambda x: x["images"].shape[2], reverse=True)

    images = [item["images"] for item in batch]
    targets = [item["targets"] for item in batch]

    widths = [img.shape[2] for img in images]
    max_width = max(widths)

    batch_size = len(images)
    channels = images[0].shape[0]
    height = images[0].shape[1]

    padded_imgs = torch.zeros(batch_size, channels, height, max_width)

    for i, img in enumerate(images):
        w = img.shape[2]
        padded_imgs[i, :, :, :w] = img

    padded_targets = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0
    )

    return {"images": padded_imgs, "targets": padded_targets}

class ClassifyDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets, resize_h = 64):
        self.images = images
        self.targets = targets
        self.resize_h = resize_h

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item]).convert("RGB")
        target = self.targets[item]

        if image._size[1] != self.resize_h:
            w = self.resize_h * image._size[0] // image._size[1]
            image = image.resize((w, self.resize_h))

        image = np.array(image)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)

        return {
            "images": image,
            "targets": torch.tensor(target, dtype=torch.long)
        }
