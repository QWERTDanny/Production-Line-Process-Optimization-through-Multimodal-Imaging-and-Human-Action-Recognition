import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
)
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

side_size = 256
mean = [0.432, 0.432, 0.432]
std = [0.291, 0.291, 0.291]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
clip_duration = (num_frames * sampling_rate) / frames_per_second
print(f"clip_duration: {clip_duration}")


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            # CenterCropVideo(crop_size),
            PackPathway(),
        ]
    ),
)


class AVerDataset(Dataset):
    def __init__(self, root, transform=transform):
        self.root = root
        self.dataset = pd.read_csv(os.path.join(self.root, "labels-train.csv"))
        self.video = self.dataset["video"].values
        self.series = self.dataset["series"].values
        self.label = self.dataset["label"].values

        self.class_to_idx = self.cls()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.clip = {}
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        label = self.label[idx]
        data = os.path.join(
            self.root,
            "train",
            f"S{str(self.series[idx]).zfill(6)}-{self.video[idx]}-A{self.label[idx]}.pt",
        )
        self.clip["video"] = torch.load(data)
        clip = transform(self.clip)
        clip = [i.to(device) for i in clip["video"]]

        return clip, label

    def cls(self):
        cls_file = os.path.join(self.root, "classes.txt")
        with open(cls_file) as f:
            classes = f.readlines()

        cls2idx = {}
        for c in classes:
            c = c.strip("\n").split(": ")
            cls2idx[int(c[0])] = c[1]
        return cls2idx

    def cls_list(self):
        cls_file = os.path.join(self.root, "classes.txt")
        with open(cls_file) as f:
            classes = f.readlines()
        classes_list = {}
        for c in classes:
            c = c.strip("\n").split(": ")
            classes_list[c[0]] = c[1]
        return classes_list
    
    def get_weights(self):
        class_wts = compute_class_weight(class_weight="balanced", 
                                        classes=np.unique(self.label),
                                        y=self.label)        
        print(class_wts)
        weights = torch.tensor(class_wts, dtype=torch.float)
        weights = weights.to(device)
        
        return weights

def get_dataloader(
    root, min_batch_size=2, test_ratio=0.2, transform=transform, seed=666
):
    dataset = AVerDataset(root=root, transform=transform)
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - test_size
    trainset, validset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )
    # test_size = int(0.5 * len(validset))
    # valid_size = len(validset) - test_size
    # validset, testset = random_split(validset, [valid_size, test_size])

    trainloader = DataLoader(
        dataset=trainset, batch_size=min_batch_size, shuffle=True, num_workers=0
    )
    validloader = DataLoader(
        dataset=validset, batch_size=min_batch_size, shuffle=False, num_workers=0
    )
    # testloader = DataLoader(
    #     dataset=testset, batch_size=min_batch_size, shuffle=False, num_workers=0
    # )

    del trainset, validset #, testset

    cls_list = dataset.cls_list()
    cls_wgt = dataset.get_weights()

    return cls_list, cls_wgt, trainloader, validloader #, testloader
