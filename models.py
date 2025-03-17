from torch import nn
from torchvision import transforms
from torch.nn import functional as F
import torch


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(14 * 14 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        return self.fc(x)


mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def get_efficient(num_classes=2):
    from torchvision.models import efficientnet_b0

    efficient_model = efficientnet_b0(weights="DEFAULT")
    efficient_model.classifier[1] = nn.Linear(
        efficient_model.classifier[1].in_features, num_classes
    )
    return efficient_model


def get_model(dataset: str):
    match dataset:
        case "emnist_non_iid" | "mnist" | "fashion" | "fashion_non_iid":
            return CNNModel()
        case "celeba" | "celeba_iid":
            return get_efficient()
        case "cifar_10":
            return get_efficient(num_classes=10)
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")


def get_transforms(dataset: str):
    match dataset:
        case "emnist_non_iid" | "mnist" | "fashion" | "fashion_non_iid":
            return mnist_transforms
        case "celeba":
            from torchvision.models import EfficientNet_B0_Weights

            return transforms.Compose(
                [transforms.ToPILImage(), EfficientNet_B0_Weights.DEFAULT.transforms()]
            )
        case "celeba_iid" | "cifar_10":
            from torchvision.models import EfficientNet_B0_Weights

            return EfficientNet_B0_Weights.DEFAULT.transforms()
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")
