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


class MLPModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 14 * 14)
        self.fc2 = nn.Linear(14 * 14, 8 * 8)
        self.fc3 = nn.Linear(8 * 8, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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
        case "celeba" | "celeba_iid" | "celeba_attractive":
            return get_efficient()
        case "cifar_10":
            return get_efficient(num_classes=10)
        case "celeba_different":
            return CelebaNet(num_classes=2)
        case "mnist_mlp":
            return MLPModel()
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")


def get_transforms(dataset: str):
    match dataset:
        case "emnist_non_iid" | "mnist" | "fashion" | "fashion_non_iid" | "mnist_mlp":
            return mnist_transforms
        case "celeba" | "celeba_attractive":
            from torchvision.models import EfficientNet_B0_Weights

            return transforms.Compose(
                [transforms.ToPILImage(), EfficientNet_B0_Weights.DEFAULT.transforms()]
            )
        case "celeba_iid" | "cifar_10":
            from torchvision.models import EfficientNet_B0_Weights

            return EfficientNet_B0_Weights.DEFAULT.transforms()
        case "celeba_different":
            return transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                ]
            )
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")


class CelebaNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(p=0.4)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout(p=0.4)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.bn3 = nn.BatchNorm2d(32)
        self.drop3 = nn.Dropout(p=0.4)

        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.bn4 = nn.BatchNorm2d(32)
        self.drop4 = nn.Dropout(p=0.4)

        self.fc = nn.Linear(32 * 14 * 14, num_classes)

    def forward(self, x):
        # One block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop1(F.relu(x))
        # print("First block", x.shape) # (None, 112, 112, 32)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop2(F.relu(x))
        # print("Second block", x.shape) # (None, 56, 56, 32)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop3(F.relu(x))
        # print("Third block", x.shape) # (None, 28, 28, 32)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop4(F.relu(x))
        # print("Fourth block", x.shape) # (None, 14, 14, 32)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x  # self.ac(x)
