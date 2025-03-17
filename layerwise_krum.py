import argparse
import copy
from functools import partial
from typing import Dict, Hashable

import torch
from flex.data import Dataset
from flex.model import FlexModel
from flex.pool import FlexPool, fed_avg, init_server_model
from flex.pool.decorators import (
    collect_clients_weights,
    deploy_server_model,
    set_aggregated_weights,
)
from flexclash.pool import bulyan, multikrum
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from aggregators import (
    clip_by_norm,
    layerwise_bulyan,
    layerwise_krum,
    krum_cosine_similarity,
    krum_cosine_similarity_layerwise,
    bulyan_cosine_similarity,
    bulyan_cosine_similarity_layerwise,
    geomed,
    layerwise_geomed,
    cosine_geomed,
    layerwise_cosine_geomed,
)
from datasets import get_dataset, poison_dataset, poison_binary_dataset
from models import get_model, get_transforms

assert torch.cuda.is_available(), "CUDA not available"
device = "cuda"
n_gpus = torch.cuda.device_count()

round = 0
krum = partial(multikrum, m=1)


parser = argparse.ArgumentParser(description="Federated Learning with Krum Aggregation")
parser.add_argument(
    "--dataset",
    type=str,
    choices=[
        "emnist_non_iid",
        "fashion",
        "mnist",
        "celeba",
        "fashion_non_iid",
        "celeba_iid",
        "cifar_10",
    ],
    default="emnist_non_iid",
    help="Dataset to use",
)
parser.add_argument(
    "--clients", type=int, default=100, help="Number of clients per round"
)

parser.add_argument(
    "--lognum", type=int, default=0, help="Number of logs to keep in the tensorboard"
)

parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument(
    "--agg",
    type=str,
    choices=[
        "krum",
        "layerwise_krum",
        "fedavg",
        "bulyan",
        "layerwise_bulyan",
        "cosine_krum",
        "layerwise_cosine_krum",
        "layerwise_cosine_bulyan",
        "cosine_bulyan",
        "geomed",
        "layerwise_geomed",
        "cosine_geomed",
        "layerwise_cosine_geomed",
    ],
    default="fedavg",
    help="Aggregation operator to use",
)
parser.add_argument(
    "--labelflipping", action="store_true", help="Whether to use label flipping"
)
parser.add_argument(
    "--poisonedclients",
    type=int,
    default=0,
    help="Number of poisoned clients per round",
)
parser.add_argument(
    "--batchsize",
    type=int,
    default=256,
    help="Batch size to use for training on clients",
)
parser.add_argument(
    "--modelreplacement", action="store_true", help="Whether to use model replacement"
)
parser.add_argument(
    "--clipgradients", action="store_true", help="Whether to clip gradients"
)

parser.add_argument(
    "--f", type=int, default=1, help="Parameter for the Krum aggregation operator"
)
parser.add_argument(
    "--m", type=int, default=5, help="Parameter for the Bulyan aggregation operator"
)
args = parser.parse_args()

CLIENTS_PER_ROUND = args.clients - args.poisonedclients
POISONED_PER_ROUND = args.poisonedclients
EPOCHS = args.epochs

match args.agg:
    case "krum":
        AGG = krum
    case "layerwise_krum":
        AGG = layerwise_krum
    case "fedavg":
        AGG = fed_avg
    case "bulyan":
        AGG = bulyan
    case "layerwise_bulyan":
        AGG = layerwise_bulyan
    case "cosine_krum":
        AGG = krum_cosine_similarity
    case "layerwise_cosine_krum":
        AGG = krum_cosine_similarity_layerwise
    case "layerwise_cosine_bulyan":
        AGG = bulyan_cosine_similarity_layerwise
    case "cosine_bulyan":
        AGG = bulyan_cosine_similarity
    case "geomed":
        AGG = geomed
    case "layerwise_geomed":
        AGG = layerwise_geomed
    case "cosine_geomed":
        AGG = cosine_geomed
    case "layerwise_cosine_geomed":
        AGG = layerwise_cosine_geomed
    case _:
        raise ValueError(f"Unknown aggregation operator: {args.agg}")

if args.clipgradients:
    AGG = clip_by_norm(AGG)

if args.agg != "fedavg" and "geomed" not in args.agg:
    AGG = partial(AGG, f=args.f)

if "bulyan" in args.agg:
    AGG = partial(AGG, m=args.m)


def get_summary_writer_filename(args):
    parts = [
        args.agg,
        "SGD" if args.epochs == 1 else "",
        "label_flipping" if args.labelflipping else "",
        "layer_gaussian" if args.layergaussian else "",
        "gradients_clipped" if args.clipgradients else "",
        "little" if args.little else "",
        f"lognum_{args.lognum}" if args.lognum > 0 else "",
    ]
    return f"runs/{args.dataset}/" + "".join(filter(None, parts))


writer = SummaryWriter(get_summary_writer_filename(args))


flex_dataset, test_data = get_dataset(args.dataset)
clean_ids = list(flex_dataset.keys())
poisoned_ids = []

if args.labelflipping or args.layergaussian or args.little:
    assert (
        POISONED_PER_ROUND > 0
    ), "Using an attack requires at least one poisoned client per round"
    if args.labelflipping:
        if args.dataset == "celeba" or args.dataset == "celeba_iid":
            flex_dataset, poisoned_ids = poison_binary_dataset(flex_dataset, 0.2)
        else:
            classes_in_dataset = len(set(test_data.y_data))
            flex_dataset, poisoned_ids = poison_dataset(
                flex_dataset, classes_in_dataset, 0.2
            )
    clean_ids = [cid for cid in clean_ids if cid not in poisoned_ids]

print(f"Running options: {args}")

data_transforms = get_transforms(args.dataset)


@init_server_model
def build_server_model():
    server_flex_model = FlexModel()
    server_flex_model["model"] = get_model(args.dataset)
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}
    return server_flex_model


@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    new_flex_model = FlexModel()
    new_flex_model["model"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["server_model"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["criterion"] = copy.deepcopy(server_flex_model["criterion"])
    new_flex_model["optimizer_func"] = copy.deepcopy(
        server_flex_model["optimizer_func"]
    )
    new_flex_model["optimizer_kwargs"] = copy.deepcopy(
        server_flex_model["optimizer_kwargs"]
    )
    return new_flex_model


@set_aggregated_weights
def set_agreggated_weights_to_server(server_flex_model: FlexModel, aggregated_weights):
    dev = aggregated_weights[0].get_device()
    dev = "cpu" if dev == -1 else "cuda" + (f":{dev}" if dev > 0 else "")
    with torch.no_grad():
        weight_dict = server_flex_model["model"].state_dict()
        for layer_key, new in zip(weight_dict, aggregated_weights):
            writer.add_scalar(f"{args.dataset}/{layer_key}", new.norm(), round)
            weight_dict[layer_key].copy_(weight_dict[layer_key].to(dev) + new)


@collect_clients_weights
def get_clients_weights(client_flex_model: FlexModel):
    weight_dict = client_flex_model["model"].state_dict()
    server_dict = client_flex_model["server_model"].state_dict()
    dev = [weight_dict[name] for name in weight_dict][0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    return [
        (weight_dict[name] - server_dict[name].to(dev)).type(torch.float)
        for name in weight_dict
    ]


@collect_clients_weights
def get_labelflip_clients_weights(client_flex_model: FlexModel):
    weight_dict = client_flex_model["model"].state_dict()
    server_dict = client_flex_model["server_model"].state_dict()
    dev = [weight_dict[name] for name in weight_dict][0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    boosting_coef = CLIENTS_PER_ROUND / POISONED_PER_ROUND
    return [
        boosting_coef
        * (weight_dict[name] - server_dict[name].to(dev)).type(torch.float)
        for name in weight_dict
    ]


get_poisoned_clients_weights = get_labelflip_clients_weights


def train(client_flex_model: FlexModel, client_data: Dataset, rank=None):
    local_device = device if rank is None else "cuda:" + str(rank)
    train_dataset = client_data.to_torchvision_dataset(transform=data_transforms)
    client_dataloader = DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True
    )
    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    model = model.train()
    model = model.to(local_device)
    criterion = client_flex_model["criterion"]
    for _ in range(EPOCHS):
        for imgs, labels in client_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()


def obtain_accuracy(server_flex_model: FlexModel, test_data: Dataset):
    model = server_flex_model["model"]
    model.eval()
    test_acc = 0
    total_count = 0
    model = model.to(device)
    test_dataset = test_data.to_torchvision_dataset(transform=data_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=False
    )
    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_acc /= total_count
    return test_acc


def obtain_metrics(server_flex_model: FlexModel, _):
    data = test_data
    model = server_flex_model["model"]
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    test_dataset = data.to_torchvision_dataset(transform=data_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=False
    )
    losses = []
    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            if "celeba" in args.dataset:
                test_acc += (output.argmax(1) == target.argmax(1)).sum().cpu().item()
            else:
                pred = output.data.max(1, keepdim=True)[1]
                test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count

    return test_loss, test_acc


def clean_up_models(client_model: FlexModel, _):
    import gc

    client_model.clear()
    gc.collect()


def train_base(pool: FlexPool, n_rounds=100):
    clean_clients = pool.clients.select(lambda id, _: id in clean_ids)
    poisoned_clients = pool.clients.select(lambda id, _: id in poisoned_ids)
    if args.persistclients:
        selected_clean_clients = clean_clients.select(CLIENTS_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients, selected_clean_clients)

    for i in tqdm(range(n_rounds), args.agg):
        global round
        round = i
        selected_clean_clients = clean_clients.select(CLIENTS_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients, selected_clean_clients)

        selected_clean_clients.map(train)
        pool.aggregators.map(get_clients_weights, selected_clean_clients)

        if args.labelflipping:
            selected_poisoned_clients = poisoned_clients.select(1)
            pool.servers.map(copy_server_model_to_clients, selected_poisoned_clients)
            selected_poisoned_clients.map(train)
            pool.aggregators.map(
                (
                    get_poisoned_clients_weights
                    if (args.modelreplacement or args.layergaussian)
                    else get_clients_weights
                ),
                selected_poisoned_clients,
            )

        pool.aggregators.map(AGG)
        pool.aggregators.map(set_agreggated_weights_to_server, pool.servers)

        selected_clean_clients.map(clean_up_models)
        if args.labelflipping:
            selected_poisoned_clients.map(clean_up_models)

        loss, acc = pool.servers.map(obtain_metrics)[0]

        writer.add_scalar("Loss", loss, round)
        writer.add_scalar("Accuracy", acc, round)

        print(f"loss: {loss:7} acc: {acc:7}")


def run_server_pool():
    global flex_dataset
    global test_data
    flex_dataset["server"] = test_data
    pool = FlexPool.client_server_pool(flex_dataset, build_server_model)
    train_base(pool)


def main():
    run_server_pool()


if __name__ == "__main__":
    main()
