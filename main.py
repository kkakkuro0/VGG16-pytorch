# python
import os
import random
import logging
from pathlib import Path
from argparse import ArgumentParser

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from torchvision import datasets, transforms

# local
from models import VGG16
from train import train

# ETC
import numpy as np


def set_seed(random_seed):
    """
    재현 가능한 실험을 위해 무작위 수 생성기의 시드를 설정하는 역할을 합니다.
    이를 통해 PyTorch, NumPy, 그리고 Python의 기본 random 모듈의 무작위성을 제어합니다.
    또한, PyTorch의 CUDA 연산에서도 일관된 결과를 얻기 위해 관련 설정을 합니다.
    """
    torch.manual_seed(random_seed)  # CPU 시드 설정
    torch.cuda.manual_seed(random_seed)  # GPU 시드 설정
    torch.cuda.manual_seed_all(random_seed)  # 다중 GPU 시드 설정

    torch.backends.cudnn.deterministic = True  # CUDA의 결정적(deterministic) 결과를 보장합니다. 이 설정은 연산의 일관성을 유지하지만, 성능을 약간 희생할 수 있습니다.
    torch.backends.cudnn.benchmark = (
        False  # 입력 크기가 달라질 때마다 최적화된 커널을 찾지 않게 하여, 실행 시간의 일관성을 유지합니다.
    )

    np.random.seed(random_seed)  # Numpy 시드 설정
    random.seed(random_seed)  # python 시드 설정
    os.environ["PYTHONHASHSEED"] = str(random_seed)  # python 해시 시드 설정, 재현 가능한 해시값을 보장


def main(args):
    # path 설정
    save_path = args.save_path / args.exp_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Logger 설정
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s]\t %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
        filename=os.path.join(args.save_path, args.exp_name, "exp.log"),
        filemode="w",
    )

    # Default parameters
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    logging.info(f"Device: {device}")

    # Dataset 정의
    logging.info("Setting Dataset")
    train_dataset = datasets.CIFAR10(
        root="./datasets/", train=True, download=True, transform=transforms.ToTensor()
    )
    valid_dataset = datasets.CIFAR10(
        root="./datasets/", train=True, download=True, transform=transforms.ToTensor()
    )

    # DataLoader 정의
    logging.info("Setting DataLoader")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,  # 데이터를 로드하는 데 사용할 CPU의 개수입니다. 이는 데이터를 병렬로 로드하여 학습 속도를 높이는 데 도움이 됩니다.
        drop_last=True,  # 데이터셋의 크기가 배치 크기로 나누어 떨어지지 않을 경우, 마지막에 남는 데이터를 무시할지 여부입니다.
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    # Model 정의
    logging.info("Stting Model")
    model = VGG16(base_dim=64).to(device)

    start_epoch = 0

    # Loss function 정의
    logging.info("Setting Objective function")
    criterion = nn.CrossEntropyLoss()

    # Optimizer 정의
    logging.info("Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    train(
        device,
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        args.epochs,
        start_epoch,
        args.patience,
        save_path,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    # Default Parameters
    parser.add_argument("--seed", type=int, default=1226)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--exp_name", type=str, default="[Tag]exp")

    # Path
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--save-path", type=Path)

    # DataLoader
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)

    # Train
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    main(args)
