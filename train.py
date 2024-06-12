# python
import logging
from tqdm import tqdm
import pathlib as Path

# torch
import torch
import torch.nn.functional as F


def train(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    epochs: int,
    start_epoch: int,
    patience: int,
    save_path: Path,
):
    # train parameter setting
    check_patience = 0
    best_acc = 0
    best_epoch = 0

    # Train
    model.to(device)
    for epoch in tqdm(range(start_epoch, epochs)):
        model.train()
        for step, (images, labels) in enumerate(data_loader):
            # gpu 연산을 위한 device 할당
            images, labels = images.to(device), labels.to(device)

            # Output
            outputs = model(images)

            # loss 계산
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Acc 계산
            outputs = torch.argmax(outputs, dim=1)
            train_acc = torch.eq(outputs, labels).sum().item() / len(outputs)

            # loss 출력
            if (step + 1) % 10 == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{epochs}] | Step [{step+1}/{len(data_loader)}], Loss: {round(loss.item(), 4)}, Acc: {round(train_acc, 4)}"
                )

            # validation
            valid_acc = validation(device, epochs, epoch, model, val_loader)

            # save best validation model
            if valid_acc > best_acc:
                logging.info(
                    f"Best Epoch [{best_epoch} > {epoch + 1}] | Best Accuracy [{best_acc} > {valid_acc}]"
                )
                best_acc = valid_acc
                best_epoch = epoch + 1
                check_patience = 0

                # save model
                output_path = save_path / "best_model.pt"
                torch.save(dict(epoch=epoch + 1, model=model), output_path)
            else:
                check_patience += 1

            # Early stopping
            if epoch > epochs // 2 and check_patience >= patience:
                break


def validation(
    device: torch.device,
    epochs: int,
    epoch: int,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
):
    # valid parameter setting
    acc = [0, 0]

    # Valid
    model.eval()
    with torch.no_grad():
        for step, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)
            acc[0] += torch.eq(outputs, labels).sum().item()
            acc[1] += len(outputs)

        valid_acc = acc[0] / acc[1]

        logging.info(f"Epoch [{epoch+1}/{epochs}] | Valid_Acc: {round(valid_acc, 4)}")

    return valid_acc
