from tacotron2.model import Tacotron2
from tacotron2.hparams import hparams
from tacotron2.loss_function import Tacotron2Loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tacotron2.data_utils import TextMelLoader, TextMelCollate
from tqdm import tqdm
import os
import re

def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(trainset, num_workers=1, shuffle=True,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def validate(model, criterion, valset, iteration, batch_size, collate_fn):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(valset, num_workers=1, shuffle=False,
                                batch_size=batch_size, pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

    model.train()
    print("Validation loss {}: {:9f}".format(iteration, val_loss))

def get_latest_checkpoint(output_dir):
    checkpoints = [f for f in os.listdir(output_dir) if re.match(r'checkpoint_\d+\.pth', f)]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda f: int(re.search(r'\d+', f).group()))
    return os.path.join(output_dir, latest_checkpoint)

# 初始化模型
model = Tacotron2(hparams)

# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 损失函数
criterion = Tacotron2Loss()

# 准备数据加载器
train_loader, valset, collate_fn = prepare_dataloaders(hparams)

epoch = 100

def train(model, criterion, optimizer, train_loader, valset, collate_fn, epochs, output_dir):
    iteration = 0
    checkpoint_path = get_latest_checkpoint(output_dir)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        model, optimizer, hparams.learning_rate, iteration = load_checkpoint(checkpoint_path, model, optimizer)
    else:
        print("No checkpoint found, starting from scratch.")

    model.train()
    for epoch in range(epochs):
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i, batch in enumerate(train_loader):
                model.zero_grad()
                x, y = model.parse_batch(batch)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                iteration += 1

                pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)

                if iteration % hparams.iters_per_checkpoint == 0:
                    print("The model has been trained for {} iterations".format(iteration))
                    validate(model, criterion, valset, iteration, hparams.batch_size, collate_fn)
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_{iteration}.pth")
                    save_checkpoint(model, optimizer, hparams.learning_rate, iteration, checkpoint_path)

if __name__ == "__main__":
    # 指定输出目录
    output_dir = hparams.output_dir
    train(model, criterion, optimizer, train_loader, valset, collate_fn, epoch, output_dir)