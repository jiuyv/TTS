import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tacotron2.model import Tacotron2
from tacotron2.hparams import hparams
from tacotron2.loss_function import Tacotron2Loss
from tacotron2.data_utils import TextMelLoader, TextMelCollate
import os
import re

def validate(model, criterion, valset, batch_size, collate_fn):
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
        val_loss /= (i + 1)
    return val_loss

def load_and_validate(weights_path, model, criterion, valset, batch_size, collate_fn):
    state_dict = torch.load(weights_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    val_loss = validate(model, criterion, valset, batch_size, collate_fn)
    return val_loss

if __name__ == "__main__":
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    model = Tacotron2(hparams).to("cuda" if torch.cuda.is_available() else "cpu")
    criterion = Tacotron2Loss()

    # 文件夹路径，包含所有权重文件
    weights_dir = r"output_pth"
    weights_files = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith('.pth')]

    steps = []
    losses = []

    for weights_path in sorted(weights_files, key=lambda x: int(re.search(r"checkpoint_(\d+)", x).group(1))):
        step = int(re.search(r"checkpoint_(\d+)", weights_path).group(1))
        val_loss = load_and_validate(weights_path, model, criterion, valset, hparams.batch_size, collate_fn)
        steps.append(step)
        losses.append(val_loss)
        print(f"Step: {step}, Validation loss: {val_loss}")

    pretrained_weights_path = "tacotron2_statedict.pt"
    pretrained_val_loss = load_and_validate(pretrained_weights_path, model, criterion, valset, hparams.batch_size, collate_fn)
    print(f"Pretrained model validation loss: {pretrained_val_loss}")


    # 绘制训练过程中的验证损失
    plt.plot(steps, losses, label='Validation Loss', marker='o')
    
    # 绘制预训练模型的验证损失
    plt.scatter([max(steps) + 1000], [pretrained_val_loss], color='red', label='Pretrained Model Loss', zorder=5)
    plt.annotate(f'Pretrained\nLoss: {pretrained_val_loss:.2f}', 
                 (max(steps) + 1000, pretrained_val_loss), 
                 textcoords="offset points", 
                 xytext=(-10,-15), 
                 ha='center', 
                 color='red')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Validation Loss vs. Training Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve_with_pretrained.png')  
    plt.show()