from datetime import datetime

import torch


def save_checkpoint(model, optimizer, file=f"CycleGAN_{datetime.now()}"):
    print("Saving checkpoint...")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, file)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location=None)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
