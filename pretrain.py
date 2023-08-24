import torch

from modules.trainer import SlicerTrainer
from modules.model import Slicer
from modules.dataloader import AudioDataset


class SlicerPreTrainer(SlicerTrainer):
    
    pass


if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device("cuda")
    load_model = False
    log_dir = r'.\logs'
    model_dir = r'.\models'
    batch_size = 4
    sample_rate = 16000

    net = Slicer().to(device)
    # net.stem.requires_grad_(False)
    # net.down_samples.requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), 1e-4)

    trainer = SlicerPreTrainer(
        net, optimizer,
        (

        ),
        load_models=load_model,
        device=device,

        log_dir=log_dir,
        model_dir=model_dir,
    )

    with trainer:
        trainer.train()

    print("end")
