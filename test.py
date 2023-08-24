import torch

from modules.trainer import SlicerTrainer
from modules.model import Slicer
from modules.dataloader import AudioDataset

if __name__ == '__main__':
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device("cuda")
    load_model = False
    log_dir = r'.\logs'
    model_dir = r'.\models'
    batch_size = 4
    sample_rate = 8000
    duration = 5 * 60

    net = Slicer(
        channels=(32, 48, 72, 108, 160, 224, 320),  # down sample 6 times (128ms in 16k SR)
        res_num=(1,) * 7,
        out_ch=1,
        rnn_channel=320,
    ).to(device)
    # net.stem.requires_grad_(False)
    # net.down_samples.requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), 1e-4)

    trainer = SlicerTrainer(
        net, optimizer,
        (
            (
                torch.randn((batch_size, duration * sample_rate)),
                torch.randint(0, 2, (batch_size, 1, duration * sample_rate // 3 ** 6), dtype=torch.float32)
            ) for _ in range(1000)
        ),
        (
            (
                torch.randn((batch_size, duration * sample_rate)),
                torch.randint(0, 2, (batch_size, 1, duration * sample_rate // 3 ** 6), dtype=torch.float32)
            ) for _ in range(1000)
        ),
        load_models=load_model,
        device=device,

        log_dir=log_dir,
        model_dir=model_dir,
    )

    with trainer:
        trainer.train()

    print("end")