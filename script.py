import torch
from torch.utils.data import DataLoader

from modules.trainer import SlicerTrainer, PreTrainer
from modules.model import slicer_small
from modules.pretrain import predictor_small
from modules.dataloader import AudioDataset


def pretrain():
    device = torch.device("cuda")
    load_model = False
    log_dir = r".\logs\pretrain"
    model_dir = r".\models\pretrain"

    train_data_path = r"E:\DataSet\audio\eureka\pv\cut\8k"
    valid_data_path = r"E:\DataSet\audio\eureka\pv\cut\8k"
    batch_size = 32
    sample_rate = 8000
    duration = 100 * 3 ** 6  # in original frames

    net = predictor_small().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), 1e-4)
    train_dataloader = DataLoader(
        AudioDataset(sample_rate=sample_rate, split_size=duration, front_overlap=0, back_overlap=0,
                     data_path=train_data_path),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True,
    )
    valid_dataloader = DataLoader(
        AudioDataset(sample_rate=sample_rate, split_size=duration, front_overlap=0, back_overlap=0,
                     data_path=valid_data_path),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
    )

    trainer = PreTrainer(
        model=net, optimizer=optimizer,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        load_models=load_model,
        device=device,

        log_dir=log_dir,
        model_dir=model_dir,
    )

    with trainer:
        trainer.train()

    print("end")


def train():  # TODO: 迁移pretrain模型
    device = torch.device("cuda")
    load_model = False
    log_dir = r'.\logs\train'
    model_dir = r'.\models\train'
    batch_size = 4
    sample_rate = 8000
    duration = 5 * 60 * sample_rate  # in original frames

    net = slicer_small().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), 1e-4)

    trainer = SlicerTrainer(
        net, optimizer,
        (
            (
                torch.randn((batch_size, duration)),
                torch.randint(0, 2, (batch_size, duration // net.encoder.down_sample_scale, 1), dtype=torch.float32)
            ) for _ in range(1000)
        ),
        (
            (
                torch.randn((batch_size, duration)),
                torch.randint(0, 2, (batch_size, duration // net.encoder.down_sample_scale, 1), dtype=torch.float32)
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


if __name__ == '__main__':
    pass
    # train()
    pretrain()
