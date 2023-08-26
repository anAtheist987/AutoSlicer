from pathlib import Path

import torch
from torch.utils.data import DataLoader

from modules.trainer import SlicerTrainer, PreTrainer
from modules.model import slicer_small
from modules.pretrain import predictor_small
from modules.dataloader import AudioDataset, FineTuneAudio


def pretrain():
    device = torch.device("cuda")
    load_model = True
    log_dir = r"logs/pretrain"
    model_dir = r"models/pretrain"

    train_data_path = r"D:\课程 2023暑假\数据集\data_8k"
    valid_data_path = r"D:\课程 2023暑假\数据集\data_8k\歌回【阿梓从小就很可爱】"
    batch_size = 16
    sample_rate = 8000
    duration = 100 * 3 ** 6  # in original frames

    net = predictor_small().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), 1e-4)
    train_dataloader = DataLoader(
        AudioDataset(sample_rate=sample_rate, split_size=duration, front_overlap=0, back_overlap=0,
                     data_path=train_data_path),
        batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True,
    )
    valid_dataloader = DataLoader(
        AudioDataset(sample_rate=sample_rate, split_size=duration, front_overlap=0, back_overlap=0,
                     data_path=valid_data_path),
        batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=True,
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


def train():
    device = torch.device("cuda")
    load_model = True
    log_dir = r'.\logs\train'
    model_dir = r'.\models\train'

    train_data_path = r"D:\课程 2023暑假\数据集\阿梓 标注\temp"
    valid_data_path = r"D:\课程 2023暑假\数据集\阿梓 标注\temp"
    mark_path = r"D:\课程 2023暑假\数据集\阿梓 标注"
    batch_size = 12
    sample_rate = 8000
    duration = 2 * 60 * sample_rate  # in original frames
    front_overlap = 1 * 60 * sample_rate
    back_overlap = 1 * 60 * sample_rate

    net = slicer_small().to(device)
    net.encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), 1e-4)
    train_dataloader = DataLoader(
        FineTuneAudio(sample_rate=sample_rate, split_size=duration,
                      front_overlap=front_overlap, back_overlap=back_overlap,
                      data_path=train_data_path, mark_path=mark_path, vector_size=net.encoder.down_sample_scale),
        batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True,
    )
    valid_dataloader = DataLoader(
        FineTuneAudio(sample_rate=sample_rate, split_size=duration,
                      front_overlap=front_overlap, back_overlap=back_overlap,
                      data_path=valid_data_path, mark_path=mark_path, vector_size=net.encoder.down_sample_scale),
        batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=True,
    )

    trainer = SlicerTrainer(
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


def transform():
    pretrain_path = Path(r"models\pretrain\ContextPredictor_model.pkl")
    train_path = Path(r"models\train\Slicer_model.pkl")

    pretrain_model = predictor_small()
    train_model = slicer_small()

    pretrain_model.load_state_dict(torch.load(pretrain_path.__str__()))
    train_model.encoder = pretrain_model.encoder

    if not train_path.parent.exists():
        train_path.parent.mkdir(parents=True)
    torch.save(train_model.state_dict(), train_path.__str__())


if __name__ == '__main__':
    pass
    # pretrain()
    # transform()
    train()
