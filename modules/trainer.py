
from abc import abstractmethod
from pathlib import Path

import torch
from torch import nn

from typing import Iterable, Tuple, Optional, Union, Protocol, List, TypeVar, Iterator

from torch.utils.tensorboard import SummaryWriter

PathLike = Union[Path, str]

T = TypeVar("T")


def get_autocast(device: Union[torch.device, str]):
    if isinstance(device, torch.device):
        device = device.type

    if hasattr(torch, "autocast"):
        return torch.autocast(device)
    else:
        if device == "cuda":
            return torch.cuda.amp.autocast(True)
        else:
            return torch.cpu.amp.autocast(True)


def inf_iterator(iterable: Iterable[T]) -> Iterator[T]:
    while True:
        for x in iterable:
            yield x


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            train_dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
            valid_dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]] = None,
            lr_scheduler=None,
            step_num: int = None,

            log_dir: PathLike = None,
            model_dir: PathLike = None,
            load_models=False,
            device: torch.device = None,
    ):
        """
        training script for a nn
        usage:
        with trainer:
            trainer.train()
        :param model: ddpm model
        :param optimizer: optimizer
        :param train_dataloader: datas for train set
        :param valid_dataloader: datas for valid set (optional)
        :param lr_scheduler:  lr scheduler (optional)
        :param step_num: train step number. -1 for infinite training.
        :param log_dir: output directory of tensorboard. default same as model_dir.
        :param model_dir: output directory of saved model. default same as log_dir.
        :param load_models: autoload model from model_dir. see also Trainer.load()
        :param device: training device, predict from model if not given.
        :return: None
        """
        if device is None:
            device = next(model.parameters()).device
        if device != torch.device("cpu"):
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        self.step_num = step_num
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = inf_iterator(valid_dataloader)
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        if model_dir is None:
            self.model_dir = self.log_dir
        if log_dir is None:
            self.log_dir = self.model_dir

        self.writer: Optional[SummaryWriter] = None
        self.scaler = scaler

        self.start_step = None

        if load_models:
            self.load()

    @abstractmethod
    def train(self):
        pass

    @property
    def model_path(self):
        return self.model_dir / f"{self.model.__class__.__name__}_model.pkl"

    @property
    def optimizer_path(self):
        return self.model_dir / f"{self.model.__class__.__name__}_optimizer.pkl"

    @property
    def lr_scheduler_path(self):
        return self.model_dir / f"{self.model.__class__.__name__}_lr_scheduler.pkl"

    def save_model(self, model_path: Optional[PathLike] = None):
        if model_path is None:
            model_path = str(self.model_path)
        if not (parent := Path(model_path).parent).exists():
            parent.mkdir(parents=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"model saved at {model_path}!")

    def save(
            self,
            model_path: Optional[PathLike] = None,
            optimizer_path: Optional[PathLike] = None,
            lr_scheduler_path: Optional[PathLike] = None,
    ):
        """
        save model and other training parameters
        """
        if model_path is None:
            model_path = str(self.model_path)
        if optimizer_path is None:
            optimizer_path = str(self.optimizer_path)
        if lr_scheduler_path is None:
            lr_scheduler_path = str(self.lr_scheduler_path)

        if not (parent := Path(model_path).parent).exists():
            parent.mkdir(parents=True)
        if not (parent := Path(optimizer_path).parent).exists():
            parent.mkdir(parents=True)
        if not (parent := Path(lr_scheduler_path).parent).exists():
            parent.mkdir(parents=True)

        # save model
        if self.model is not None:
            for param in self.model.parameters():
                if param.isnan().any().item():
                    print("nan parameter, model unsaved!")
                    break
            else:
                torch.save(self.model.state_dict(), model_path)
                print(f"model saved at {model_path}!")

        # save optimizer
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), optimizer_path)
            print(f"optimizer saved at {optimizer_path}!")

        # save lr_scheduler
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), lr_scheduler_path)
            print(f"lr_scheduler saved at {lr_scheduler_path}!")

    def load(self, model_path=None, optimizer_path=None, lr_scheduler_path=None):
        """
        load model and other training parameters
        """
        if model_path is None:
            model_path = self.model_path
        if optimizer_path is None:
            optimizer_path = self.optimizer_path
        if lr_scheduler_path is None:
            lr_scheduler_path = self.lr_scheduler_path

        if model_path.is_file() and self.model is not None:
            self.model.load_state_dict(torch.load(model_path), strict=False)
            print("model loaded!")

        if optimizer_path.is_file() and self.optimizer is not None:
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            if self.start_step is None:
                try:
                    self.start_step = self.optimizer.state.values().__iter__().__next__()["step"]
                    if isinstance(self.start_step, torch.Tensor):
                        self.start_step = self.start_step.long().item()
                    print("training step loaded!")
                except StopIteration:
                    pass
            print("optimizer loaded!")

        if lr_scheduler_path.is_file() and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(torch.load(lr_scheduler_path))
            print("lr_scheduler loaded!")

    def close(self):
        """
        save model and close summary writer
        """
        self.save()
        if self.writer is not None:
            self.writer.close()
        self.writer = None

    def __enter__(self):
        self.writer = SummaryWriter(self.log_dir.__str__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PreTrainer(Trainer):
    def __init__(
            self, *args,
            print_interval=10,
            val_interval=100,
            save_interval=2000,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.print_interval = print_interval
        self.val_interval = val_interval
        self.save_interval = save_interval

    def train(self, start_step=None):
        step = start_step
        if start_step is None:
            step = self.start_step
            if self.start_step is None:
                step = 0

        while True:
            for audios, labels, *_ in self.train_dataloader:
                if self.step_num is not None and step >= self.step_num:
                    return

                audios = audios.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                self.model.train()

                with get_autocast(self.device):
                    loss = self.model.forward(audios, labels)

                if self.scaler is None:
                    loss.backward()
                    loss = loss.detach().item()
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    loss = loss.detach().item()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # print loss state
                if step % self.print_interval == 0:
                    print(f"step: {step}, loss: {loss}")

                # do record
                if self.writer is not None:
                    # record loss state
                    self.writer.add_scalars("loss", {"train": loss}, step)

                    # do validation and record
                    if step % self.val_interval == 0 and self.valid_dataloader is not None:
                        self.model.eval()
                        with torch.no_grad():
                            audios, labels, *_ = self.valid_dataloader.__iter__().__next__()
                            audios = audios.to(self.device)
                            labels = labels.to(self.device)
                            with get_autocast(self.device):
                                loss = self.model.forward(audios, labels)
                            loss = loss.detach().item()

                            self.writer.add_scalars("loss", {"valid": loss}, step)

                        print(f"step: {step}, valid loss: {loss}")

                    # save model
                    if step % self.save_interval == 0:
                        self.save()

                step += 1


class SlicerTrainer(Trainer):
    def __init__(
            self, *args,
            print_interval=10,
            val_interval=100,
            save_interval=2000,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.print_interval = print_interval
        self.val_interval = val_interval
        self.save_interval = save_interval

    def train(self, start_step=None):
        step = start_step
        if start_step is None:
            step = self.start_step
            if self.start_step is None:
                step = 0

        while True:
            for audios, labels, *_ in self.train_dataloader:
                if self.step_num is not None and step >= self.step_num:
                    return

                audios = audios.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                self.model.train()

                with get_autocast(self.device):
                    loss = self.model.forward(audios, labels)

                if self.scaler is None:
                    loss.backward()
                    loss = loss.detach().item()
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    loss = loss.detach().item()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # print loss state
                if step % self.print_interval == 0:
                    print(f"step: {step}, loss: {loss}")

                # do record
                if self.writer is not None:
                    # record loss state
                    self.writer.add_scalars("loss", {"train": loss}, step)

                    # do validation and record
                    if step % self.val_interval == 0 and self.valid_dataloader is not None:
                        self.model.eval()
                        with torch.no_grad():
                            audios, labels, *_ = self.valid_dataloader.__iter__().__next__()
                            audios = audios.to(self.device)
                            labels = labels.to(self.device)
                            with get_autocast(self.device):
                                loss = self.model.forward(audios, labels)
                            loss = loss.detach().item()

                            self.writer.add_scalars("loss", {"valid": loss}, step)

                        print(f"step: {step}, valid loss: {loss}")

                    # save model
                    if step % self.save_interval == 0:
                        self.save()

                step += 1


