from typing import Optional

import torch

import soundfile

from torch.utils.data import Dataset
from pathlib import Path


# read all .wav files under data_path
def scanner(path: Path):
    for i in path.iterdir():
        if i.is_dir():
            yield from scanner(i)
        elif i.is_file():
            yield i


class AudioDataset(Dataset):
    """
    A Dataset to read audio
    """

    def __getitem__(self, index):
        for path, num in self.split_list:
            if index >= num:
                index -= num
            else:
                path = str(path)
                break
        else:
            raise IndexError("index out of dataset total slices")
        chunk, sr = soundfile.read(path, frames=self.chunk_size, start=index * self.split_size, dtype="float32")  # TODO: 改成了soundfile，未验证
        assert sr == self.sample_rate
        if len(chunk.shape) > 1:
            chunk = chunk.mean(-1)  # l,c -> l
        wf = torch.from_numpy(chunk)
        return wf

    def __len__(self):
        return self.index_num

    def __init__(
            self, sample_rate=8000,
            split_size=4 * 60 * 8000, front_overlap=2 * 60 * 8000, back_overlap=2 * 60 * 8000,
            data_path=Path("D:\\课程 2023暑假\\数据集\\"), data_suffix: Optional[str] = ".wav",
    ):
        data_path = Path(data_path)
        self.split_size = split_size
        self.chunk_size = split_size + front_overlap + back_overlap
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.index_num = 0
        self.split_list = []

        for path in scanner(data_path):
            if data_suffix is None or path.suffix == data_suffix:
                info = soundfile.info(str(path))
                if info.samplerate != self.sample_rate:
                    print(f"Skipped file not satisfying sample rate: {path}")
                    continue
                split_num = (info.frames -
                             (front_overlap + back_overlap)) // split_size
                split_num = int(split_num)
                self.index_num += split_num
                self.split_list.append((path, split_num))


class PretrainAudio(AudioDataset):
    pass  # TODO: 实现随机取负样本
