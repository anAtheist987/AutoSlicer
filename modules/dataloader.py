from typing import Optional, Union

import torch
import pandas as pd
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


def read_time(time: str):
    secs = 0
    split_len = time.strip().split(':').__len__()
    if split_len == 1:
        secs = float(time)
    elif split_len == 2:
        m, s = time.strip().split(':')
        secs = int(m) * 60 + float(s)
    elif split_len == 3:
        h, m, s = time.strip().split(':')
        secs = int(h) * 3600 + int(m) * 60 + float(s)

    return secs


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
        chunk, sr = soundfile.read(path, frames=self.chunk_size, start=index * self.split_size, dtype="float32")
        soundfile.write(r"D:\课程 2023暑假\new.wav", chunk, sr)
        assert sr == self.sample_rate
        if len(chunk.shape) > 1:
            chunk = chunk.mean(-1)  # l,c -> l
        wf = torch.from_numpy(chunk)
        return wf

    def __len__(self):
        return self.index_num

    def __init__(
            self, data_path: Union[Path, str], sample_rate=8000,
            split_size=4 * 60 * 8000, front_overlap=2 * 60 * 8000, back_overlap=2 * 60 * 8000,
            data_suffix: Optional[str] = ".wav",
    ):
        data_path = Path(data_path)
        self.split_size = split_size
        self.chunk_size = split_size + front_overlap + back_overlap  # 每个chunk中的样本点个数
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


class FineTuneAudio(AudioDataset):
    def __getitem__(self, index):
        wf = super().__getitem__(index)
        name = ''
        for path, num in self.split_list:
            if index >= num:
                index -= num
            else:
                name = path.name
                break
        else:
            raise IndexError("index out of dataset total slices")

        path = Path(self.mark_path.__str__() + "//" + name[:-4] + ".csv")
        if path.exists():
            mark_file = pd.read_csv(path, sep=self.sep, )
        else:
            raise FileNotFoundError("mark file corresponding to audio " + name + " is not found")
        start = list(map(read_time, mark_file.loc[:, "Start"]))
        duration = list(map(read_time, mark_file.loc[:, "Duration"]))

        start = (list(map(self.times_rate, start)))
        duration = (list(map(self.times_rate, duration)))
        # 乘以采样率并除以vector_size，因此可以表示采样后每个vector对应的标签

        tag = torch.zeros([self.chunk_size // self.vector_size])
        chunk_start = index * self.split_size // self.vector_size
        chunk_end = chunk_start + self.chunk_size // self.vector_size
        # 创建chunk的基础标签

        zipped = dict(zip(start, duration))
        for start_pt, duration_len in zipped.items():
            end_pt = start_pt + duration_len
            if start_pt > chunk_end or end_pt < chunk_start:  # no intersection
                pass
            elif start_pt >= chunk_start:
                if end_pt <= chunk_end:
                    tag[(start_pt - chunk_start): (end_pt - chunk_start)] = 1
                else:
                    tag[(start_pt - chunk_start):] = 1
            elif end_pt <= chunk_end:
                tag[: end_pt - chunk_start] = 1
            else:
                tag[:] = 1
        # 按照.csv中的开始和持续时长生成标签，处理重叠部分

        return wf, tag[None, :]

    def __init__(
            self,
            data_path: Union[Path, str], mark_path: Union[Path, str],
            sample_rate=8000,
            split_size=4 * 60 * 8000, front_overlap=2 * 60 * 8000, back_overlap=2 * 60 * 8000,
            data_suffix: Optional[str] = ".wav",
            sep="\t", vector_size=729
    ):
        data_path = Path(data_path)
        mark_path = Path(mark_path)
        super().__init__(sample_rate=sample_rate, split_size=split_size, front_overlap=front_overlap,
                         back_overlap=back_overlap, data_path=data_path, data_suffix=data_suffix)
        self.mark_path = mark_path
        self.sep = sep
        self.tags = []
        self.vector_size = vector_size
        for path in scanner(mark_path):
            if mark_path.suffix == ".csv":
                mark_file = pd.read_csv(mark_path, sep=sep, )
                start = list(map(read_time, mark_file.loc["Start"]))
                duration = list(map(read_time, mark_file.loc["Duration"]))
                full_duration = soundfile.info(str(self.data_path) + mark_path.name[:-4] + ".wav").duration
                # 去掉文件后缀（".csv"）

                start = list(map(self.times_rate, start))
                duration = list(map(self.times_rate, duration))
                full_duration *= self.sample_rate
                # 按照采样点生成标签

                tag = torch.zeros([full_duration])
                for start_pt, duration_len in start, duration:
                    tag[start_pt: start_pt + duration_len] = 1
                self.tags.append((path, tag))

    def times_rate(self, x):
        return int(x * self.sample_rate // self.vector_size)
