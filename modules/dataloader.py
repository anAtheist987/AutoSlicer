from typing import Optional

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
        secs = int(time)
    elif split_len == 2:
        m, s = time.strip().split(':')
        secs = int(m) * 60 + int(s)
    elif split_len == 3:
        h, m, s = time.strip().split(':')
        secs = int(h) * 3600 + int(m) * 60 + int(s)

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


class FineTuneAudio(AudioDataset):
    def __getitem__(self, item):

        return super().__getitem__(item), self.tags[item]

    def __init__(self, audio_path=Path(r"D:\课程 2023暑假\数据集\歌回【阿梓从小就很可爱】"), mark_path=Path(r"D:\课程 2023暑假\数据集\阿梓 标注"),
                 sep="\t"):
        super().__init__()
        self.tags = []
        for path in scanner(mark_path):
            if mark_path.suffix == ".csv":
                mark_file = pd.read_csv(mark_path, sep=sep, )
                start = list(map(read_time, mark_file.loc["Start"]))
                duration = list(map(read_time, mark_file.loc["Duration"]))
                full_duration = soundfile.info(str(audio_path) + mark_path.name[:-4]).duration
                # 去掉文件后缀（假定为".wav"）

                start = list(map(self.times_rate, start))
                duration = list(map(self.times_rate, duration))
                full_duration *= self.sample_rate
                # 按照采样点生成标签

                tag = torch.zeros([full_duration])
                for start_pt, duration_len in start, duration:
                    tag[start_pt: start_pt + duration_len] = 1
                self.tags.append(tag)


    def times_rate(self, x):
        return x * self.sample_rate