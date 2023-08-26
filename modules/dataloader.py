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
        # 乘以采样率，因此可以按样本点个数表示

        tag = torch.zeros([self.chunk_size])
        chunk_start = index * self.chunk_size
        chunk_end = (index + 1) * self.chunk_size
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

        return wf, tag

    def __init__(
            self, sample_rate=8000,
            split_size=4 * 60 * 8000, front_overlap=2 * 60 * 8000, back_overlap=2 * 60 * 8000,
            data_path=Path("D:\\课程 2023暑假\\数据集\\"), data_suffix: Optional[str] = ".wav",
            mark_path=Path(r"D:\课程 2023暑假\数据集\阿梓 标注"), sep="\t"
    ):
        super().__init__(sample_rate, split_size, front_overlap, back_overlap, data_path, data_suffix)
        self.mark_path = mark_path
        self.sep = sep
        self.tags = []
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
        return int(x * self.sample_rate)


if __name__ == '__main__':
    audio = FineTuneAudio(data_path=Path(r"D:\课程 2023暑假\数据集\阿梓 标注\temp"))
    print(audio[0])
