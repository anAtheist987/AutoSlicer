from typing import Iterator, Optional

import torch

import librosa
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


'''
写下来以免明天忘了：
啥比更新……load发黄是因为更新后torchaudio会在运行过程中才将backend里的load，info等函数加进来
因此目前得用torchaudio.backend.load()才不会发黄
附上代码：

def _init_audio_backend():
    backends = list_audio_backends()
    if "sox_io" in backends:
        set_audio_backend("sox_io")
    elif "soundfile" in backends:
        set_audio_backend("soundfile")
    else:
        warnings.warn("No audio backend is available.")
        set_audio_backend(None)
        
        
if _is_backend_dispatcher_enabled():
    from torchaudio._backend.utils import get_info_func, get_load_func, get_save_func

    torchaudio.info = get_info_func()
    torchaudio.load = get_load_func()
    torchaudio.save = get_save_func()
else:
    utils._init_audio_backend()
    

'''


class AudioDataset(Dataset):
    def __getitem__(self, index):
        path = None
        for path, num in self.split_list:
            if index >= num:
                index -= num
            else:
                path = str(path)
                break
        chunk = librosa.load(path, sr=self.sample_rate, mono=True,
                             offset=index * (self.split_size / self.sample_rate),
                             duration=self.chunk_size / self.sample_rate)
        wf = torch.tensor(chunk[0])
        return wf

    def __len__(self):
        return self.index_num

    def __init__(
            self, sample_rate=8000,
            split_size=4 * 60 * 8000, front_overlap=2 * 60 * 8000, back_overlap=2 * 60 * 8000,
            data_path=Path("D:\\课程 2023暑假\\数据集\\"), data_suffix: Optional[str] = ".wav",
    ):
        self.split_size = split_size
        self.chunk_size = split_size + front_overlap + back_overlap
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.index_num = 0
        self.split_list = []

        for path in scanner(data_path):
            if data_suffix is None or data_path.suffix == data_suffix:
                info = soundfile.info(str(path))
                if info.samplerate != self.sample_rate:
                    print(f"Skipped file not satisfying sample rate: {path}")
                    continue
                split_num = (info.frames -
                             (front_overlap + back_overlap)) // split_size
                split_num = int(split_num)
                self.index_num += split_num
                self.split_list.append((path, split_num))
