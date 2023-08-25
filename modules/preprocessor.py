import ffmpeg
from pathlib import Path


def scanner(path: Path):
    for i in path.iterdir():
        if i.is_dir():
            yield from scanner(i)
        elif i.is_file():
            yield i


def switch(src: Path, dst: Path = None, sample_rate=8000):
    if dst is None:
        dst = str(src.parent / (src.stem + "_new" + src.suffix))
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True)
    stream = (
        ffmpeg
        .input(src.__str__())
        .output(str(dst), **{'ac': 1, 'ar': sample_rate, })
    )
    stream.run()


if __name__ == '__main__':
    data_path = Path(r"E:\DataSet\audio\eureka\pv\cut")
    dst_dir = Path(r"E:\DataSet\audio\eureka\pv\cut\8k")
    for audio_file in scanner(data_path):
        relative = audio_file.relative_to(data_path)
        switch(audio_file, dst=dst_dir / relative.parent / (relative.stem + ".wav"))
