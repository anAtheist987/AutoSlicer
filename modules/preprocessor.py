import ffmpeg
from pathlib import Path

data_path = Path(r"D:\课程 2023暑假\数据集")


def scanner(path: Path):
    for i in path.iterdir():
        if i.is_dir():
            yield from scanner(i)
        elif i.is_file() and i.suffix == ".m4s":
            yield i


def switch(src: Path):
    stream = (
        ffmpeg
        .input(src.__str__())
        .output(str(src.parent / src.stem) + ".wav", **{'ac': 1, 'ar': 8000, })
    )
    stream.run()


if __name__ == '__main__':
    for path in scanner(data_path):
        switch(path)


