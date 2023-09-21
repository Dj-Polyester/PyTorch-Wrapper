from pathlib import Path
from . import Debug


def fromRaw(rawPath: str | Path):
    if isinstance(rawPath, str):
        return Path(rawPath)
    elif isinstance(rawPath, Path):
        return rawPath
    raise Debug.TypeError(filePath=rawPath)


def mkdirIfNotExists(path: Path):
    dirname = path.parents[0]
    if not dirname.exists():
        dirname.mkdir(parents=True)
