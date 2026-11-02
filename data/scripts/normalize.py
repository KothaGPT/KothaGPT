from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.split())


if __name__ == "__main__":
    source = Path("data/raw/sample.txt")
    if source.exists():
        print(normalize(source.read_text(encoding="utf-8")))
