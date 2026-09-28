"""Package only compression source, excluding Git history and model weights."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
FILES = (
    ".gitignore",
    "LICENSE",
    "README_COMPRESSION.md",
    "requirements-compress.txt",
    "compress_script.sh",
    "package_compression.py",
    "autoregressive/sample/compress.py",
    "autoregressive/models/gpt.py",
    "autoregressive/models/generate.py",
    "tokenizer/tokenizer_image/vq_model.py",
    "utils/drop_path.py",
)

def main():
    destination = ROOT.parent / "LlamaGen_compression_code.zip"
    for name in FILES:
        source = ROOT / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Expected a regular source file: {source}")
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, f"LlamaGen/{name}")
    print(f"Created {destination} ({destination.stat().st_size:,} bytes, {len(FILES)} files)")

if __name__ == "__main__":
    main()
