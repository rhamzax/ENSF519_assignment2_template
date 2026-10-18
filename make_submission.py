"""Build the Assignment 2 submission archive with Python's standard library."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ENSF519_assignment2_submission.zip"
ARCHIVE_ROOT = "ENSF519_assignment2"

TOP_LEVEL_FILES = (
    ".python-version",
    "00_dataset_and_tensors.ipynb",
    "01_cnn_model.ipynb",
    "02_training_and_testing.ipynb",
    "README.md",
    "report.md",
    "pyproject.toml",
    "uv.lock",
    "make_submission.sh",
    "make_submission.py",
)


def submission_files():
    """Return the files that should be included in a submission."""
    files = [ROOT / name for name in TOP_LEVEL_FILES]
    source_files = sorted((ROOT / "assignment2").rglob("*.py"))
    asset_files = sorted(
        path for path in (ROOT / "assets").rglob("*")
        if path.is_file()
        and all(
            not part.startswith(".")
            for part in path.relative_to(ROOT / "assets").parts
        )
    )

    if not source_files or not asset_files:
        raise FileNotFoundError("Expected Python source files and assets are missing")
    files.extend(source_files)
    files.extend(asset_files)

    for path in files:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing or linked submission file: {path}")

    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def main():
    files = submission_files()
    with NamedTemporaryFile(dir=ROOT, suffix=".zip", delete=False) as temp:
        temporary_path = Path(temp.name)

    try:
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED) as archive:
            for path in files:
                name = f"{ARCHIVE_ROOT}/{path.relative_to(ROOT).as_posix()}"
                archive.write(path, arcname=name)
        temporary_path.replace(OUTPUT)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Created {OUTPUT.name} ({len(files)} files)")


if __name__ == "__main__":
    main()
