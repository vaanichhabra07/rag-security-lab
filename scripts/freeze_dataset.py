
import json
import hashlib
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
METADATA = DATA / "metadata"
BACKUPS = ROOT / "backups"

VERSION = "1.0-rc1"

BACKUPS.mkdir(parents=True, exist_ok=True)


def sha256_file(path):

    sha = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(65536), b""
        ):
            sha.update(block)

    return sha.hexdigest()


def main():

    report = json.loads(
        (METADATA / "validation_report.json")
        .read_text(encoding="utf-8")
    )

    assert report["structural_validation_passed"], (
        "Dataset validation must pass before freezing."
    )

    files = [
        PROCESSED / "master_dataset.jsonl",
        PROCESSED / "master_dataset.csv",
        SPLITS / "train.jsonl",
        SPLITS / "validation.jsonl",
        SPLITS / "test_seen.jsonl",
        SPLITS / "test_unseen.jsonl",
        SPLITS / "test_benign.jsonl",
        METADATA / "dataset_card.md",
        METADATA / "dataset_statistics.json",
        METADATA / "split_statistics.json",
        METADATA / "validation_report.json"
    ]

    # Include corpus and generation scripts.

    for folder in [
        DATA / "corpus",
        ROOT / "scripts"
    ]:

        files.extend(
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix in (
                ".txt", ".json", ".jsonl",
                ".csv", ".md", ".py"
            )
        )

    # Include reproducibility metadata.

    files.extend(
        p for p in METADATA.iterdir()
        if p.is_file()
        and p.name not in (
            "checksums.json",
            "dataset_version.json"
        )
    )

    # Include intermediate processed records.

    files.extend(
        p for p in PROCESSED.glob("*.jsonl")
        if p.is_file()
    )

    files = sorted(set(files))

    checksums = {}

    for path in files:

        assert path.exists(), (
            f"Missing file: {path}"
        )

        relative = path.relative_to(ROOT).as_posix()

        checksums[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size
        }

    checksum_path = METADATA / "checksums.json"

    checksum_path.write_text(
        json.dumps(checksums, indent=4),
        encoding="utf-8"
    )

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True
    ).strip()

    manifest = {
        "dataset_name": "RAG Security Lab",
        "version": VERSION,
        "frozen_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "research_candidate",
        "total_samples": 1500,
        "benign_samples": 250,
        "adversarial_samples": 1250,
        "institutional_documents": 50,
        "qa_pairs": 250,
        "attack_families": 5,
        "splits": {
            "train": 600,
            "validation": 300,
            "test_seen": 250,
            "test_unseen": 250,
            "test_benign": 100
        },
        "random_seed": 42,
        "structural_validation_passed": True,
        "research_validation_complete": False,
        "validation_warnings": report.get(
            "warnings", []
        ),
        "git_commit_before_freeze": git_commit,
        "files_checksummed": len(checksums),
        "checksum_algorithm": "SHA-256"
    }

    manifest_path = METADATA / "dataset_version.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8"
    )

    backup_path = (
        BACKUPS / f"rag_security_dataset_{VERSION}.zip"
    )

    if backup_path.exists():
        raise FileExistsError(
            "Backup already exists. Preserve the existing "
            "snapshot or use a new dataset version."
        )

    archive_files = files + [
        checksum_path,
        manifest_path
    ]

    with zipfile.ZipFile(
        backup_path,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        for path in archive_files:

            archive.write(
                path,
                path.relative_to(ROOT).as_posix()
            )

    # Verify archived files against checksums.

    with zipfile.ZipFile(backup_path, "r") as archive:

        assert archive.testzip() is None

        for filename, info in checksums.items():

            actual = hashlib.sha256(
                archive.read(filename)
            ).hexdigest()

            assert actual == info["sha256"], (
                f"Checksum mismatch: {filename}"
            )

    print("=" * 55)
    print("DATASET FREEZE COMPLETE")
    print("=" * 55)

    print(f"Version: {VERSION}")
    print("Total samples: 1500")
    print(f"Files checksummed: {len(checksums)}")
    print("Archive integrity: PASSED")
    print("Checksum verification: PASSED")

    print(f"\nChecksums: {checksum_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Backup: {backup_path}")

    print(
        f"Backup size: "
        f"{backup_path.stat().st_size / 1024**2:.2f} MB"
    )

    print("\nDATASET SNAPSHOT SAVED SUCCESSFULLY")


if __name__ == "__main__":
    main()
