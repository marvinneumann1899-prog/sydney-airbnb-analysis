"""Download or verify the exact public input files recorded in data/sources.json."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check local files without network access")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/sources.json").read_text())
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        target = args.data_dir / item["file"]
        if target.exists():
            if checksum(target) != item["sha256"]:
                raise SystemExit(f"Checksum mismatch: {target.name}. Existing data was not overwritten.")
            print(f"Verified: {target.name}")
            continue
        if args.verify_only:
            raise SystemExit(f"Missing input: {target.name}")
        temporary = target.with_suffix(target.suffix + ".part")
        request = Request(item["url"], headers={"User-Agent": "SydneyAirbnbResearch/1.0"})
        try:
            with urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if checksum(temporary) != item["sha256"]:
                raise SystemExit(f"Downloaded content differs from the recorded snapshot: {target.name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified: {target.name}")


if __name__ == "__main__":
    main()
