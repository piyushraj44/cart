"""Download the two permitted datasets into data/raw/ (they are not committed to git).

    python -m ml.download_data            # text dataset + ContextRico ground truth (~1 MB)
    python -m ml.download_data --images   # also the ContextRico screenshots (~226 MB zip)

Sources (original, unmodified):
  * yamanalab/ec-darkpattern  (Apache-2.0)   https://github.com/yamanalab/ec-darkpattern
  * ContextRico-DP, Zenodo 15036779 (CC-BY-4.0) https://zenodo.org/records/15036779
Checksums are verified against the MD5 values published by Zenodo.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
import zipfile

from ml.common import DATA_RAW

ZENODO = "https://zenodo.org/api/records/15036779/files/{name}/content"
CR_FILES = {
    "groundtrue.json": "73ca3644c308864eea9dc63ca87e200f",
    "DPfeatures.xlsx": "172dd2c2624a5e7f6e1dd5e318830bda",
    "correction.xlsx": "c0a4cf7a3caab3f92b6cfdbbd56ddf3a",
}
CR_ZIP = ("ContextRico-DP.zip", "b86ab8780d325c320d078bad90603757")


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, expected: str) -> None:
    dest = DATA_RAW / "contextrico" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and md5(dest) == expected:
        print(f"ok (cached)  {dest.relative_to(DATA_RAW.parent.parent)}")
        return
    print(f"downloading  {name} ...")
    urllib.request.urlretrieve(ZENODO.format(name=name), dest)
    got = md5(dest)
    if got != expected:
        sys.exit(f"MD5 mismatch for {name}: {got} != {expected}")
    print(f"ok           {dest.relative_to(DATA_RAW.parent.parent)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", action="store_true", help="also download and extract the ContextRico screenshots")
    args = ap.parse_args()

    repo = DATA_RAW / "ec-darkpattern"
    if not (repo / "dataset" / "dataset.tsv").exists():
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/yamanalab/ec-darkpattern.git", str(repo)], check=True)
    print("ok           data/raw/ec-darkpattern/dataset/dataset.tsv")

    for name, h in CR_FILES.items():
        fetch(name, h)
    if args.images:
        fetch(*CR_ZIP)
        out = DATA_RAW / "contextrico" / "images"
        if not (out / "ContextRico-DP").exists():
            with zipfile.ZipFile(DATA_RAW / "contextrico" / CR_ZIP[0]) as z:
                z.extractall(out)
        print("ok           data/raw/contextrico/images/ContextRico-DP/")


if __name__ == "__main__":
    main()
