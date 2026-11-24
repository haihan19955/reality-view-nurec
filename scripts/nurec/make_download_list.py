"""Pick N NuRec scenes (balanced across release batches) and write download links for the ones not yet downloaded.

    python scripts/nurec/make_download_list.py 16      # -> $RV_WORK/nurec_links.tsv  (url, file name, bytes)

Requires accepting the dataset terms on Hugging Face (nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) and `huggingface-cli login`.
Download the files into $NUREC_SCENES keeping the repo-relative paths.
"""
import os
import random
import sys

from huggingface_hub import HfApi

from realityview import paths

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
REPO, REV = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec", "25.05"
files = list(HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REV, path_in_repo="sample_set/25.07_release", recursive=True))
usdz = [f for f in files if f.path.endswith(".usdz") and getattr(f, "size", 0) and f.size < 1.8e9]
by_batch = {}
for f in usdz:
    by_batch.setdefault(f.path.split("/")[2], []).append(f)
random.seed(0)
for b in by_batch.values():
    random.shuffle(b)
pick = []
while len(pick) < N and any(by_batch.values()):
    for b in sorted(by_batch):
        if by_batch[b] and len(pick) < N:
            pick.append(by_batch[b].pop())


def downloaded(f):
    p = os.path.join(paths.NUREC_SCENES, f.path)
    return os.path.exists(p) and os.path.getsize(p) == f.size


todo = [f for f in pick if not downloaded(f)]
print("done", len(pick) - len(todo), "todo", len(todo), "%.1f GB" % (sum(f.size for f in todo) / 1e9))
os.makedirs(paths.WORK, exist_ok=True)
out = os.path.join(paths.WORK, "nurec_links.tsv")
with open(out, "w") as fo:
    for f in todo:
        fo.write(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{f.path}?download=true\t{f.path}\t{f.size}\n")
print("wrote", out)
