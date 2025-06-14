
import json
from collections import defaultdict
from pathlib import Path

# --- paths ---------------------------------------------------
root = Path("/work/tttpd9bjo/MSCOCO")                    # adjust if COCO lives elsewhere
ann_file = root / "annotations/captions_val2017.json"
out_file = root / "val2017_5000_captions.txt"   # plain-text output

# --- load JSON ----------------------------------------------
with ann_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

# --- build {image_id: [captions]} ---------------------------
captions_by_img = defaultdict(list)
for ann in data["annotations"]:
    captions_by_img[ann["image_id"]].append(ann["caption"].strip())

# --- keep one caption per image (remove next 3 lines if you want all 5) ----
for img_id, caps in captions_by_img.items():
    captions_by_img[img_id] = caps[:1]          # first caption only

# --- flatten and save ---------------------------------------
with out_file.open("w", encoding="utf-8") as f:
    for img_id in sorted(captions_by_img):          # sort keys first
        for cap in captions_by_img[img_id]:
            f.write(cap + "\n")

print(f"Saved {sum(len(v) for v in captions_by_img.values())} captions to {out_file}")
