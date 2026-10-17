# NAIP Transmission Tower Dataset — v1-corrected

This dataset contains annotated transmission towers in NAIP aerial imagery at
0.6 m ground sample distance. There is a single object class, `tower`, and it
is always class index 0.

Everything is in one zip, `naip_towers_v1.zip`. It's about 1.9 GB, mostly
because of the augmented training images, so give it a bit. The 72 images
removed from the test set (explained below) aren't in it, but I've kept them
if you ever want a look.

A good place to start is exploring the data and writing your evaluation
pipeline. Neither needs a GPU, and building the evaluation before you have any
results means we can compare implementations without either of us being
nudged by the numbers.

---

## Why 389 images and not 461

The paper says 461, and that number is out of date now, along with every
performance figure in it. Here's what happened.

An audit turned up 45 of the 50 New York training images sitting in the test
set as well, byte-for-byte identical. Two scripts had pulled from the same
NAIP source tiles without either one checking what the other had used: one
built the training set, one built what was supposed to be a fresh test
corridor. Tiling is deterministic, so slicing the same tile twice produced the
same patches under the same filenames. The upshot is that the models were
being graded partly on images they'd already trained on.

Four source tiles were involved, so all 72 test patches from those tiles came
out. The removal happened on the test side rather than the training side,
which is lucky: the trained models are still valid and nothing needs
retraining. `ny_leak_excluded.txt` lists what went.

A second bug turned up alongside it. The script that groups patches into
corridors for the bootstrap identifies a corridor by matching the filename
against a pattern, and any filename that didn't match was quietly treated as a
corridor of one image. Five source tiles use an older naming convention, so
about 92 of their patches each became their own corridor. The paper reports
122 corridors; there are really 26. That one moves the error bars rather than
the averages, but it still matters, since patches from the same stretch of
powerline aren't independent observations and the corridor grouping exists
precisely to handle that.

**So don't aim at the numbers in the paper.** I do have corrected ones, but
I've deliberately left them out of this README, since arriving at them
independently is the whole point. Send me yours and I'll send the comparison
straight back.

---

## Contents

```
test/images/            389 .jpg, 640x640
test/labels/            389 .txt, YOLO format
train/                  288 base training images and labels
val/                    67 validation images and labels, used for early stopping
augmented_train/        9,670 images and labels, generated from train/
augmentation_naip.py    the script that produced augmented_train/
augmentation.yaml       its config
tower_metadata.csv      per-instance metadata
ny_leak_excluded.txt    the 72 removed filenames
```

Train on `augmented_train/`, not `train/`. The 288 base images expand out to
9,670 through 36 rotations plus photometric and noise transforms, and that
expansion is what every model in the paper saw. I've included the script and
config so you can see what it's doing, and regenerate it yourself if you'd
rather; it's deterministic at seed 42.

One thing in there I'd flag. Rotation uses `BORDER_REFLECT_101`, so a tower
near the edge of a patch gets mirrored into the corner as a second copy with
no label on it, and the model is then being told that copy is background. I
honestly don't know how much this matters, but it's the sort of thing a fresh
pair of eyes might weigh differently than I do.

Each label file has one line per tower, with coordinates normalized 0-1
against the 640x640 patch:

```
0 0.512344 0.338281 0.047656 0.061719
```

Two things that are easy to trip over. Class index is always 0, but CVAT
single-class exports sometimes write 1, so worth checking if you ever
regenerate labels from a raw export. And an empty `.txt` means the patch
genuinely has no towers in it, so those are real negatives and they belong in
your evaluation.

### Splits are frozen

Please don't redraw them. If you resample, your results stop being comparable
to mine or to the paper's. I checked spatial independence at the NAIP tile
level across all three pairs (train/test, train/val, val/test) after the New
York removal, and they're clean now.

The dataset will grow; a national-scale version is under discussion. This
release stays pinned as `v1-corrected` either way, so whatever you produce
against it stays valid.

### Metadata

`tower_metadata.csv` has one row per annotated tower, so an image with three
towers shows up three times. Columns: `filename`, `state`, `split`, `lon`,
`lat`, `nlcd_code`, `nlcd_class`, `land_use`, `voltage_tier`, `dist_m`.

Two caveats. `nlcd_class` reads `"pending"` for 546 of the 839 test rows,
because the land cover spatial join never got finished for New York and West
Virginia. `land_use` is filled in regardless, so those categories still work,
but treat them as provisional. `voltage_tier` is either `345kV` or `500kV`,
and both show up in more than one state on purpose, so that voltage isn't
confounded with geography.

---

## Environment

Plan on Colab. The original work ran on an RTX 5060 Ti, a Blackwell card that
needed a PyTorch nightly build to work at all, but you can ignore all of that.
Colab's T4 or A100 runs fine on stable PyTorch.

```
torch                       # stable, whatever Colab ships
torchvision
ultralytics                 # YOLOv8m / v9m / v10m / v11m
transformers                # DETR
albumentations==1.4.18      # PINNED, only needed to regenerate augmentation
torchmetrics
opencv-python
pandas numpy pillow tqdm
```

The albumentations pin only matters if you regenerate `augmented_train/`
yourself. Versions after 1.4.18 changed the API in ways that alter the
augmentation behaviour without throwing an error, so you'd get different
training data with no signal that anything had changed. If you just use the
augmented images as shipped, you don't need albumentations at all.

Starting checkpoints, all public and COCO-pretrained:

| Model | Source |
|---|---|
| YOLOv8m / v9m / v10m / v11m | `ultralytics`, downloads automatically (v11 is `yolo11m.pt`) |
| RetinaNet | `torchvision`, `retinanet_resnet50_fpn_v2(weights="DEFAULT")` |
| Faster R-CNN | `torchvision`, `fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")` |
| DETR | `facebook/detr-resnet-50` on HuggingFace |

Rather than launching all seven at once, start with YOLOv8m on a free Colab
T4 and let me know how long it takes and roughly what it costs you in compute
units. It's the fastest of the seven, so that one run tells us whether the
whole set is realistic on free-tier hardware, and we can work out the rest
from there. Leave DETR until last whatever happens. It converges slowly and
won't finish inside Colab's 12-hour session limit without checkpointing
between sessions.

---

## Evaluation

Write your own evaluation code. If we build the pipelines separately and land
on similar numbers, that tells us something neither of us could learn alone.

The catch is that two perfectly reasonable AP implementations can differ by
several points on convention alone, which would leave us unable to tell a real
disagreement from a definitional one. So please match the decisions below.
Everything else is yours.

**Match these:**

Use an IoU threshold of 0.5, so a prediction counts only if it overlaps a real
tower box by at least half.

Match greedily, highest confidence first. Each prediction takes the ground
truth box it overlaps most, and a box that has already been claimed can't be
claimed again, so a second prediction on the same tower is a false positive.

Use all-point interpolation for the area under the precision-recall curve:
build the cumulative curve, force precision non-increasing from the right,
then integrate. This is not the 11-point VOC2007 version.

Compute AP per image, then average. This differs from pooling every detection
across the test set into one curve; the two weight images differently and can
disagree about which model is best. Ours does per-image.

Filter predictions at confidence 0.25 for the main evaluation. For
precision-recall curves use a much lower floor, around 0.001, or the curve
gets cut off partway.

One thing I'd rather leave open than dictate: when a patch has no towers and
the model correctly predicts nothing, there's no curve to integrate, so
somebody has to pick a value. Ours returns 1.0. I'm not confident that's
right, since it rewards whichever model is most conservative. Handle it
however you think makes sense and just tell me what you chose.

### What I'd like back

A CSV with one row for every image, for every model you run:

```
filename, model, ap50, precision, recall, n_gt, n_pred
```

Per-image rows rather than one summary line per model. If our numbers come
out different, that lets us find the specific images we disagree on, which is
usually where the explanation is.

---

Both of the bugs up top were found by re-reading code that had already been
reviewed and shipped, which is a decent argument for fresh eyes. So if
something in here doesn't make sense to you, please say so rather than
working around it.
