# Master's Thesis: Fish Detection and Counting with SAM3 and YOLO

At the current stage, fish in aquarium videos are segmented and counted. SAM3 generates masks that are used to train a YOLO segmentation model. Both models are evaluated against manually annotated ground truth.

**Current pipeline:** videos → frames → SAM3 masks → YOLO dataset → YOLO training → evaluation

## Structure

```
Masterarbeit/
├── recordings_counted/     # videos and extracted frames                 [not in git]
├── gt_labels_new_tofill/   # ground truth (Label Studio)                 [not in git]
├── sam3/                   # SAM3: mask generation, YOLO dataset, evaluation
├── yolo/                   # YOLO: training, evaluation
└── Paper/                  # literature
```

## Before running

- **Data is not in git.** `recordings_counted/` and `gt_labels_new_tofill/` must exist locally in the repository root, as well as the model and output folders (see `.gitignore`).
- **Paths:** Every notebook and script has a configuration block at the top (`*_DIR`, `MODEL_PATH`, `CONFIG`). Adjust paths there if needed.
