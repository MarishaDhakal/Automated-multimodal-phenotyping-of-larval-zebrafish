# SAM model checkpoints (not included in this repository)

The pipeline uses the **Segment Anything Model (SAM)** for well/arena and object
segmentation. The checkpoint files are large (hundreds of MB to several GB) and are
**not** distributed with this repository.

## Required checkpoint

Production uses the **ViT-H** checkpoint:

| File | Model type | Size | Download |
|------|-----------|------|----------|
| `sam_vit_h_4b8939.pth` | `vit_h` | ~2.4 GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |

Optional smaller checkpoint (not used by the production pipeline):

| File | Model type | Size | Download |
|------|-----------|------|----------|
| `sam_vit_b_01ec64.pth` | `vit_b` | ~375 MB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |

## Download

```bash
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Pointing the pipeline at the checkpoint

All three extractors default to `checkpoints/sam_vit_h_4b8939.pth` in this
checkout. If the file is stored elsewhere, export `SAM_CHECKPOINT` with its
absolute path before launching the pipeline. For standalone heartbeat and
L-bend runs, `--checkpoint` overrides that setting.
