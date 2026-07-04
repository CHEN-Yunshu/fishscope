# FishScope (anonymous demo)

Automated segmentation of zebrafish trunk myotomes and injury from birefringence
microscopy, with the muscle-regeneration read-outs. This anonymous release accompanies a
double-blind submission: it contains the inference code, one trained multi-task model,
and three sample images (a 1-vs-3 days-post-injury pair and one extra).

## Install
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

## Run
    python inference.py --image samples/12_1dpi_S.png --out out      # CPU
    python inference.py --image samples/12_1dpi_S.png --out out --gpu 0

For each image this writes `out/<name>_overlay.png` (myotome in green, injury in red) and
appends a row to `out/results.csv` with:
  * myotome_area_px, injury_area_px
  * injured_area_ratio         injury area / myotome area
  * relative_birefringence     mean(injured intensity) / mean(healthy-muscle intensity),
                               the established manual measure, computed automatically
  * myotome_count              approximate watershed count (auxiliary; see paper)
  * injury_uncertainty         test-time-augmentation disagreement of the injury head;
                               a higher value flags the case for manual review

## Files
    inference.py            standalone inference + read-outs
    model.py                network (U-Net++ + confidence-adaptive trimap + boundary
                            refinement module, multi-task myotome/injury head)
    weights/fishscope_mt.pt trained multi-task weights (one cross-validation fold)
    samples/                example birefringence images (1 and 3 dpi)
    example_output/         precomputed overlays and results.csv for the samples

## Notes
* Myotome instance separation / count is unsolved and reported only as an auxiliary,
  approximate read-out.
* The full dataset is available under the original data-use conditions upon publication.
