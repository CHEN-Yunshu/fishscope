# FishScope

Automated analysis of zebrafish myotomes and injury in birefringence microscopy.
This repository provides an inference demo for myotome and injury segmentation,
relative wound birefringence, injured-area ratio, and approximate watershed counting.

## Related paper

**FishScope: Automated Segmentation of Birefringence Imaging Data for Quantifying
Zebrafish Muscle Regeneration**

Yunshu Chen, Litao Yang, Marco Podobnik, Safoura Zardadi, Margo Montandon,
Guanliang Chen, Peter D. Currie, Zongyuan Ge, and Jackie Rong.
IEEE BIBM 2026.

Citation metadata and the full author list are provided in [CITATION.cff](CITATION.cff).
The paper is accepted; a DOI and proceedings page numbers are not yet included.

## License

The original code, documentation, bundled weights, and sample images are available
under the custom [FishScope Non-Commercial Research License](LICENSE).
Use, modification, and redistribution are permitted only for non-commercial
scientific research, subject to the license conditions. Commercial use, including
commercial product development or paid analysis services, requires separate
written permission from the relevant copyright holders.

This is a source-available research license, not an open-source license.
Third-party dependencies retain their own licenses. Refer to [LICENSE](LICENSE)
for the complete terms.

## Release scope

The demo includes a U-Net++ backbone with confidence-adaptive trimaps and boundary
refinement, parallel myotome and injury heads, one trained checkpoint from one
cross-validation fold, and three example images with saved outputs.

The bundled counting routine uses distance-transform watershed on the semantic
myotome mask. The paper's fine-tuned Cellpose-SAM instance model, its checkpoints,
and the full training and evaluation pipeline are not included in this demo.
The example outputs are demonstrations, not the paper's five-fold benchmark results.

## Quick start

Use Python 3.11. Run these commands in a terminal (macOS or Linux):

```bash
git clone https://github.com/CHEN-Yunshu/fishscope.git
cd fishscope
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python inference.py --image samples/12_1dpi_S.png --out out
```

On Windows Command Prompt, create the environment with `py -3.11 -m venv .venv`
and activate it with `.venv\Scripts\activate`. In PowerShell, the activation
command is `.\.venv\Scripts\Activate.ps1`.
The default device is CPU. To request CUDA GPU 0 with a CUDA-enabled PyTorch installation:

```bash
python inference.py --image samples/12_1dpi_S.png --out out --gpu 0
```

Run the paired 3-dpi image into the same output directory:

```bash
python inference.py --image samples/12_3dpi_S.png --out out
```

Use `python inference.py --help` for the checkpoint and output-directory options.

The CPU example was checked with Python 3.11.5, PyTorch 2.0.1, NumPy 1.24.3,
OpenCV 4.8.1.78, SciPy 1.9.3, and scikit-image 0.20.0. These are the tested
versions, not an exhaustive compatibility list.

## Inputs and outputs

Images are read as 8-bit grayscale and resized to 1392 pixels wide by 1024 pixels
high. Reported areas refer to this resized grid. Each run writes
`out/<image_name>_overlay.png` (green: myotomes; red: injury) and appends one row to
`out/results.csv`. Re-running an image appends another row.

| Output column | Meaning |
| --- | --- |
| `image` | Input filename. |
| `myotome_area_px` | Number of predicted myotome pixels. |
| `injury_area_px` | Number of predicted injury pixels. |
| `injured_area_ratio` | Predicted injury area divided by predicted myotome area. |
| `relative_birefringence` | Mean intensity in the predicted injury divided by mean intensity in predicted myotome pixels outside the injury. |
| `myotome_count` | Approximate watershed count, using distance-map peaks at least 18 pixels apart. |
| `injury_uncertainty` | Prediction variation under horizontal and vertical flips, averaged over uncertain injury pixels; larger values indicate greater disagreement. |

Relative birefringence is a mask-based adaptation of the manual assay: the healthy
reference uses all predicted uninjured myotome pixels, and no background
subtraction is applied. It is not an implementation of the complete manual Fiji
measurement protocol.

If either the injury region or the healthy reference is empty, the current script
writes `0` for `relative_birefringence`; this represents an invalid ratio and should
be excluded from ratio analyses. A `myotome_count` of `-1` indicates that watershed
counting failed. Counts can be affected by merged or fragmented regions and are
not fixed anatomical segment totals.

## Repository contents

```text
inference.py             Inference, image overlays, and quantitative read-outs
model.py                 Network definitions
requirements.txt         Dependencies for this inference demo
weights/fishscope_mt.pt   Trained multi-task checkpoint (one fold)
samples/                 Three example birefringence images
example_output/          Saved overlays and results.csv for those images
```

The samples include `12_1dpi_S.png` and `12_3dpi_S.png`, a paired time series from
one fish, plus `66_1dpi_T.png`. The full imaging dataset and expert annotations
are not distributed in this repository.

For questions or reproducible bug reports, use the
[issue tracker](https://github.com/CHEN-Yunshu/fishscope/issues).
