#!/usr/bin/env python
"""FishScope inference demo: zebrafish myotome + injury segmentation and birefringence read-outs.

Usage:
  python inference.py --image samples/12_1dpi_S.png --out out
  # add --gpu 0 to use a GPU
Writes out/<name>_overlay.png (myotome green, injury red) and appends a row to
out/results.csv with the myotome area, injury area, injured-area ratio, relative
birefringence of the wound, an approximate watershed myotome count, and an injury
uncertainty score (higher = flag the case for manual review).
"""
import os, csv, argparse, numpy as np, cv2, torch
from model import build_model
H, W = 1024, 1392
VIEWS = ["", "h", "v", "hv"]

def load_model(weights, device):
    m = build_model("brunetpp_mt", in_ch=1, deep_supervision=True).to(device)
    ck = torch.load(weights, map_location=device)
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    m.load_state_dict({k: v.float() if hasattr(v, "float") else v for k, v in sd.items()})
    m.eval(); return m

def tta_injury_uncertainty(m, x):
    ps = []
    for v in VIEWS:
        xi = x
        if "h" in v: xi = torch.flip(xi, dims=[3])
        if "v" in v: xi = torch.flip(xi, dims=[2])
        with torch.no_grad(): o = m(xi)
        p = torch.sigmoid(o["injury"])
        if "v" in v: p = torch.flip(p, dims=[2])
        if "h" in v: p = torch.flip(p, dims=[3])
        ps.append(p)
    P = torch.stack(ps); Pm = P.mean(0)[0, 0].cpu().numpy(); Ps = P.std(0)[0, 0].cpu().numpy()
    reg = (Pm > 0.1) & (Pm < 0.9)
    return float(Ps[reg].mean()) if reg.sum() > 0 else 0.0

def myotome_count(myo):
    try:
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        d = ndi.distance_transform_edt(myo)
        coords = peak_local_max(d, min_distance=18, labels=myo)
        mk = np.zeros(myo.shape, dtype=int)
        for i, (y, x) in enumerate(coords, 1): mk[y, x] = i
        return int(watershed(-d, mk, mask=myo).max())
    except Exception:
        return -1

def run(m, image_path, device):
    g0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(g0.astype(np.float32), (W, H))
    x = torch.from_numpy(gray / 255.).float()[None, None].to(device)
    with torch.no_grad(): o = m(x)
    myo = torch.sigmoid(o.get("refined", o["coarse"]))[0, 0].cpu().numpy() > 0.5
    inj = torch.sigmoid(o["injury"])[0, 0].cpu().numpy() > 0.5
    heal = myo & (~inj)
    ratio = float(inj.sum()) / max(int(myo.sum()), 1)
    biref = float(gray[inj].mean() / (gray[heal].mean() + 1e-6)) if inj.sum() > 0 and heal.sum() > 0 else 0.0
    unc = tta_injury_uncertainty(m, x)
    cnt = myotome_count(myo.astype(np.uint8))
    rgb = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rgb[myo] = (0.5 * rgb[myo] + 0.5 * np.array([90, 200, 90])).astype(np.uint8)
    rgb[inj] = (0.4 * rgb[inj] + 0.6 * np.array([60, 60, 235])).astype(np.uint8)
    return dict(myotome_area_px=int(myo.sum()), injury_area_px=int(inj.sum()),
                injured_area_ratio=round(ratio, 4), relative_birefringence=round(biref, 4),
                myotome_count=cnt, injury_uncertainty=round(unc, 4)), rgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="weights/fishscope_mt.pt")
    ap.add_argument("--out", default="out")
    ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args()
    device = "cuda:%d" % a.gpu if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    os.makedirs(a.out, exist_ok=True)
    m = load_model(a.weights, device)
    res, ov = run(m, a.image, device)
    name = os.path.splitext(os.path.basename(a.image))[0]
    cv2.imwrite(os.path.join(a.out, name + "_overlay.png"), ov)
    csvp = os.path.join(a.out, "results.csv"); new = not os.path.exists(csvp)
    with open(csvp, "a", newline="") as f:
        w = csv.DictWriter(f, ["image"] + list(res.keys()))
        if new: w.writeheader()
        w.writerow(dict(image=os.path.basename(a.image), **res))
    print(name, res)

if __name__ == "__main__":
    main()
