"""BRU-Net++ : U-Net++ backbone + dynamic trimap + Boundary Refinement Module (BRM).

Faithful re-implementation from the paper
"Boundary-Refined U-Net++ for Fast Segmentation of Zebrafish Myotomes".

Models exposed via build_model(name):
  - "unetpp"      : plain U-Net++ baseline (deep supervision), returns coarse logits.
  - "unetpp_tri"  : U-Net++ + dynamic-trimap as extra input channel (no BRM).
  - "brunetpp"    : full BRU-Net++ (trimap + BRM).
The forward returns a dict so train.py can access coarse map P, trimap T and refined Pr.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetPlusPlus(nn.Module):
    """Nested U-Net (U-Net++) with dense skip pathways and deep supervision.

    forward -> dict(feat=F (X0,4 features), coarse=logits, ds=[deep-sup logits])
    """
    def __init__(self, in_ch=1, nb=(32, 64, 128, 256, 512), deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # column 0 (encoder)
        self.x00 = VGGBlock(in_ch, nb[0], nb[0])
        self.x10 = VGGBlock(nb[0], nb[1], nb[1])
        self.x20 = VGGBlock(nb[1], nb[2], nb[2])
        self.x30 = VGGBlock(nb[2], nb[3], nb[3])
        self.x40 = VGGBlock(nb[3], nb[4], nb[4])

        # nested decoder nodes  X[i][j]  (in_ch = nb[i]*j + nb[i+1])
        self.x01 = VGGBlock(nb[0] * 1 + nb[1], nb[0], nb[0])
        self.x11 = VGGBlock(nb[1] * 1 + nb[2], nb[1], nb[1])
        self.x21 = VGGBlock(nb[2] * 1 + nb[3], nb[2], nb[2])
        self.x31 = VGGBlock(nb[3] * 1 + nb[4], nb[3], nb[3])

        self.x02 = VGGBlock(nb[0] * 2 + nb[1], nb[0], nb[0])
        self.x12 = VGGBlock(nb[1] * 2 + nb[2], nb[1], nb[1])
        self.x22 = VGGBlock(nb[2] * 2 + nb[3], nb[2], nb[2])

        self.x03 = VGGBlock(nb[0] * 3 + nb[1], nb[0], nb[0])
        self.x13 = VGGBlock(nb[1] * 3 + nb[2], nb[1], nb[1])

        self.x04 = VGGBlock(nb[0] * 4 + nb[1], nb[0], nb[0])

        if deep_supervision:
            self.ds1 = nn.Conv2d(nb[0], 1, 1)
            self.ds2 = nn.Conv2d(nb[0], 1, 1)
            self.ds3 = nn.Conv2d(nb[0], 1, 1)
        self.final = nn.Conv2d(nb[0], 1, 1)

    def forward(self, x):
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        x01 = self.x01(torch.cat([x00, self.up(x10)], 1))
        x11 = self.x11(torch.cat([x10, self.up(x20)], 1))
        x21 = self.x21(torch.cat([x20, self.up(x30)], 1))
        x31 = self.x31(torch.cat([x30, self.up(x40)], 1))

        x02 = self.x02(torch.cat([x00, x01, self.up(x11)], 1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21)], 1))
        x22 = self.x22(torch.cat([x20, x21, self.up(x31)], 1))

        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12)], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.up(x22)], 1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1))

        out = {"feat": x04, "coarse": self.final(x04)}
        if self.deep_supervision:
            out["ds"] = [self.ds1(x01), self.ds2(x02), self.ds3(x03)]
        return out


def make_trimap(coarse_logits, erode_k=3, dilate_k=5, thresh=0.5):
    """Dynamic trimap from coarse probability via morphology (differentiable-free, detached).

    Returns soft trimap in {0, 0.5, 1}: 1 = sure foreground (erosion),
    0 = sure background (outside dilation), 0.5 = uncertain band.
    Erosion/dilation implemented with min/max pooling on the binarized map.
    """
    with torch.no_grad():
        p = (torch.sigmoid(coarse_logits) > thresh).float()
        pe = -F.max_pool2d(-p, erode_k, 1, erode_k // 2)          # erosion
        pd = F.max_pool2d(p, dilate_k, 1, dilate_k // 2)          # dilation
        trimap = torch.full_like(p, 0.5)
        trimap[pd == 0] = 0.0
        trimap[pe == 1] = 1.0
    return trimap


def make_trimap_cat(coarse_logits, band_frac=0.10):
    """Confidence-Adaptive Trimap (CAT) -- our novelty.

    Instead of a fixed-kernel morphological band tuned to myoseptal width, the
    uncertain band is derived from the model's own prediction confidence and the
    threshold is set adaptively *per image* so the band is always the least-confident
    `band_frac` of pixels. This removes all hand-tuned, structure-specific kernel
    sizes -> the band adapts to any boundary scale / morphology (generalization).
    Returns soft trimap in {0, 0.5, 1}.
    """
    with torch.no_grad():
        p = torch.sigmoid(coarse_logits)
        conf = (p - 0.5).abs()                       # 0 = most uncertain
        trimap = torch.zeros_like(p)
        for b in range(p.shape[0]):
            thr = torch.quantile(conf[b].flatten(), band_frac)
            uncertain = conf[b] <= thr
            trimap[b] = torch.where(p[b] > 0.5, torch.ones_like(p[b]), torch.zeros_like(p[b]))
            trimap[b][uncertain] = 0.5
    return trimap


class BRM(nn.Module):
    """Boundary Refinement Module: gated residual refinement conditioned on trimap."""
    def __init__(self, ch):
        super().__init__()
        self.conv_in = nn.Conv2d(ch, ch, 3, padding=1)
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, feat, band):
        # band: gating mask (hard {0,1} for morph/cat, soft [0,1] for learnable)
        fc = self.conv_in(feat)
        f_gate = fc * band                       # Eq.2 (band-localized gating)
        f_ref = feat + f_gate                    # Eq.3 residual
        f_ref = self.refine(f_ref)
        return self.out(f_ref)


class BRUNetPlusPlus(nn.Module):
    def __init__(self, in_ch=1, nb=(32, 64, 128, 256, 512), use_brm=True,
                 use_trimap_input=True, erode_k=3, dilate_k=5, deep_supervision=True,
                 trimap_mode="morph", band_frac=0.10, multitask=False, instseg=False):
        super().__init__()
        self.use_brm = use_brm
        self.use_trimap_input = use_trimap_input
        self.erode_k, self.dilate_k = erode_k, dilate_k
        self.trimap_mode = trimap_mode      # "morph" (paper) | "cat" (adaptive, ours)
        self.band_frac = band_frac
        self.multitask = multitask          # add a parallel injury-segmentation head
        extra = 1 if use_trimap_input else 0
        self.backbone = UNetPlusPlus(in_ch + extra, nb, deep_supervision)
        # a cheap first-pass to produce coarse map for the trimap (shares weights via 2-pass)
        self.coarse_head = nn.Conv2d(in_ch, 1, 1) if use_trimap_input else None
        if use_brm:
            self.brm = BRM(nb[0])
        if trimap_mode == "learn":   # learnable band: a small head predicts the boundary band
            self.band_head = nn.Sequential(
                nn.Conv2d(nb[0], nb[0] // 2, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(nb[0] // 2, 1, 1))
        if multitask:
            self.injury_head = nn.Sequential(
                nn.Conv2d(nb[0], nb[0], 3, padding=1), nn.BatchNorm2d(nb[0]), nn.ReLU(inplace=True),
                nn.Conv2d(nb[0], 1, 1))
        self.instseg = instseg  # myoseptal-boundary head for instance separation
        if instseg:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(nb[0], nb[0], 3, padding=1), nn.BatchNorm2d(nb[0]), nn.ReLU(inplace=True),
                nn.Conv2d(nb[0], 1, 1))

    def _trimap(self, logits):
        if self.trimap_mode == "cat":
            return make_trimap_cat(logits, self.band_frac)
        return make_trimap(logits, self.erode_k, self.dilate_k)

    def forward(self, x):
        if self.use_trimap_input:
            # pass-1: a light coarse prediction directly from input to seed the trimap
            coarse0 = self.coarse_head(x)
            tri0 = self._trimap(coarse0)
            out = self.backbone(torch.cat([x, tri0], 1))
        else:
            out = self.backbone(x)

        coarse = out["coarse"]
        if self.trimap_mode == "learn":
            band_logit = self.band_head(out["feat"])
            out["band_logit"] = band_logit
            gate = torch.sigmoid(band_logit)          # soft band for gating
            band_mask = (gate > 0.5).float()
            out["trimap"] = band_mask * 0.5 + (torch.sigmoid(coarse) > 0.5).float() * (1 - band_mask)
        else:
            trimap = self._trimap(coarse)
            out["trimap"] = trimap
            band_mask = ((trimap > 0) & (trimap < 1)).float()
            gate = band_mask
        out["band"] = band_mask
        if self.use_brm:
            out["refined"] = self.brm(out["feat"], gate)
        else:
            out["refined"] = coarse
        if self.multitask:
            out["injury"] = self.injury_head(out["feat"])
        if self.instseg:
            out["boundary"] = self.boundary_head(out["feat"])
        return out


class UNet(nn.Module):
    """Standard U-Net baseline (returns dict for a uniform train interface)."""
    def __init__(self, in_ch=1, nb=(32, 64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.e0 = VGGBlock(in_ch, nb[0], nb[0])
        self.e1 = VGGBlock(nb[0], nb[1], nb[1])
        self.e2 = VGGBlock(nb[1], nb[2], nb[2])
        self.e3 = VGGBlock(nb[2], nb[3], nb[3])
        self.b = VGGBlock(nb[3], nb[4], nb[4])
        self.d3 = VGGBlock(nb[4] + nb[3], nb[3], nb[3])
        self.d2 = VGGBlock(nb[3] + nb[2], nb[2], nb[2])
        self.d1 = VGGBlock(nb[2] + nb[1], nb[1], nb[1])
        self.d0 = VGGBlock(nb[1] + nb[0], nb[0], nb[0])
        self.final = nn.Conv2d(nb[0], 1, 1)

    def forward(self, x):
        e0 = self.e0(x); e1 = self.e1(self.pool(e0)); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.up(b), e3], 1))
        d2 = self.d2(torch.cat([self.up(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up(d2), e1], 1))
        d0 = self.d0(torch.cat([self.up(d1), e0], 1))
        return {"coarse": self.final(d0)}


class AttnGate(nn.Module):
    def __init__(self, fg, fl, fint):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(fg, fint, 1), nn.BatchNorm2d(fint))
        self.wx = nn.Sequential(nn.Conv2d(fl, fint, 1), nn.BatchNorm2d(fint))
        self.psi = nn.Sequential(nn.Conv2d(fint, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        a = self.psi(self.relu(self.wg(g) + self.wx(x)))
        return x * a


class AttUNet(nn.Module):
    """Attention U-Net baseline (returns dict)."""
    def __init__(self, in_ch=1, nb=(32, 64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.e0 = VGGBlock(in_ch, nb[0], nb[0]); self.e1 = VGGBlock(nb[0], nb[1], nb[1])
        self.e2 = VGGBlock(nb[1], nb[2], nb[2]); self.e3 = VGGBlock(nb[2], nb[3], nb[3])
        self.b = VGGBlock(nb[3], nb[4], nb[4])
        self.a3 = AttnGate(nb[4], nb[3], nb[3]); self.d3 = VGGBlock(nb[4] + nb[3], nb[3], nb[3])
        self.a2 = AttnGate(nb[3], nb[2], nb[2]); self.d2 = VGGBlock(nb[3] + nb[2], nb[2], nb[2])
        self.a1 = AttnGate(nb[2], nb[1], nb[1]); self.d1 = VGGBlock(nb[2] + nb[1], nb[1], nb[1])
        self.a0 = AttnGate(nb[1], nb[0], nb[0]); self.d0 = VGGBlock(nb[1] + nb[0], nb[0], nb[0])
        self.final = nn.Conv2d(nb[0], 1, 1)

    def forward(self, x):
        e0 = self.e0(x); e1 = self.e1(self.pool(e0)); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); b = self.b(self.pool(e3))
        u3 = self.up(b); d3 = self.d3(torch.cat([u3, self.a3(u3, e3)], 1))
        u2 = self.up(d3); d2 = self.d2(torch.cat([u2, self.a2(u2, e2)], 1))
        u1 = self.up(d2); d1 = self.d1(torch.cat([u1, self.a1(u1, e1)], 1))
        u0 = self.up(d1); d0 = self.d0(torch.cat([u0, self.a0(u0, e0)], 1))
        return {"coarse": self.final(d0)}


class TVSeg(nn.Module):
    """torchvision FCN / DeepLabV3 (ResNet50) baseline, ImageNet-pretrained backbone.
    Grayscale input is repeated to 3 channels; single-class output."""
    def __init__(self, arch="fcn"):
        super().__init__()
        from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
        from torchvision.models import ResNet50_Weights
        ctor = fcn_resnet50 if arch == "fcn" else deeplabv3_resnet50
        # IMAGENET1K_V2 backbone weights are cached locally (V1/DEFAULT is not downloadable here)
        self.net = ctor(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V2, num_classes=1)
        self.variant = arch

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return {"coarse": self.net(x)["out"]}


def build_model(name, in_ch=1, deep_supervision=True):
    name = name.lower()
    if name in ("unet", "unet_boundary"):
        m = UNet(in_ch); m.variant = name; return m
    if name == "segformer_brm":
        return SegFormerBRM("mit_b1")
    if name == "segformer":
        return SMPWrap("mit_b1")
    if name == "attunet":
        m = AttUNet(in_ch); m.variant = name; return m
    if name in ("fcn", "deeplabv3"):
        return TVSeg("fcn" if name == "fcn" else "deeplab")
    if name == "unetpp":
        m = UNetPlusPlus(in_ch, deep_supervision=deep_supervision)
        m.variant = "unetpp"
        return m
    if name == "unetpp_tri":
        return BRUNetPlusPlus(in_ch, use_brm=False, use_trimap_input=True,
                              deep_supervision=deep_supervision)
    if name == "brunetpp":
        return BRUNetPlusPlus(in_ch, use_brm=True, use_trimap_input=True,
                              deep_supervision=deep_supervision, trimap_mode="morph")
    if name == "brunetpp_cat":   # ours: confidence-adaptive trimap (no hand-tuned kernel)
        return BRUNetPlusPlus(in_ch, use_brm=True, use_trimap_input=True,
                              deep_supervision=deep_supervision, trimap_mode="cat")
    if name == "brunetpp_mt":    # ours: multi-task (somite + injury) with adaptive trimap
        return BRUNetPlusPlus(in_ch, use_brm=True, use_trimap_input=True,
                              deep_supervision=deep_supervision, trimap_mode="cat", multitask=True)
    if name == "brunetpp_lcat":  # ours: learnable band (no kernel, no quantile hyperparam)
        return BRUNetPlusPlus(in_ch, use_brm=True, use_trimap_input=True,
                              deep_supervision=deep_supervision, trimap_mode="learn")
    if name == "brunetpp_inst":  # ours: CAT+BRM + myoseptal-boundary head for instances
        return BRUNetPlusPlus(in_ch, use_brm=True, use_trimap_input=True,
                              deep_supervision=deep_supervision, trimap_mode="cat", instseg=True)
    raise ValueError(f"unknown model {name}")


if __name__ == "__main__":
    for nm in ["unetpp", "unetpp_tri", "brunetpp"]:
        m = build_model(nm)
        x = torch.randn(2, 1, 256, 256)
        o = m(x)
        keys = {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in o.items()}
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{nm:12s} params={n:.2f}M  out={keys}")


class SMPWrap(nn.Module):
    """SegFormer (MiT) encoder + U-Net decoder baseline via segmentation_models_pytorch.
    Pads H,W to a multiple of 32 for the transformer stages; returns {coarse: logits}."""
    def __init__(self, enc="mit_b1"):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.Unet(encoder_name=enc, encoder_weights="imagenet", in_channels=1, classes=1)
    def forward(self, x):
        import torch.nn.functional as F
        H, W = x.shape[-2:]; Hp=(H+31)//32*32; Wp=(W+31)//32*32
        xr = F.interpolate(x, size=(Hp,Wp), mode="bilinear", align_corners=False) if (Hp,Wp)!=(H,W) else x
        y = self.net(xr)
        if (Hp,Wp)!=(H,W): y = F.interpolate(y, size=(H,W), mode="bilinear", align_corners=False)
        return {"coarse": y}


class SegFormerBRM(nn.Module):
    """SegFormer (MiT) encoder+decoder backbone + our coarse head, Confidence-Adaptive
    Trimap and Boundary Refinement Module. Tests whether our refinement helps a strong
    transformer backbone. Returns {coarse, refined, band}."""
    def __init__(self, enc="mit_b1", band_frac=0.10):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.Unet(encoder_name=enc, encoder_weights="imagenet", in_channels=1, classes=1)
        C = self.net.segmentation_head[0].in_channels
        self.coarse_head = nn.Conv2d(C, 1, 1)
        self.brm_conv = nn.Conv2d(C, C, 3, padding=1)
        self.refine = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(inplace=True),
                                    nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.refined_head = nn.Conv2d(C, 1, 1)
        self.band_frac = band_frac
    def forward(self, x):
        import torch.nn.functional as F
        H, W = x.shape[-2:]; Hp=(H+31)//32*32; Wp=(W+31)//32*32
        xr = F.interpolate(x, size=(Hp,Wp), mode="bilinear", align_corners=False) if (Hp,Wp)!=(H,W) else x
        feat = self.net.decoder(self.net.encoder(xr))
        pl = self.coarse_head(feat)
        with torch.no_grad():
            p = torch.sigmoid(pl); conf = (p-0.5).abs(); band = torch.zeros_like(p)
            for b in range(p.shape[0]):
                thr = torch.quantile(conf[b].flatten().float(), self.band_frac)
                band[b] = (conf[b] <= thr).float()
        fp = feat + self.brm_conv(feat) * band
        prl = self.refined_head(self.refine(fp))
        if (Hp,Wp)!=(H,W):
            pl = F.interpolate(pl, size=(H,W), mode="bilinear", align_corners=False)
            prl = F.interpolate(prl, size=(H,W), mode="bilinear", align_corners=False)
            band = F.interpolate(band, size=(H,W), mode="nearest")
        return {"coarse": pl, "refined": prl, "band": band}
