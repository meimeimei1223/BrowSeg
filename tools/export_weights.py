"""
Convert a TotalSegmentator / nnU-Net v2 model folder (PlainConvUNet, 3d_fullres)
into the self-contained .tsw file read by the C++ engine.

This is a one-time, offline conversion; the C++/WASM runtime never needs Python.

usage: python export_weights.py <task_id> <out.tsw> [--fp16] [--model-dir DIR] [--fold F]
       (--model-dir: any nnU-Net 3d_fullres model folder, e.g. a fine-tuned model; --fold: 0 / all)

File layout:
  "TSCPPW01"                       8 bytes magic
  uint32 header_len                little endian
  header (ASCII, header_len bytes) line based, see below
  padding to 64 bytes, then tensor blobs (each 64-byte aligned)
"""
import sys, os, json, glob, struct
import numpy as np
import torch

R = os.path.join(os.path.expanduser("~"), ".totalsegmentator", "nnunet", "results")


def find_model_dir(task_id):
    ds = glob.glob(os.path.join(R, f"Dataset{int(task_id):03d}_*"))
    assert len(ds) == 1, ds
    cands = glob.glob(os.path.join(ds[0], "*__nnUNetPlans__3d_fullres"))
    assert len(cands) == 1, cands
    return cands[0]


def main():
    task_id, out_path = sys.argv[1], sys.argv[2]
    fp16 = "--fp16" in sys.argv
    opt = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    mdir = opt("--model-dir") or find_model_dir(task_id)
    fold = opt("--fold", "0")
    plans = json.load(open(os.path.join(mdir, "plans.json")))
    dsj = json.load(open(os.path.join(mdir, "dataset.json")))
    cfg = plans["configurations"]["3d_fullres"]
    if "architecture" in cfg:  # newer nnU-Net v2 plans (e.g. Dataset343 trunk_cavities): same values under arch_kwargs
        a = cfg["architecture"]; k = a["arch_kwargs"]
        assert a["network_class_name"].endswith(".PlainConvUNet"), a["network_class_name"]
        assert k["conv_op"].endswith("Conv3d") and k["norm_op"].endswith("InstanceNorm3d") and k["conv_bias"], k
        assert k["norm_op_kwargs"] == {"eps": 1e-05, "affine": True}, k["norm_op_kwargs"]
        assert k["nonlin"].endswith("LeakyReLU") and k["nonlin_kwargs"].get("negative_slope", 0.01) == 0.01, k["nonlin_kwargs"]
        assert k["dropout_op"] is None, k["dropout_op"]
        cfg = dict(cfg, UNet_class_name="PlainConvUNet", pool_op_kernel_sizes=k["strides"], conv_kernel_sizes=k["kernel_sizes"],
                   n_conv_per_stage_encoder=k["n_conv_per_stage"], n_conv_per_stage_decoder=k["n_conv_per_stage_decoder"],
                   UNet_base_num_features=k["features_per_stage"][0], unet_max_num_features=max(k["features_per_stage"]))
        assert [min(cfg["UNet_base_num_features"] * 2 ** i, cfg["unet_max_num_features"]) for i in range(k["n_stages"])] \
            == k["features_per_stage"], k["features_per_stage"]   # the old-format formula must give the same widths
    assert cfg["UNet_class_name"] == "PlainConvUNet", cfg["UNet_class_name"]
    assert cfg["normalization_schemes"] == ["CTNormalization"]
    assert plans["transpose_forward"] == [0, 1, 2]
    ck = torch.load(os.path.join(mdir, f"fold_{fold}", "checkpoint_final.pth"), map_location="cpu", weights_only=False)
    sd = ck["network_weights"]
    mirror = ck.get("inference_allowed_mirroring_axes")

    n_stages = len(cfg["pool_op_kernel_sizes"])
    feats = [min(cfg["UNet_base_num_features"] * 2 ** i, cfg["unet_max_num_features"]) for i in range(n_stages)]
    labels = dsj["labels"]
    num_classes = len(labels)
    ip = plans["foreground_intensity_properties_per_channel"]["0"]

    tensors = []  # (name, array)

    def add(name, key):
        t = sd[key].float().numpy()
        tensors.append((name, t))

    for s in range(n_stages):
        for c in range(cfg["n_conv_per_stage_encoder"][s]):
            k = f"encoder.stages.{s}.0.convs.{c}"
            add(f"enc.{s}.{c}.w", k + ".conv.weight"); add(f"enc.{s}.{c}.b", k + ".conv.bias")
            add(f"enc.{s}.{c}.nw", k + ".norm.weight"); add(f"enc.{s}.{c}.nb", k + ".norm.bias")
    for s in range(n_stages - 1):
        add(f"up.{s}.w", f"decoder.transpconvs.{s}.weight"); add(f"up.{s}.b", f"decoder.transpconvs.{s}.bias")
        for c in range(cfg["n_conv_per_stage_decoder"][s]):
            k = f"decoder.stages.{s}.convs.{c}"
            add(f"dec.{s}.{c}.w", k + ".conv.weight"); add(f"dec.{s}.{c}.b", k + ".conv.bias")
            add(f"dec.{s}.{c}.nw", k + ".norm.weight"); add(f"dec.{s}.{c}.nb", k + ".norm.bias")
    add("seg.w", f"decoder.seg_layers.{n_stages - 2}.weight"); add("seg.b", f"decoder.seg_layers.{n_stages - 2}.bias")

    lines = ["TSCPP_WEIGHTS 1",
             f"name {plans['dataset_name']}",
             f"task_id {int(task_id)}",
             "arch PlainConvUNet",
             f"patch {' '.join(map(str, cfg['patch_size']))}",
             f"spacing {' '.join(repr(float(x)) for x in cfg['spacing'])}",
             f"ct_norm {ip['mean']!r} {ip['std']!r} {ip['percentile_00_5']!r} {ip['percentile_99_5']!r}",
             f"n_stages {n_stages}",
             f"features {' '.join(map(str, feats))}",
             f"strides {' '.join(str(v) for k in cfg['pool_op_kernel_sizes'] for v in k)}",
             f"kernels {' '.join(str(v) for k in cfg['conv_kernel_sizes'] for v in k)}",
             f"n_conv_enc {' '.join(map(str, cfg['n_conv_per_stage_encoder']))}",
             f"n_conv_dec {' '.join(map(str, cfg['n_conv_per_stage_decoder']))}",
             f"num_classes {num_classes}",
             f"mirroring {0 if not mirror else 1}",
             "norm InstanceNorm3d 1e-05",
             "nonlin LeakyReLU 0.01"]
    for name, idx in sorted(labels.items(), key=lambda kv: kv[1]):
        lines.append(f"label {idx} {name}")

    # compute offsets
    blobs = []
    off = 0
    tlines = []
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr.astype(np.float16 if fp16 else np.float32))
        b = arr.tobytes()
        tlines.append(f"tensor {name} {'f16' if fp16 else 'f32'} {arr.ndim} {' '.join(map(str, arr.shape))} {off} {len(b)}")
        blobs.append((off, b))
        off += (len(b) + 63) // 64 * 64
    header = ("\n".join(lines + tlines + ["end"]) + "\n").encode("utf-8")  # label names may be non-ASCII
    with open(out_path, "wb") as f:
        f.write(b"TSCPPW01")
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        pos = f.tell()
        pad = (64 - pos % 64) % 64
        f.write(b"\0" * pad)
        base = f.tell()
        for o, b in blobs:
            f.seek(base + o)
            f.write(b)
    print(f"wrote {out_path}: {len(tensors)} tensors, {off/1e6:.1f} MB data, fp16={fp16}")


if __name__ == "__main__":
    main()
