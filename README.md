# BrowSeg

TotalSegmentator in the web browser.

BrowSeg runs the CT segmentation models of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
(nnU-Net) entirely inside the browser: the CT is read, resampled, segmented and written back on your own computer.
Nothing is installed and no image leaves the computer; the only download is the model weights, which the browser
keeps after the first use. The network runs on the GPU through WebGPU, or on the CPU (WebAssembly) when WebGPU is not
available.

For the three liver tasks (liver, Couinaud segments, intrahepatic vessels), the label maps were compared voxel by voxel
with TotalSegmentator 2.18.0 run on the CPU, on 30 abdominal CTs of the Medical Segmentation Decathlon liver task
(from LiTS): 80 of the 90 case x task pairs were identical on the GPU path and 81 of 90 on the CPU path, and the others
differed by 1 to 8 voxels. The records are in `paper/`.

> **Not a medical device.** BrowSeg is research software. Do not use its output for diagnosis or treatment.

## Use it

Open **https://browseg.github.io/BrowSeg/** in Chrome and drop a folder with the DICOM files of one CT series, or a
NIfTI file (`.nii`, `.nii.gz`), onto the page. The page shows whether the network runs on the GPU (WebGPU) or on the
CPU. Then pick a task:

| Task | TotalSegmentator task | Output | Weights download (first time only) |
|---|---|---|---|
| Liver | `total --roi_subset liver` | liver | 191 MB |
| Abdominal organs | `total --roi_subset ...` | selected organs | 191 MB |
| All structures | `total` | 117 structures | 624 MB |
| Liver segments | `liver_segments` | Couinaud segments 1–8 | 189 MB |
| Liver vessels | `liver_vessels` | intrahepatic vessels, liver tumour | 189 MB |

The paper checked the agreement with TotalSegmentator for Liver, Liver segments and Liver vessels only; the other
tasks run the same code but were not part of that check.

Results can be saved as NIfTI label maps (`.nii.gz`) and as meshes (OBJ / STL).

### Liver bundle (3D models for liver surgery simulation)

The *Analyze* tab has **Run liver bundle**. It runs four models one after the other and writes one ZIP of OBJ meshes:

| Files | Source |
|---|---|
| `liver.obj`, `gb.obj` (gallbladder), `ivc.obj` (inferior vena cava) | `total --roi_subset liver,gallbladder,inferior_vena_cava,portal_vein_and_splenic_vein` |
| `S1.obj` … `S8.obj` (Couinaud segments) | `liver_segments` |
| `portal.obj`, `vein.obj` (portal vein, hepatic veins) | OpenPVHV (below), clipped to the liver + 10 mm |
| `tumor.obj` (tumour candidates of 1 mL or more) | `liver_vessels` |

`ivc.obj` keeps the inferior vena cava from 10 mm above the top of the liver to 10 mm below its bottom. Two options
change the set: *Put the inferior vena cava into the hepatic vein file* (for simulators that read no `ivc.obj`) and
*Add the splenic vein to the portal vein* (TotalSegmentator's portal and splenic vein, not clipped).
*Extra organs (optional)* adds other TotalSegmentator structures as OBJ files in the `extra/` folder of the ZIP.
`export_log.txt` in the ZIP records the models, options and times. The weights download for the bundle is about
0.69 GB the first time.

TotalSegmentator does not separate the portal vein from the hepatic veins, so the bundle uses the separately
released OpenPVHV model for them. Its output in BrowSeg was compared with OpenPVHV run in Python on the CPU.

**Tested environments.** Chrome 153 and 154 on Windows 11 (NVIDIA GeForce RTX 4070) and Chrome 142 on Ubuntu 22.04 (NVIDIA
T1200, 4 GB). On Linux, WebGPU had to be enabled in Chrome: open `chrome://flags`, set **Unsafe WebGPU Support** and
**Vulkan** to Enabled, and relaunch Chrome. Without WebGPU, BrowSeg runs on the CPU.

**Input.** Uncompressed DICOM (one series) or NIfTI. Compressed DICOM and multi-frame DICOM are not supported.

**Memory.** The application can use up to 4 GiB (the limit of 32-bit WebAssembly). The largest tested CT
(512 x 512 x 1,026 voxels) used 3.68 GiB on the CPU path and 3.15 GiB on the GPU path.

**Two builds.** The page picks the build automatically:
- *multi-threaded* (fast pre/post-processing) — needs a cross-origin-isolated page (COOP/COEP headers);
  on GitHub Pages this is done by a small service worker, which reloads the page once on the first visit;
- *single-threaded* — used when isolation is not available (for example when the app is embedded in another
  page). Same results, slower pre/post-processing.

**Your own server.** BrowSeg is a set of static files: copy `web/` and `weights/` to any web server (or run
`node serve.mjs`). Check the weight files against the SHA-256 values in `weights/MANIFEST.json`.

## Check the results yourself

To compare BrowSeg with TotalSegmentator on your own CT:

1. In BrowSeg, run a task and save the labels (`browseg_labels.nii.gz`).
2. Run TotalSegmentator 2.18.0 (the version checked in the paper) on the same CT **on the CPU**:
   `TotalSegmentator -i <CT> -o ts.nii.gz --ml --task liver_segments --device cpu`
   (for the liver: `--task total --roi_subset liver`).
3. `python tools/compare_with_totalsegmentator.py browseg_labels.nii.gz ts.nii.gz`

The script prints the number of differing voxels and the Dice coefficient per label. Compare against a CPU run of
TotalSegmentator: on a GPU it computes the network in mixed precision, which can change the output.

## Reproduce the paper's agreement

1. Obtain Task03_Liver of the Medical Segmentation Decathlon (http://medicaldecathlon.com/, CC BY-SA 4.0). Place the
   30 cases listed in `paper/data/lits_cases.csv` as `cases/<record_id>/<task03_file>` (for example
   `cases/lits_092/liver_92.nii.gz`). The SHA-256 column lets you confirm that each file is the one used in the paper.
2. Open https://browseg.github.io/BrowSeg/bench.html?local=1&tasks=total:liver;liver_segments:-;liver_vessels:- in
   Chrome, choose the `cases` folder and press *Run* (add `&cpu=1` for the CPU path).
3. The page compares the hash of every label map with the paper's (`paper/data/desktop_hashes.json` for the GPU path,
   `paper/data/desktop_hashes_wasm20.json` for the CPU path). *Download results (JSON)* saves the comparison;
   `python tools/verify_bench.py <file>.json` prints it again.

Equal hashes mean identical label maps. The CT never leaves your computer.

## Models and licences

The weights in `weights/` are the official TotalSegmentator checkpoints (fold 0 of the `3d_fullres` nnU-Net
models) converted to BrowSeg's `.tsw` format, unchanged in value. Only tasks that TotalSegmentator releases
under the Apache License 2.0 are included; see `weights/MANIFEST.json` for the upstream release, file
hashes and licence of every file, and `weights/NOTICE` for the required attribution. `openpvhv_950.tsw` (used only
by the liver bundle) is the OpenPVHV model weights v1.0 (Apache License 2.0, https://doi.org/10.5281/zenodo.23619753)
converted in the same way; it is attached to the release `paper-v3.0`. When you use BrowSeg in
research, please cite TotalSegmentator and nnU-Net (below) and the BrowSeg paper.

BrowSeg itself (the C++ / WebAssembly engine in `src/` and `web/`, the JavaScript, the tools and the built bundles in
`web/dist/` and `web/dist-st/`) is released under the Apache License 2.0 (`LICENSE`). `DEVELOPING.md` describes how
to build it.

## Paper

Manuscript under review. The bibliographic details will be added here after peer review.

## Citations

- Wasserthal J, Breit H-C, Meyer MT, et al. TotalSegmentator: Robust segmentation of 104 anatomic structures
  in CT images. Radiol Artif Intell 2023;5(5):e230024. doi:10.1148/ryai.230024
- Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep
  learning-based biomedical image segmentation. Nat Methods 2021;18:203–211. doi:10.1038/s41592-020-01008-z
- liver_segments model: Tian J, Liu L, Shi Z, Xu F. Automatic Couinaud segmentation from CT volumes on liver using
  GLC-UNet. In: Machine Learning in Medical Imaging (MLMI 2019), LNCS. Springer; 2019:274–282. doi:10.1007/978-3-030-32692-0_32
- liver_vessels model: Simpson AL, Antonelli M, Bakas S, et al. A large annotated medical image dataset for the development
  and evaluation of segmentation algorithms. arXiv:1902.09063, 2019

- OpenPVHV (portal vein / hepatic vein model used by the liver bundle): preprint, Research Square 2026,
  doi:10.21203/rs.3.rs-11500369/v1; weights doi:10.5281/zenodo.23619753

## Version of the paper

The paper's measurements were made with the build at commit 022925c (`web/dist/tsc.wasm` SHA-256 `fda8ec7a…`;
see `EXPORT_INFO.txt`). The tag of the paper's version is `paper-v1.4`.

The liver bundle and OpenPVHV (version 3) were measured with a development build of the same source; the published
build of the tag `paper-v3.0` gave the same output files, byte for byte, on all 30 cases (`web/dist/tsc.wasm` SHA-256
`602d0e8f…`).
