// Liver bundle (BrowSeg V3): the label maps of four tasks -> the BLISS simulator model/ set (14 OBJ files).
//
// The same steps as LiverModelExporter (Desktop/LiverModelExporter, src/exporter.cpp, Engine::segment + Engine::write,
// version 0.3, 2026-10-18), ported without its GPU, threads and file system so that the browser gives the same files:
//   liver, gb      TS total (roi_subset liver, gallbladder, inferior_vena_cava, portal_vein_and_splenic_vein)
//   portal         model 950 "portal...", inside liver + margin_mm (option: + TS portal_vein_and_splenic_vein, uncut)
//   vein           model 950 "hepatic...", inside liver + margin_mm (option: + the IVC below, for BLISS)
//   ivc            TS inferior_vena_cava between the liver's top and bottom + ivc_margin_mm (whole cross-sections)
// Version 0.2 (2026-10-26, author): portal and vein from model 950 only, the IVC as its own file. Version 0.1 was
// LiverModelExporter 0.3 (TS portal trunk + splenic vein in portal, TS IVC in vein, both clipped to liver + margin).
//   tumor          TS liver_vessels "liver_tumor": 26-connected pieces, kept when >= tumor_min_ml
//   S1..S8         TS liver_segments
// Mesh settings are tsc_core's mesh_options_for(); the frame is LPS mm (x, y negated) by default, as the simulator's
// model/. Only vessels = model950 is implemented (the exporter's "split" fallback is not).
#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

#include "tsc/common.h"
#include "tsc/mesh.h"

namespace browseg {

struct TaskLabels {
    tsc::Volume<uint8_t> seg;         // CT grid, or a crop of it (memory): affine = the CT grid's
    std::vector<std::string> labels;  // label index -> name
    std::array<int64_t, 3> origin{0, 0, 0};  // index of seg(0, 0, 0) on the CT grid
    tsc::Shape3 full{0, 0, 0};               // CT grid shape (0 = seg is the whole grid)
    tsc::Shape3 grid() const { return full[0] ? full : seg.shape; }
    int label(const std::string& name) const;
    int label_like(const std::string& part) const;  // first label (> 0) whose name contains part
};

struct BundleOptions {
    double margin_mm = 10.0;
    double tumor_min_ml = 1.0;
    bool lps = true;
    bool ivc_into_vein = false;   // the IVC in vein.obj instead of ivc.obj (BLISS reads no ivc.obj)
    bool splenic_vein = false;    // TS portal_vein_and_splenic_vein added to portal.obj, uncut
    double ivc_margin_mm = 10.0;  // IVC kept from the liver's top - margin to its bottom + margin (cranio-caudal)
    std::string version = "BrowSeg V3 liver bundle";  // second comment line of every OBJ
};

struct BundleOrgan {
    std::string stem, source, note;
    double voxel_ml = 0, mesh_ml = 0;
    size_t vertices = 0, triangles = 0;
    bool written = false, placeholder = false;
};

struct BundleTumor {
    double ml = 0;
    double pos[3] = {0, 0, 0};  // centroid, output frame (mm)
    bool keep = false;
};

struct BundleResult {
    std::vector<std::pair<std::string, std::string>> files;  // (file name, contents): OBJ files, preset_choice.txt
    std::vector<BundleOrgan> organs;                          // file order: liver, gb, portal, vein, ivc, tumor, S1..S8
    std::vector<BundleTumor> tumors;                          // largest first
    std::string log;                                          // tumor candidates + files (for export_log.txt)
    // display label map on the CT grid: 1 liver, 2 portal, 3 vein, 4 gallbladder, 5 kept tumor, 6 IVC, 11..18 S1..S8
    tsc::Volume<uint8_t> display;
    std::vector<std::string> display_names;
};

// A task's label map on the CT grid -> its crop to the box of the labelled voxels (all voxels outside are 0, so nothing
// is lost; a map without labels becomes a 1-voxel crop).
TaskLabels crop_to_labels(tsc::Volume<uint8_t>&& seg, const std::vector<std::string>& labels);

// tasks: "total", "liver_segments", "custom:950:liver", "liver_vessels" (all on the same CT grid, cropped or not).
// The masks are made in a box that holds every voxel that can reach an output (liver + margin_mm, gallbladder,
// liver_segments, liver_vessels); the files are the same as without the box.
BundleResult build_liver_bundle(const std::map<std::string, TaskLabels>& tasks, const BundleOptions& opt);

}  // namespace browseg
