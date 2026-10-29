// C API exported to JavaScript (Emscripten). All heavy work happens here;
// the JS side only moves bytes (DICOM files, NIfTI volumes, weights) in and results out.
//
// Flow used by the web app:
//   1. volume : tsc_add_dicom()* + tsc_load_dicom_volume()   or   tsc_set_volume() (NIfTI)
//               -> canonical RAS CT kept in memory (tsc_ct_*)
//   2. tasks  : tsc_run() (CPU) or tsc_begin()/tiles/tsc_advance()/tsc_finish() (WebGPU),
//               any number of times on the same CT -> label map + label names
//   3. meshes : tsc_set_labels() (edited label map) + tsc_build_meshes("id\tname\n...")
// Tasks: "total" (+ roi_subset), "liver_segments", "liver_vessels",
// and "custom:<model id>[:<crop roi>]" for fine-tuned models.
#include <emscripten.h>

#include <chrono>
#include <map>
#include <memory>
#include <sstream>

#include "tsc/dicom.h"
#include "tsc/mesh.h"
#include "tsc/pipeline.h"
#include "liver_bundle.h"

using namespace tsc;

namespace {
std::vector<DicomSlice> g_slices;
int g_bad_files = 0;
std::map<int, ModelConfig> g_cfgs;
std::map<int, std::unique_ptr<UNet>> g_nets;
ClassMaps g_cm;
bool g_have_cm = false;
std::vector<std::string> g_labels;  // names of the last task result / meshed labels
std::unique_ptr<Runner> g_runner;
std::vector<float> g_tile_in, g_tile_out;
double g_t0 = 0;
Volume<float> g_ct;
Volume<uint8_t> g_seg;
Mesh g_mesh;  // meshes of the requested labels, concatenated
struct LabelMesh {
    int label;
    uint32_t v0, nv, i0, ni;
    int64_t voxels;
    double mesh_ml;
};
std::vector<LabelMesh> g_label_meshes;
std::string g_error, g_log, g_tmp;
// V3 liver bundle: label maps of the bundle tasks, kept by tsc_bundle_keep() (see liver_bundle.h)
std::map<std::string, browseg::TaskLabels> g_bundle;
std::string g_bundle_blob;  // "name\0contents\0" for every output file, then "__log__\0<log>\0"
double g_time_task = 0, g_time_mesh = 0;

void report(const std::string& stage, double f) {
    EM_ASM({
        if (Module.onProgress) Module.onProgress(UTF8ToString($0), $1);
    }, stage.c_str(), f);
}
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::vector<std::string> split_roi(const char* roi) {
    std::vector<std::string> r;
    if (!roi) return r;
    std::stringstream ss(roi);
    std::string t;
    while (std::getline(ss, t, ','))
        if (!t.empty() && t != "-") r.push_back(t);
    return r;
}
std::map<int, const ModelConfig*> model_map() {
    std::map<int, const ModelConfig*> m;
    for (auto& kv : g_cfgs) m[kv.first] = &kv.second;
    for (int tid : {298, 291, 292, 293, 294, 295, 570, 8, 343, 299, 300})  // V2: trunk_cavities, body, body_fast
        if (!m.count(tid)) m[tid] = nullptr;  // make_task reports missing models
    return m;
}
int fail(const std::exception& e) {
    g_error = e.what();
    return 1;
}
void need_ct() {
    if (!g_ct.size()) throw std::runtime_error("no CT volume loaded");
}
}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE const char* tsc_version() {
    return "BrowSeg 0.1 (TotalSegmentator 2.13/2.18 tasks, C++/WASM)";
}

EMSCRIPTEN_KEEPALIVE void tsc_set_threads(int n) { set_num_threads(n); }
// input resampling order: 1 = TotalSegmentator >= 2.16 (default), 3 = TS <= 2.15
EMSCRIPTEN_KEEPALIVE int tsc_set_resampling_order(int order) { if (order != 1 && order != 3) return -1; set_resampling_order(order); return 0; }

// Loads a .tsw model. build_net = 0 keeps only the configuration (the network runs on
// WebGPU). Returns the model's task id, or -1 on error.
EMSCRIPTEN_KEEPALIVE int tsc_load_weights(const uint8_t* data, size_t len, int build_net) {
    try {
        ModelWeights w = load_weights(data, len, !build_net);  // WebGPU path: the config is enough
        const int tid = w.cfg.task_id;
        std::unique_ptr<UNet> net = build_net ? std::make_unique<UNet>(w) : nullptr;  // may throw: commit after
        g_cfgs[tid] = w.cfg;
        if (net) g_nets[tid] = std::move(net);
        else g_nets.erase(tid);
        return tid;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE int tsc_has_model(int tid) { return g_cfgs.count(tid) ? 1 : 0; }

EMSCRIPTEN_KEEPALIVE int tsc_load_classmaps(const char* text) {
    try {
        g_cm = load_classmaps(text);
        g_have_cm = true;
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

// Writes the task ids of all models the task needs into out (max n); returns the count or -1.
EMSCRIPTEN_KEEPALIVE int tsc_task_models(const char* task, const char* roi, int* out, int n) {
    try {
        if (!g_have_cm) throw std::runtime_error("class maps not loaded");
        const std::vector<int> ids = task_model_ids(task, split_roi(roi), g_cm);
        int c = 0;
        for (int t : ids)
            if (c < n) out[c++] = t;
        return c;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

// ---------------------------------------------------------------- 1. volume
EMSCRIPTEN_KEEPALIVE void tsc_clear_dicom() {
    g_slices.clear();
    g_bad_files = 0;
}

// Returns 1 if the file was an image slice, 0 otherwise.
EMSCRIPTEN_KEEPALIVE int tsc_add_dicom(const uint8_t* data, size_t len) {
    try {  // (bad_alloc must not escape extern "C")
        DicomSlice s;
        std::string err;
        if (!parse_dicom(data, len, s, err)) {
            ++g_bad_files;
            g_error = err;
            return 0;
        }
        g_slices.push_back(std::move(s));
        return 1;
    } catch (const std::exception& e) {
        ++g_bad_files;
        g_error = e.what();
        return 0;
    }
}

// drops everything tied to the current volume (called only once the new volume will be built)
static void drop_volume_state() {
    g_runner.reset();          // a runner keeps references to the previous CT
    g_ct = Volume<float>();
    g_seg = Volume<uint8_t>();
    g_mesh = Mesh();
    g_label_meshes.clear();
    g_labels.clear();
}

// Series found in the added DICOM files, one per line (default choice first):
// uid \t description \t modality \t series number \t slices \t other images \t rows \t cols \t
// pixel spacing row \t col \t slice spacing \t extent mm \t series time \t contrast agent \t problem ("" = buildable)
EMSCRIPTEN_KEEPALIVE const char* tsc_list_series() {
    try {
        g_tmp.clear();
        if (g_slices.empty()) return g_tmp.c_str();
        auto clean = [](std::string v) { for (char& c : v) if (c == '\t' || c == '\n' || c == '\r') c = ' '; return v; };
        for (auto& e : list_series(g_slices)) {
            char num[160];
            std::snprintf(num, sizeof num, "%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.2f", e.series_number, e.num_slices, e.num_other,
                          e.rows, e.cols, e.pixel_spacing[0], e.pixel_spacing[1], e.slice_spacing, e.extent_mm);
            g_tmp += clean(e.series_uid) + "\t" + clean(e.description) + "\t" + clean(e.modality) + "\t" + num + "\t" +
                     clean(e.series_time) + "\t" + clean(e.contrast_agent) + "\t" + clean(e.problem) + "\n";
        }
    } catch (const std::exception& e) {
        fail(e);
        g_tmp.clear();
    }
    return g_tmp.c_str();
}

// Builds the canonical CT from the added DICOM slices: series `uid` (NULL = default: largest usable CT
// series; "" = the series without a UID). The previous volume is kept if the series cannot be built
// (it is freed only once all checks passed). keep != 0 keeps the slices to build another series later.
EMSCRIPTEN_KEEPALIVE int tsc_load_dicom_series(const char* uid, int keep) {
    try {
        std::string log;
        Volume<float> v = build_volume(g_slices, uid ? uid : "", log, uid != nullptr, drop_volume_state);
        g_ct = std::move(v);
        g_log = log;
        if (!keep) {
            g_slices.clear();
            g_slices.shrink_to_fit();
        }
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

// Builds the canonical CT from the added DICOM slices (largest CT series). Returns 0 on success.
EMSCRIPTEN_KEEPALIVE int tsc_load_dicom_volume() {
    try {
        std::string log;
        Volume<float> v = build_volume(g_slices, "", log, false, drop_volume_state);
        g_ct = std::move(v);
        g_log = log;
        g_slices.clear();
        g_slices.shrink_to_fit();
        g_seg = Volume<uint8_t>();
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

// CT from a NIfTI (data in NIfTI order: x fastest), affine = sform/qform (row-major 4x4).
EMSCRIPTEN_KEEPALIVE int tsc_set_volume(const float* data, int nx, int ny, int nz, const double* affine) {
    try {
        if (nx <= 0 || ny <= 0 || nz <= 0) throw std::runtime_error("bad volume size");
        {   // orientation checked before the current volume is dropped
            Volume<float> probe(Shape3{1, 1, 1});
            for (int i = 0; i < 16; ++i) probe.affine[i] = affine[i];
            (void)as_closest_canonical(probe);
        }
        drop_volume_state();
        Volume<float> v(Shape3{nx, ny, nz});
        for (int i = 0; i < 16; ++i) v.affine[i] = affine[i];
        parallel_for(nx, [&](int64_t b, int64_t e, int) {
            for (int64_t x = b; x < e; ++x)
                for (int64_t y = 0; y < ny; ++y)
                    for (int64_t z = 0; z < nz; ++z) v(x, y, z) = data[x + (int64_t)nx * (y + (int64_t)ny * z)];
        });
        g_ct = as_closest_canonical(v);
        g_log = "NIfTI volume " + std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) + "\n";
        g_seg = Volume<uint8_t>();
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

// NIfTI voxel data in the file's own type (raw bytes, little endian, x fastest): converted to float and reoriented to
// closest-canonical RAS in one pass (volume_from_nifti_raw). Same volume as tsc_set_volume with the float data, with
// about half of the peak memory for int16 CTs. slope/inter: scl_slope/scl_inter (slope 0 -> 1, 0 as in Nifti.read).
EMSCRIPTEN_KEEPALIVE int tsc_set_volume_raw(const uint8_t* raw, int datatype, double slope, double inter, int nx, int ny, int nz,
                                            const double* affine) {
    try {
        Affine a{};
        for (int i = 0; i < 16; ++i) a[i] = affine[i];
        g_ct = volume_from_nifti_raw(raw, datatype, slope, inter, Shape3{nx, ny, nz}, a, drop_volume_state);
        g_log = "NIfTI volume " + std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) + " (raw)\n";
        g_seg = Volume<uint8_t>();
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

EMSCRIPTEN_KEEPALIVE const float* tsc_ct_ptr() { return g_ct.data.data(); }
EMSCRIPTEN_KEEPALIVE const double* tsc_ct_affine() { return g_ct.affine.data(); }
// 0..2 shape, 3..5 zooms (mm)
EMSCRIPTEN_KEEPALIVE double tsc_ct_info(int what) {
    if (what < 0 || what > 5) return -1;
    if (what < 3) return (double)g_ct.shape[what];
    return zooms_from_affine(g_ct.affine)[what - 3];
}

// ---------------------------------------------------------------- 2. tasks
// CPU path (networks in WASM). Returns 0 on success.
EMSCRIPTEN_KEEPALIVE int tsc_run(const char* task, const char* roi) {
    try {
        if (!g_have_cm) throw std::runtime_error("class maps not loaded");
        need_ct();
        g_runner.reset();
        g_log.clear();
        std::map<int, const UNet*> nets;
        for (auto& kv : g_nets) nets[kv.first] = kv.second.get();
        g_t0 = now_s();
        std::unique_ptr<Runner> r = make_runner(g_ct, task, split_roi(roi), model_map(), g_cm, report);
        g_seg = run_runner(*r, nets, report);
        g_labels = r->labels();
        g_log += "timings:\n" + r->timings();
        g_time_task = now_s() - g_t0;
        return 0;
    } catch (const std::exception& e) {
        g_runner.reset();
        return fail(e);
    }
}

// step-wise API: the JS side runs the networks (WebGPU) tile by tile
EMSCRIPTEN_KEEPALIVE int tsc_begin(const char* task, const char* roi) {
    try {
        if (!g_have_cm) throw std::runtime_error("class maps not loaded");
        need_ct();
        g_runner.reset();  // free the previous runner first (peak memory)
        g_log.clear();
        g_t0 = now_s();
        g_runner = make_runner(g_ct, task, split_roi(roi), model_map(), g_cm, report);
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

// task id of the model the current stage needs, 0 when finished
EMSCRIPTEN_KEEPALIVE int tsc_stage() { return g_runner && !g_runner->done() ? g_runner->model_id() : 0; }
EMSCRIPTEN_KEEPALIVE const char* tsc_stage_name() { return g_runner ? g_runner->stage_name() : ""; }
EMSCRIPTEN_KEEPALIVE int tsc_num_tiles() { return g_runner && g_runner->window() ? g_runner->window()->num_tiles() : 0; }

// current sliding window, or null (no runner / finished / tile index out of range)
static SlidingWindow* cur_window(int t = 0) {
    SlidingWindow* sw = g_runner && !g_runner->done() ? g_runner->window() : nullptr;
    return sw && t >= 0 && t < sw->num_tiles() ? sw : nullptr;
}

EMSCRIPTEN_KEEPALIVE float* tsc_tile_input(int t) {
    SlidingWindow* sw = cur_window(t);
    if (!sw) return nullptr;
    g_tile_in.resize((size_t)sw->patch_numel());
    sw->tile_input(t, g_tile_in.data());
    return g_tile_in.data();
}

EMSCRIPTEN_KEEPALIVE float* tsc_tile_output() {
    SlidingWindow* sw = cur_window();
    if (!sw) return nullptr;
    g_tile_out.resize((size_t)sw->patch_numel() * sw->config().num_classes);
    return g_tile_out.data();
}

EMSCRIPTEN_KEEPALIVE void tsc_tile_accumulate(int t) {
    SlidingWindow* sw = cur_window(t);
    if (sw && g_tile_out.size() >= (size_t)sw->patch_numel() * sw->config().num_classes) sw->accumulate(t, g_tile_out.data());
}

// GPU accumulation path: the JS side accumulates fp16 logits on the GPU and copies
// them into tsc_logits_ptr() before tsc_advance(); C++ only tracks the counts.
EMSCRIPTEN_KEEPALIVE void tsc_tile_count(int t) { if (SlidingWindow* sw = cur_window(t)) sw->accumulate_counts(t); }
EMSCRIPTEN_KEEPALIVE uint16_t* tsc_logits_ptr() { SlidingWindow* sw = cur_window(); return sw ? sw->logits_data() : nullptr; }
EMSCRIPTEN_KEEPALIVE const float* tsc_gaussian_ptr() { SlidingWindow* sw = cur_window(); return sw ? sw->gaussian() : nullptr; }
// axis 0..2: tile origin in the padded volume; axis 3..5: padded volume shape
EMSCRIPTEN_KEEPALIVE double tsc_tile_geom(int t, int axis) {
    SlidingWindow* sw = cur_window(t);
    if (!sw || axis < 0 || axis > 5) return -1;
    return axis < 3 ? (double)sw->tile_origin(t)[axis] : (double)sw->padded_shape()[axis - 3];
}

EMSCRIPTEN_KEEPALIVE int tsc_advance() {
    try {
        if (!g_runner || g_runner->done()) throw std::runtime_error("no task stage in progress");
        g_runner->advance();
        return 0;
    } catch (const std::exception& e) {
        g_runner.reset();
        return fail(e);
    }
}

EMSCRIPTEN_KEEPALIVE int tsc_finish() {
    try {
        if (!g_runner || !g_runner->done()) throw std::runtime_error("task not finished");
        g_log += "timings:\n" + g_runner->timings();
        g_labels = g_runner->labels();
        g_seg = g_runner->take_result();
        g_runner.reset();
        g_tile_in = std::vector<float>();
        g_tile_out = std::vector<float>();
        g_time_task = now_s() - g_t0;
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

// label map of the last task (canonical, C order x,y,z) and its label names
EMSCRIPTEN_KEEPALIVE const uint8_t* tsc_labels_ptr() { return g_seg.data.data(); }
EMSCRIPTEN_KEEPALIVE int tsc_num_label_names() { return (int)g_labels.size(); }
EMSCRIPTEN_KEEPALIVE const char* tsc_label_name(int label) {
    g_tmp = label >= 0 && label < (int)g_labels.size() ? g_labels[label] : std::to_string(label);
    return g_tmp.c_str();
}

// ---------------------------------------------------------------- 3. meshes
// Replaces the label map (e.g. after manual editing); same grid as the CT.
EMSCRIPTEN_KEEPALIVE int tsc_set_labels(const uint8_t* data) {
    try {
        need_ct();
        g_seg = Volume<uint8_t>(g_ct.shape, g_ct.affine);
        std::memcpy(g_seg.data.data(), data, g_seg.data.size());
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

// spec: one "id<TAB>name" line per label to mesh (the name selects the mesh settings).
EMSCRIPTEN_KEEPALIVE int tsc_build_meshes(const char* spec) {
    try {
        const double t = now_s();
        g_mesh = Mesh();
        g_label_meshes.clear();
        std::vector<std::pair<int, std::string>> items;
        std::stringstream ss(spec ? spec : "");
        std::string line;
        while (std::getline(ss, line)) {
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            items.push_back({std::atoi(line.substr(0, tab).c_str()), line.substr(tab + 1)});
        }
        std::vector<int64_t> counts(256, 0);
        for (auto v : g_seg.data) counts[v]++;
        for (size_t k = 0; k < items.size(); ++k) {
            const int l = items[k].first;
            if (l <= 0 || l > 255) continue;
            Mesh m = counts[l] ? mask_to_mesh(g_seg, l, mesh_options_for(items[k].second, false)) : Mesh();
            LabelMesh lm{l, (uint32_t)g_mesh.num_vertices(), (uint32_t)m.num_vertices(), (uint32_t)g_mesh.indices.size(),
                         (uint32_t)m.indices.size(), counts[l], m.num_triangles() ? mesh_volume_ml(m) : 0.0};
            for (auto& idx : m.indices) g_mesh.indices.push_back(idx + lm.v0);
            g_mesh.vertices.insert(g_mesh.vertices.end(), m.vertices.begin(), m.vertices.end());
            g_mesh.normals.insert(g_mesh.normals.end(), m.normals.begin(), m.normals.end());
            g_label_meshes.push_back(lm);
            report("mesh", (double)(k + 1) / items.size());
        }
        g_time_mesh = now_s() - t;
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

// 0 vertices(float*) 1 indices(uint32*) 2 normals(float*) 5 error string 6 log string
EMSCRIPTEN_KEEPALIVE const void* tsc_result_ptr(int what) {
    switch (what) {
        case 0: return g_mesh.vertices.data();
        case 1: return g_mesh.indices.data();
        case 2: return g_mesh.normals.data();
        case 3: return g_seg.data.data();
        case 5: return g_error.c_str();
        case 6: return g_log.c_str();
    }
    return nullptr;
}

// 0 #vertices 1 #triangles 6 #slices added 7 #bad files 8 threads 10 #meshed labels
EMSCRIPTEN_KEEPALIVE double tsc_result_int(int what) {
    switch (what) {
        case 0: return (double)g_mesh.num_vertices();
        case 1: return (double)g_mesh.num_triangles();
        case 6: return (double)g_slices.size();
        case 7: return g_bad_files;
        case 8: return num_threads();
        case 10: return (double)g_label_meshes.size();
    }
    return 0;
}

// per meshed label i: 0 label id, 1 first vertex, 2 #vertices, 3 first index, 4 #indices,
// 5 voxels, 6 mask volume ml, 7 mesh volume ml
EMSCRIPTEN_KEEPALIVE double tsc_label_info(int i, int what) {
    if (i < 0 || (size_t)i >= g_label_meshes.size()) return -1;
    const LabelMesh& l = g_label_meshes[(size_t)i];
    auto z = zooms_from_affine(g_seg.affine);
    switch (what) {
        case 0: return l.label;
        case 1: return l.v0;
        case 2: return l.nv;
        case 3: return l.i0;
        case 4: return l.ni;
        case 5: return (double)l.voxels;
        case 6: return l.voxels * z[0] * z[1] * z[2] / 1000.0;
        case 7: return l.mesh_ml;
    }
    return 0;
}

// 1 task seconds 2 mesh seconds
EMSCRIPTEN_KEEPALIVE double tsc_result_double(int what) {
    switch (what) {
        case 1: return g_time_task;
        case 2: return g_time_mesh;
    }
    return 0;
}

EMSCRIPTEN_KEEPALIVE void tsc_free_results() {
    g_mesh = Mesh();
    g_label_meshes.clear();
}

// ---------------------------------------------------------------- 4. liver bundle (V3)
// JS runs the four tasks one by one (tsc_run, or the step-wise GPU API) and calls tsc_bundle_keep() after each;
// tsc_bundle_build() then makes the BLISS simulator file set as LiverModelExporter does.
EMSCRIPTEN_KEEPALIVE void tsc_bundle_clear() {
    g_bundle.clear();
    g_bundle_blob = std::string();
}

// moves the label map of the last task into the bundle under `key` (task name), cropped to its labelled voxels (memory)
EMSCRIPTEN_KEEPALIVE int tsc_bundle_keep(const char* key) {
    try {
        if (!g_seg.size()) throw std::runtime_error("no task result to keep");
        browseg::TaskLabels t = browseg::crop_to_labels(std::move(g_seg), g_labels);
        g_seg = Volume<uint8_t>();
        g_bundle[key ? key : ""] = std::move(t);
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

// builds the files; afterwards the display label map is the current label map (tsc_labels_ptr / tsc_label_name)
// ivc_into_vein, splenic_vein: options of bundle 0.2 (0 = default: ivc.obj, no splenic vein); ivc_margin_mm: IVC cut
EMSCRIPTEN_KEEPALIVE int tsc_bundle_build(double margin_mm, double tumor_min_ml, int lps, int ivc_into_vein, int splenic_vein,
                                          double ivc_margin_mm, const char* version) {
    try {
        const double t = now_s();
        browseg::BundleOptions o;
        o.margin_mm = margin_mm;
        o.tumor_min_ml = tumor_min_ml;
        o.lps = lps != 0;
        o.ivc_into_vein = ivc_into_vein != 0;
        o.splenic_vein = splenic_vein != 0;
        o.ivc_margin_mm = ivc_margin_mm;
        if (version && *version) o.version = version;
        browseg::BundleResult r = browseg::build_liver_bundle(g_bundle, o);
        g_bundle.clear();  // memory: the four label maps are no longer needed
        g_bundle_blob.clear();
        for (auto& f : r.files) {
            g_bundle_blob += f.first;
            g_bundle_blob.push_back('\0');
            g_bundle_blob += f.second;
            g_bundle_blob.push_back('\0');
        }
        g_bundle_blob += "__log__";
        g_bundle_blob.push_back('\0');
        g_bundle_blob += r.log;
        g_bundle_blob.push_back('\0');
        g_seg = std::move(r.display);
        g_labels = r.display_names;
        g_time_mesh = now_s() - t;
        return 0;
    } catch (const std::exception& e) {
        g_bundle.clear();
        return fail(e);
    }
}
EMSCRIPTEN_KEEPALIVE const char* tsc_bundle_blob_ptr() { return g_bundle_blob.data(); }
EMSCRIPTEN_KEEPALIVE double tsc_bundle_blob_len() { return (double)g_bundle_blob.size(); }

}  // extern "C"
