#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "resample.h"

namespace tsc {

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// TS change_spacing(img, new_spacing, order=resampling_order, dtype=int32); new_spacing is (x, y, z)
template <class T>
Volume<int32_t> resample_to_spacing(const Volume<T>& v, const std::array<double, 3>& new_spacing, int order) {
    const auto z = zooms_from_affine(v.affine);
    Volume<int32_t> out;
    if (new_spacing[0] == 0.0) {  // TS resample=None: image handed to nnU-Net unchanged
        out.shape = v.shape;
        out.affine = v.affine;
        out.data.resize(v.data.size());
        for (size_t i = 0; i < v.data.size(); ++i) out.data[i] = (int32_t)(double)v.data[i];
        return out;
    }
    std::array<double, 3> zoom;
    for (int d = 0; d < 3; ++d) zoom[d] = z[d] / new_spacing[d];
    out.affine = v.affine;
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) out.affine[r * 4 + c] = v.affine[r * 4 + c] / zoom[c];
    if (z[0] == new_spacing[0] && z[1] == new_spacing[1] && z[2] == new_spacing[2]) {  // np.array_equal -> no resampling
        out.shape = v.shape;
        out.affine = v.affine;
        out.data.resize(v.data.size());
        for (size_t i = 0; i < v.data.size(); ++i) out.data[i] = (int32_t)(double)v.data[i];
        return out;
    }
    out.shape = zoom_output_shape(v.shape, zoom);
    if (zoom[0] == 1.0 && zoom[1] == 1.0 && zoom[2] == 1.0) {  // scipy early exit
        out.data.resize(v.data.size());
        for (size_t i = 0; i < v.data.size(); ++i) out.data[i] = (int32_t)(double)v.data[i];
        return out;
    }
    auto r = scipy_zoom(v.data.data(), v.shape, out.shape, order);
    out.data.resize(r.size());
    for (size_t i = 0; i < r.size(); ++i) out.data[i] = (int32_t)r[i];  // astype(int32): truncation
    return out;
}

// TS change_spacing(seg, ..., target_shape, order=0, dtype=uint8)
Volume<uint8_t> resample_labels_to_shape(const Volume<uint8_t>& v, const Shape3& target, const Affine& aff) {
    std::array<double, 3> zoom;
    for (int d = 0; d < 3; ++d) zoom[d] = (double)target[d] / (double)v.shape[d];
    Volume<uint8_t> out(zoom_output_shape(v.shape, zoom), aff);
    if (out.shape != target) throw std::runtime_error("label resampling produced unexpected shape");
    if (zoom[0] == 1.0 && zoom[1] == 1.0 && zoom[2] == 1.0) {
        out.data = v.data;
        return out;
    }
    out.data = scipy_zoom_nearest_u8(v.data.data(), v.shape, out.shape);
    return out;
}

}  // namespace

// ---------------------------------------------------------------- class maps
int ClassMaps::task_index(const std::string& task_name, const std::string& cls) const {
    auto it = task.find(task_name);
    if (it == task.end()) throw std::runtime_error("unknown task in class maps: " + task_name);
    for (size_t i = 0; i < it->second.size(); ++i)
        if (it->second[i] == cls) return (int)i;
    throw std::runtime_error("class '" + cls + "' not in task " + task_name);
}

ClassMaps load_classmaps(const std::string& text) {
    ClassMaps cm;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        std::istringstream ls(line);
        std::string kind;
        ls >> kind;
        if (kind == "task") {
            std::string t, name;
            int idx;
            ls >> t >> idx >> name;
            auto& v = cm.task[t];
            if ((int)v.size() <= idx) v.resize(idx + 1);
            v[idx] = name;
        } else if (kind == "part") {
            int tid, idx;
            std::string name;
            ls >> tid >> idx >> name;
            cm.part[tid][idx] = name;
        }
    }
    for (auto& kv : cm.task)
        if (!kv.second.empty() && kv.second[0].empty()) kv.second[0] = "background";
    return cm;
}

ClassMaps load_classmaps_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return load_classmaps(ss.str());
}

// ---------------------------------------------------------------- task specs (python_api.py)
namespace {
int g_resampling_order = 1;
}
void set_resampling_order(int order) {
    if (order != 1 && order != 3) throw std::runtime_error("resampling order must be 1 or 3");
    g_resampling_order = order;
}
int resampling_order() { return g_resampling_order; }

TaskSpec make_task(const std::string& task, const std::vector<std::string>& roi_subset,
                   const std::map<int, const ModelConfig*>& models, const ClassMaps& cm) {
    auto model = [&](int tid) -> const ModelConfig* {
        auto it = models.find(tid);
        if (it == models.end() || !it->second) throw std::runtime_error("model " + std::to_string(tid) + " not loaded");
        return it->second;
    };
    TaskSpec s;
    s.name = task;
    s.resampling_order = g_resampling_order;
    if (task == "total") {
        s.labels = cm.task.at("total");
        s.spacing = {1.5, 1.5, 1.5};
        s.step_size = 0.8;  // TS 2.18 nnunet.py: step_size = 0.8 for task_name in [total, total_v3, total_mr], else 0.5
        for (int tid : {291, 292, 293, 294, 295}) {
            const auto& pm = cm.part.at(tid);
            bool needed = roi_subset.empty();
            for (auto& r : roi_subset)
                for (auto& kv : pm) needed |= kv.second == r;
            if (!needed) continue;
            TaskModel m;
            m.task_id = tid;
            m.cfg = model(tid);
            m.to_task.assign(m.cfg->num_classes, -1);
            for (auto& kv : pm) m.to_task.at(kv.first) = cm.task_index("total", kv.second);
            s.models.push_back(m);
        }
        if (!roi_subset.empty()) {  // roi_subset: rough 6 mm segmentation for cropping
            s.crop_cfg = model(298);
            for (auto& r : roi_subset) {
                const int li = cm.task_index("total", r);
                s.crop_labels.push_back(li);
                s.keep.push_back(li);
            }
        }
    } else if (task == "liver_segments") {
        s.labels = cm.task.at("liver_segments");
        s.spacing = {0.8046879768371582, 0.8046879768371582, 1.5};
        s.step_size = 0.5;
        s.crop_cfg = model(298);
        s.crop_labels.push_back(cm.task_index("total", "liver"));
        TaskModel m;
        m.task_id = 570;
        m.cfg = model(570);
        m.to_task.resize(m.cfg->num_classes);
        for (int i = 0; i < m.cfg->num_classes; ++i) m.to_task[i] = i;  // single model: raw prediction
        s.models.push_back(m);
        for (auto& r : roi_subset) s.keep.push_back(cm.task_index("liver_segments", r));
    } else if (task == "liver_vessels") {
        // TS: task_id 8, resample None (nnU-Net resamples internally), crop ["liver"]
        s.labels = cm.task.at("liver_vessels");
        s.spacing = {0.0, 0.0, 0.0};  // no TS-side resampling
        s.step_size = 0.5;
        s.crop_cfg = model(298);
        s.crop_labels.push_back(cm.task_index("total", "liver"));
        TaskModel m;
        m.task_id = 8;
        m.cfg = model(8);
        m.to_task.resize(m.cfg->num_classes);
        for (int i = 0; i < m.cfg->num_classes; ++i) m.to_task[i] = i;
        s.models.push_back(m);
        for (auto& r : roi_subset) s.keep.push_back(cm.task_index("liver_vessels", r));
    } else if (task == "trunk_cavities" || task == "body" || task == "body_fast") {
        // V2 (2026-10-06). TS 2.18 map_tasks_config.py: trunk_cavities = task 343, resample 1.5 mm, no crop;
        // body = task 299 at 1.5 mm, body --fast = task 300 at 6 mm, no crop. step_size 0.5 (0.8 only for total).
        // The body tasks get TS's body post-processing (nnunet.py, task_name == "body") in finish_parts().
        const int tid = task == "trunk_cavities" ? 343 : task == "body" ? 299 : 300;
        TaskModel m;
        m.task_id = tid;
        m.cfg = model(tid);
        s.labels = m.cfg->labels;
        if ((int)s.labels.size() < m.cfg->num_classes) s.labels.resize(m.cfg->num_classes);
        const double sp = tid == 300 ? 6.0 : 1.5;
        s.spacing = {sp, sp, sp};
        s.step_size = 0.5;
        m.to_task.resize(m.cfg->num_classes);
        for (int i = 0; i < m.cfg->num_classes; ++i) m.to_task[i] = i;
        s.models.push_back(m);
        if (task != "trunk_cavities") s.postprocess = "body";
        for (auto& r : roi_subset) {
            const int li = m.cfg->label_index(r);
            if (li < 0) throw std::runtime_error("ROI not in " + task + ": " + r);
            s.keep.push_back(li);
        }
    } else if (task.rfind("custom:", 0) == 0) {
        // fine-tuned model: "custom:<model id>[:<crop roi>]", no TS-side resampling (nnU-Net
        // resamples internally), optional crop to a total ROI via the 6 mm model
        std::stringstream ts(task.substr(7));
        std::string id_s, crop;
        std::getline(ts, id_s, ':');
        std::getline(ts, crop);
        const int tid = std::atoi(id_s.c_str());
        TaskModel m;
        m.task_id = tid;
        m.cfg = model(tid);
        s.labels = m.cfg->labels;
        if ((int)s.labels.size() < m.cfg->num_classes) s.labels.resize(m.cfg->num_classes);
        s.spacing = {0.0, 0.0, 0.0};
        s.step_size = 0.5;
        m.to_task.resize(m.cfg->num_classes);
        for (int i = 0; i < m.cfg->num_classes; ++i) m.to_task[i] = i;
        s.models.push_back(m);
        if (!crop.empty() && crop != "-") {
            s.crop_cfg = model(298);
            std::stringstream cs(crop);
            std::string r;
            while (std::getline(cs, r, '+')) s.crop_labels.push_back(cm.task_index("total", r));
        }
        for (auto& r : roi_subset) {
            const int li = m.cfg->label_index(r);
            if (li < 0) throw std::runtime_error("ROI not in custom model: " + r);
            s.keep.push_back(li);
        }
    } else {
        throw std::runtime_error("unsupported task: " + task);
    }
    if (s.models.empty()) throw std::runtime_error("no model contains the requested roi_subset");
    s.crop_addon_mm = 20.0;  // TS: crop_addon = [20,20,20] when the default crop model is used
    return s;
}

// ---------------------------------------------------------------- TaskRunner
TaskRunner::TaskRunner(const Volume<float>& ct, const TaskSpec& spec, ProgressFn progress_fn, PipelineDebug* dbg)
    : ct_(ct), spec_(spec), progress_(std::move(progress_fn)), dbg_(dbg) {
    tm_ = now_s();
    if (spec_.crop_cfg) {
        // ---- rough segmentation at 6 mm (crop model). TS 2.18 runs it as task_name="total" -> step 0.8
        //      (TS <= 2.13 used 0.5 here; with the 64-voxel patch the tiling is identical up to 96 voxels per axis at 6 mm)
        progress("resample_6mm", 0);
        const double s6 = spec_.crop_cfg->spacing[0];
        Volume<int32_t> in6 = resample_to_spacing(ct_, {s6, s6, s6}, spec_.resampling_order);
        mark("resample 6mm");
        if (dbg_) {
            dbg_->crop_pred.keep = dbg_->keep;
            if (dbg_->keep) dbg_->crop_in = in6;
        }
        sw_ = std::make_unique<SlidingWindow>(*spec_.crop_cfg, in6, 0.8, dbg_ ? &dbg_->crop_pred : nullptr);
        mark("crop: preprocess");
        in_crop_stage_ = true;
        cur_model_id_ = spec_.crop_cfg->task_id;
        stage_name_ = "crop_model";
    } else {
        for (int d = 0; d < 3; ++d) { b0_[d] = 0; b1_[d] = ct_.shape[d]; }
        progress("resample_part", 0);
        start_parts(resample_to_spacing(ct_, spec_.spacing, spec_.resampling_order), ct_.shape, ct_.affine);
    }
}

TaskRunner::~TaskRunner() = default;

void TaskRunner::mark(const char* what) {
    const double t = now_s();
    char buf[96];
    std::snprintf(buf, sizeof buf, "  %-28s %7.0f ms\n", what, (t - tm_) * 1000);
    timings_ += buf;
    tm_ = t;
}

void TaskRunner::progress(const std::string& s, double f) const {
    if (progress_) progress_(s, f);
}

// resampled: the (cropped) CT on the task grid (int32), back_*: grid to return to
void TaskRunner::start_parts(Volume<int32_t> resampled, const Shape3& back_shape, const Affine& back_affine) {
    back_shape_ = back_shape;
    back_affine_ = back_affine;
    in_p_ = std::move(resampled);
    mark("resample task spacing");
    if (dbg_ && dbg_->keep) dbg_->part_in = in_p_;
    combined_ = Volume<uint8_t>(in_p_.shape, in_p_.affine);
    if (dbg_) dbg_->part_pred.resize(spec_.models.size());
    start_part(0);
}

void TaskRunner::start_part(size_t i) {
    part_ = i;
    const TaskModel& m = spec_.models[i];
    if (dbg_) dbg_->part_pred[i].keep = dbg_->keep;
    sw_ = std::make_unique<SlidingWindow>(*m.cfg, in_p_, spec_.step_size, dbg_ ? &dbg_->part_pred[i] : nullptr);
    mark("part: preprocess");
    cur_model_id_ = m.task_id;
    stage_name_ = "part_model";
    if (spec_.models.size() > 1) stage_name_ += " " + std::to_string(i + 1) + "/" + std::to_string(spec_.models.size());
}

void TaskRunner::advance() {
    if (!sw_) return;
    if (in_crop_stage_) {
        mark("crop: tiles (net+accum)");
        Volume<uint8_t> seg6 = sw_->finish();
        sw_.reset();
        in_crop_stage_ = false;
        progress("crop_model", 1);
        mark("crop: finish/argmax");
        Volume<uint8_t> seg6_full = resample_labels_to_shape(seg6, ct_.shape, ct_.affine);
        mark("crop: labels -> full res");
        std::vector<uint8_t> is_roi(256, 0);
        for (int l : spec_.crop_labels) is_roi.at(l) = 1;
        // ---- crop_to_mask with addon (mm -> voxels, truncated)
        int64_t lo[3], hi[3];
        bbox_where(seg6_full.data.data(), seg6_full.shape, [&](uint8_t v) { return is_roi[v] != 0; }, lo, hi);
        if (dbg_ && dbg_->keep) {
            dbg_->crop_seg = seg6;
            dbg_->crop_mask = Volume<uint8_t>(ct_.shape, ct_.affine);
            for (size_t i = 0; i < seg6_full.data.size(); ++i) dbg_->crop_mask.data[i] = is_roi[seg6_full.data[i]];
        }
        result_ = Volume<uint8_t>(ct_.shape, ct_.affine);
        if (hi[0] < 0) {  // "Crop is empty. Returning empty segmentation."
            progress("done", 1);
            return;
        }
        const auto zooms0 = zooms_from_affine(ct_.affine);
        for (int d = 0; d < 3; ++d) {
            const int64_t addon = (int64_t)(spec_.crop_addon_mm / zooms0[d]);  // (np.array(addon) / zooms).astype(int)
            b0_[d] = std::max<int64_t>(0, lo[d] - addon);
            b1_[d] = std::min<int64_t>(ct_.shape[d], hi[d] + 1 + addon);
            if (dbg_) { dbg_->bbox[d][0] = (int)b0_[d]; dbg_->bbox[d][1] = (int)b1_[d]; }
        }
        Volume<int32_t> crop(Shape3{b1_[0] - b0_[0], b1_[1] - b0_[1], b1_[2] - b0_[2]}, ct_.affine);
        for (int r = 0; r < 3; ++r)
            crop.affine[r * 4 + 3] = ct_.affine[r * 4 + 0] * b0_[0] + ct_.affine[r * 4 + 1] * b0_[1] + ct_.affine[r * 4 + 2] * b0_[2] +
                                     ct_.affine[r * 4 + 3];
        parallel_for(crop.shape[0], [&](int64_t b, int64_t e, int) {
            for (int64_t i = b; i < e; ++i)
                for (int64_t j = 0; j < crop.shape[1]; ++j)
                    for (int64_t k = 0; k < crop.shape[2]; ++k) crop(i, j, k) = (int32_t)(double)ct_(i + b0_[0], j + b0_[1], k + b0_[2]);
        });
        mark("bbox + crop");
        progress("resample_part", 0);
        start_parts(resample_to_spacing(crop, spec_.spacing, spec_.resampling_order), crop.shape, crop.affine);
        return;
    }
    // ---- a part model finished: merge its labels (TS: seg_combined[seg == jdx] = class index)
    mark("part: tiles (net+accum)");
    Volume<uint8_t> seg = sw_->finish();
    sw_.reset();
    const auto& map = spec_.models[part_].to_task;
    parallel_for((int64_t)seg.data.size(), [&](int64_t b, int64_t e, int) {
        for (int64_t i = b; i < e; ++i) {
            const int t = map[seg.data[(size_t)i]];
            if (t >= 0) combined_.data[(size_t)i] = (uint8_t)t;
        }
    }, 1 << 16);
    if (dbg_ && dbg_->keep) dbg_->part_seg.push_back(std::move(seg));
    mark("part: finish/argmax+merge");
    progress(stage_name_, 1);
    if (part_ + 1 < spec_.models.size()) {
        start_part(part_ + 1);
        return;
    }
    finish_parts();
}

// scipy.ndimage.label with its default structure (6-connectivity). Components are numbered 1.. in the order of
// their first voxel in the scan order of `v` (C order, last axis fastest), like ndimage on an array of that order.
// Returns the number of components; sizes[c] = voxel count of component c (sizes[0] unused).
static int64_t label6(const Volume<uint8_t>& v, uint8_t value, std::vector<int32_t>& lab, std::vector<int64_t>& sizes) {
    const Shape3 sh = v.shape;
    const size_t n = v.data.size();
    lab.assign(n, 0);
    std::vector<int32_t> parent(1, 0);
    auto find = [&](int32_t x) { while (parent[(size_t)x] != x) x = parent[(size_t)x] = parent[(size_t)parent[(size_t)x]]; return x; };
    auto unite = [&](int32_t a, int32_t b) {  // keep the smaller (earlier) root so numbering follows first appearance
        a = find(a); b = find(b);
        if (a == b) return;
        if (a < b) parent[(size_t)b] = a; else parent[(size_t)a] = b;
    };
    for (int64_t i = 0; i < sh[0]; ++i)
        for (int64_t j = 0; j < sh[1]; ++j)
            for (int64_t k = 0; k < sh[2]; ++k) {
                const size_t p = (size_t)v.idx(i, j, k);
                if (v.data[p] != value) continue;
                int32_t l = 0;
                const int32_t nb[3] = {k > 0 ? lab[p - 1] : 0, j > 0 ? lab[(size_t)v.idx(i, j - 1, k)] : 0, i > 0 ? lab[(size_t)v.idx(i - 1, j, k)] : 0};
                for (int32_t q : nb)
                    if (q) { if (!l) l = q; else unite(l, q); }
                if (!l) { l = (int32_t)parent.size(); parent.push_back(l); }
                lab[p] = l;
            }
    // flatten and renumber the roots in order of first appearance
    std::vector<int32_t> newid(parent.size(), 0);
    int32_t next = 0;
    sizes.assign(1, 0);
    for (size_t p = 0; p < n; ++p) {
        if (!lab[p]) continue;
        const int32_t r = find(lab[p]);
        if (!newid[(size_t)r]) { newid[(size_t)r] = ++next; sizes.push_back(0); }
        lab[p] = newid[(size_t)r];
        ++sizes[(size_t)lab[p]];
    }
    return next;
}

// TS nnunet.py (task_name == "body"), on the prediction at task resolution before it goes back to the CT grid:
//   keep_largest_blob_multilabel(..., ["body_trunc"])   -> only the largest body_trunc component stays (ties: first)
//   remove_small_blobs_multilabel(..., ["body_extremities"], interval=[50000 / voxel volume, 1e10])
//     -> body_extremities components with count <= 50000 mm3 / voxel volume (or > 1e10) are removed
// Removed voxels become background (0).
static void body_postprocess(Volume<uint8_t>& seg, const TaskSpec& spec, const ModelConfig& cfg) {
    std::vector<int32_t> lab;
    std::vector<int64_t> sizes;
    const int trunc = cfg.label_index("body_trunc"), ext = cfg.label_index("body_extremities");
    if (trunc < 0 || ext < 0) throw std::runtime_error("body model without body_trunc / body_extremities labels");
    if (label6(seg, (uint8_t)trunc, lab, sizes) > 0) {
        size_t best = 1;
        for (size_t c = 2; c < sizes.size(); ++c)
            if (sizes[c] > sizes[best]) best = c;  // np.argmax: first maximum
        for (size_t p = 0; p < seg.data.size(); ++p)
            if (lab[p] && (size_t)lab[p] != best) seg.data[p] = 0;
    }
    const double vox_vol = spec.spacing[0] * spec.spacing[1] * spec.spacing[2];
    const double lo = 50000.0 / vox_vol, hi = 1e10;
    if (label6(seg, (uint8_t)ext, lab, sizes) > 0) {
        for (size_t p = 0; p < seg.data.size(); ++p)
            if (lab[p] && ((double)sizes[(size_t)lab[p]] <= lo || (double)sizes[(size_t)lab[p]] > hi)) seg.data[p] = 0;
    }
}

void TaskRunner::finish_parts() {
    if (spec_.postprocess == "body") {
        body_postprocess(combined_, spec_, *spec_.models.at(0).cfg);
        mark("body post-processing");
    }
    if (dbg_ && dbg_->keep) dbg_->combined = combined_;
    // ---- back to the (cropped) CT grid (order 0), un-crop, keep roi_subset labels
    Volume<uint8_t> back = resample_labels_to_shape(combined_, back_shape_, back_affine_);
    combined_ = Volume<uint8_t>();
    in_p_ = Volume<int32_t>();
    std::vector<uint8_t> keep(256, spec_.keep.empty() ? 1 : 0);
    for (int l : spec_.keep) keep.at(l) = 1;
    keep[0] = 1;
    if (result_.shape != ct_.shape) result_ = Volume<uint8_t>(ct_.shape, ct_.affine);
    parallel_for(back_shape_[0], [&](int64_t b, int64_t e, int) {
        for (int64_t i = b; i < e; ++i)
            for (int64_t j = 0; j < back_shape_[1]; ++j)
                for (int64_t k = 0; k < back_shape_[2]; ++k) {
                    const uint8_t l = back(i, j, k);
                    result_(i + b0_[0], j + b0_[1], k + b0_[2]) = keep[l] ? l : 0;
                }
    });
    mark("back to orig + uncrop");
    progress("done", 1);
}


std::unique_ptr<Runner> make_runner(const Volume<float>& ct, const std::string& task, const std::vector<std::string>& roi_subset,
                                    const std::map<int, const ModelConfig*>& models, const ClassMaps& cm, ProgressFn progress,
                                    PipelineDebug* dbg) {
    return std::make_unique<TaskRunner>(ct, make_task(task, roi_subset, models, cm), progress, dbg);
}

std::vector<int> task_model_ids(const std::string& task, const std::vector<std::string>& roi_subset, const ClassMaps& cm) {
    std::map<int, ModelConfig> dummy;
    std::map<int, const ModelConfig*> m;
    std::vector<int> ids0 = {298, 291, 292, 293, 294, 295, 570, 8, 343, 299, 300};  // V2: trunk_cavities, body, body_fast
    if (task.rfind("custom:", 0) == 0) ids0.push_back(std::atoi(task.substr(7).c_str()));
    for (int tid : ids0) {
        dummy[tid].task_id = tid;
        dummy[tid].num_classes = 256;
        m[tid] = &dummy[tid];
    }
    std::vector<TaskSpec> specs;
    if (specs.empty()) {
        // (a custom model's roi_subset only selects labels; the placeholder configs have none to check)
        const bool custom = task.rfind("custom:", 0) == 0;
        specs.push_back(make_task(task, custom ? std::vector<std::string>{} : roi_subset, m, cm));
    }
    std::vector<int> ids;
    auto add = [&](int t) {
        if (std::find(ids.begin(), ids.end(), t) == ids.end()) ids.push_back(t);
    };
    for (auto& s : specs) {
        if (s.crop_cfg) add(298);
        for (auto& md : s.models) add(md.task_id);
    }
    return ids;
}

Volume<uint8_t> run_runner(Runner& r, const std::map<int, const UNet*>& nets, ProgressFn progress) {
    while (!r.done()) {
        const UNet& net = *nets.at(r.model_id());
        SlidingWindow& sw = *r.window();
        const int* P = net.config().patch;
        for (int t = 0; t < sw.num_tiles(); ++t) {
            if (progress) progress(r.stage_name(), (double)t / sw.num_tiles());
            Tensor x(1, P[0], P[1], P[2]);
            sw.tile_input(t, x.data.data());
            Tensor y = net.forward(x);
            sw.accumulate(t, y.data.data());
        }
        r.advance();
    }
    return r.take_result();
}

Volume<uint8_t> run_task(const Volume<float>& ct, const TaskSpec& spec, const std::map<int, const UNet*>& nets,
                         ProgressFn progress, PipelineDebug* dbg) {
    TaskRunner r(ct, spec, progress, dbg);
    while (!r.done()) {
        const UNet& net = *nets.at(r.model_id());
        SlidingWindow& sw = *r.window();
        const int* P = net.config().patch;
        for (int t = 0; t < sw.num_tiles(); ++t) {
            if (progress) progress(r.stage_name(), (double)t / sw.num_tiles());
            Tensor x(1, P[0], P[1], P[2]);
            sw.tile_input(t, x.data.data());
            Tensor y = net.forward(x);
            sw.accumulate(t, y.data.data());
        }
        r.advance();
    }
    if (dbg) dbg->timings = r.timings();
    return r.take_result();
}

Volume<uint8_t> segment_roi(const Volume<float>& ct, const UNet& crop_model, const UNet& part_model,
                            const SegmentOptions& opt, PipelineDebug* dbg) {
    // total + roi_subset with the organs model; its labels coincide with the total class indices
    TaskSpec s;
    s.name = "total";
    s.resampling_order = g_resampling_order;
    s.crop_cfg = &crop_model.config();
    s.spacing = {part_model.config().spacing[0], part_model.config().spacing[0], part_model.config().spacing[0]};
    s.step_size = part_model.config().spacing[0] < 3.0 ? 0.8 : 0.5;
    TaskModel m;
    m.task_id = part_model.config().task_id;
    m.cfg = &part_model.config();
    m.to_task.assign(m.cfg->num_classes, -1);
    for (int i = 1; i < m.cfg->num_classes; ++i) m.to_task[i] = i;
    s.models.push_back(m);
    for (auto& r : opt.roi_subset) {
        s.crop_labels.push_back(crop_model.config().label_index(r));
        s.keep.push_back(part_model.config().label_index(r));
    }
    return run_task(ct, s, {{crop_model.config().task_id, &crop_model}, {part_model.config().task_id, &part_model}}, opt.progress,
                    dbg);
}

}  // namespace tsc
