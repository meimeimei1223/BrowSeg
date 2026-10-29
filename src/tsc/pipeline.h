// TotalSegmentator tasks re-implemented step by step (python_api.py + nnunet.py, TS 2.13).
//
// A task is described by a TaskSpec, mirroring TS's per-task settings:
//   * optional crop: rough 6 mm segmentation with the total_6mm model (298) on the
//     whole CT, crop box = union of the crop ROIs + addon (crop_to_mask)
//   * resampling of the (cropped) CT to the task spacing (order 3 spline, int32)
//   * one or more part models (nnU-Net sliding window) on that grid; with several
//     models the labels are merged into the task class map (later models win)
//   * nearest-neighbour back to the CT grid, un-crop, optional roi_subset filter
// Everything runs in closest-canonical RAS space, which is equivalent to TS's
// original-grid bookkeeping (flips/permutations commute with every step).
//
// Supported tasks: "total" (all 5 part models, or only those needed for a
// roi_subset, which also enables the 6 mm crop) and "liver_segments".
#pragma once

#include <map>
#include <memory>

#include "nnunet.h"

namespace tsc {

// TS class maps (weights/classmaps.txt, written by tools/export_classmaps.py)
struct ClassMaps {
    std::map<std::string, std::vector<std::string>> task;  // task -> label index -> class name
    std::map<int, std::map<int, std::string>> part;        // part model task id -> model label -> class name
    int task_index(const std::string& task_name, const std::string& cls) const;
};
ClassMaps load_classmaps(const std::string& text);
ClassMaps load_classmaps_file(const std::string& path);

struct TaskModel {
    int task_id = 0;
    const ModelConfig* cfg = nullptr;
    // model label -> task label; -1 = "not assigned" (keeps what earlier models wrote)
    std::vector<int> to_task;
};

struct TaskSpec {
    std::string name;
    const ModelConfig* crop_cfg = nullptr;  // set when the task crops with the 6 mm model
    std::vector<int> crop_labels;           // labels of the crop model that form the crop mask
    double crop_addon_mm = 20.0;
    std::array<double, 3> spacing{1.5, 1.5, 1.5};  // resample target (x, y, z) in mm
    double step_size = 0.5;
    int resampling_order = 1;               // spline order of the input resampling (TS >= 2.16: 1, TS <= 2.15: 3)
    std::vector<TaskModel> models;
    std::vector<int> keep;                  // task labels kept (roi_subset); empty = all
    std::vector<std::string> labels;        // task label index -> class name
    std::string postprocess;                // "body": TS nnunet.py's body post-processing on the task grid
};

// Input resampling order used by make_task (default 1 = TotalSegmentator >= 2.16; 3 reproduces TS <= 2.15).
void set_resampling_order(int order);
int resampling_order();

// Builds the TS configuration of `task` ("total" or "liver_segments").
// models: task id -> model configuration (all models the task may need).
TaskSpec make_task(const std::string& task, const std::vector<std::string>& roi_subset,
                   const std::map<int, const ModelConfig*>& models, const ClassMaps& cm);

struct PipelineDebug {
    bool keep = false;
    Volume<int32_t> crop_in;        // crop model input (6mm, int32)
    Volume<uint8_t> crop_seg;       // crop model output (6mm)
    Volume<uint8_t> crop_mask;      // crop mask at original resolution
    int bbox[3][2] = {};            // crop box in canonical voxel indices
    Volume<int32_t> part_in;        // part model input (task spacing, int32)
    std::vector<Volume<uint8_t>> part_seg;  // each part model's argmax
    Volume<uint8_t> combined;       // merged labels on the task grid
    PredictDebug crop_pred;
    std::vector<PredictDebug> part_pred;
    std::string timings;
};

struct SegmentOptions {
    std::vector<std::string> roi_subset{"liver"};
    ProgressFn progress;
};

// Step-wise execution so the networks can run outside C++ (WebGPU):
//   while (!r.done()) { for each tile t of *r.window(): tile_input -> net(r.model_id()) -> accumulate; r.advance(); }
class Runner {
public:
    virtual ~Runner() = default;
    virtual bool done() const = 0;
    virtual int model_id() const = 0;          // task id of the model the current window needs
    virtual const char* stage_name() const = 0;
    virtual SlidingWindow* window() = 0;
    virtual void advance() = 0;                // call after all tiles were accumulated
    virtual Volume<uint8_t> take_result() = 0;
    virtual const std::string& timings() const = 0;
    virtual const std::vector<std::string>& labels() const = 0;  // output label index -> name
};

class TaskRunner : public Runner {
public:
    TaskRunner(const Volume<float>& ct, const TaskSpec& spec, ProgressFn progress = nullptr, PipelineDebug* dbg = nullptr);
    ~TaskRunner() override;
    bool done() const override { return !sw_; }
    int model_id() const override { return cur_model_id_; }
    const char* stage_name() const override { return stage_name_.c_str(); }
    SlidingWindow* window() override { return sw_.get(); }
    void advance() override;
    Volume<uint8_t> take_result() override { return std::move(result_); }
    const std::string& timings() const override { return timings_; }
    const std::vector<std::string>& labels() const override { return spec_.labels; }

private:
    void start_parts(Volume<int32_t> resampled, const Shape3& back_shape, const Affine& back_affine);
    void start_part(size_t i);
    void finish_parts();
    void mark(const char* what);
    void progress(const std::string& s, double f) const;

    const Volume<float>& ct_;
    TaskSpec spec_;
    ProgressFn progress_;
    PipelineDebug* dbg_;
    std::unique_ptr<SlidingWindow> sw_;
    int cur_model_id_ = 0;
    std::string stage_name_;
    bool in_crop_stage_ = false;
    size_t part_ = 0;
    int64_t b0_[3] = {0, 0, 0}, b1_[3] = {0, 0, 0};
    Shape3 back_shape_{0, 0, 0};
    Affine back_affine_{};
    Volume<int32_t> in_p_;
    Volume<uint8_t> combined_, result_;
    std::string timings_;
    double tm_ = 0;
};

// Tasks: "total", "liver_segments", "liver_vessels" (TotalSegmentator).
std::unique_ptr<Runner> make_runner(const Volume<float>& ct, const std::string& task, const std::vector<std::string>& roi_subset,
                                    const std::map<int, const ModelConfig*>& models, const ClassMaps& cm,
                                    ProgressFn progress = nullptr, PipelineDebug* dbg = nullptr);
// task ids of every model a task needs (crop model included)
std::vector<int> task_model_ids(const std::string& task, const std::vector<std::string>& roi_subset, const ClassMaps& cm);

// Runs any runner with C++ networks. nets: task id -> network.
Volume<uint8_t> run_runner(Runner& r, const std::map<int, const UNet*>& nets, ProgressFn progress = nullptr);

// Convenience: runs a task with C++ networks. nets: task id -> network.
Volume<uint8_t> run_task(const Volume<float>& ct, const TaskSpec& spec, const std::map<int, const UNet*>& nets,
                         ProgressFn progress = nullptr, PipelineDebug* dbg = nullptr);

// Backward-compatible liver entry point: task "total" with roi_subset (crop + organ model).
Volume<uint8_t> segment_roi(const Volume<float>& ct, const UNet& crop_model, const UNet& part_model,
                            const SegmentOptions& opt, PipelineDebug* dbg = nullptr);

}  // namespace tsc
