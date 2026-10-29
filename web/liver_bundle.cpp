// Liver bundle: see liver_bundle.h. The helper functions below are LiverModelExporter's (src/exporter.cpp), unchanged
// except for the removal of threads and files, so that the masks and meshes are the same.
#include "liver_bundle.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>

namespace browseg {

using tsc::Mesh;
using tsc::MeshOptions;
using tsc::Shape3;
using tsc::Volume;

int TaskLabels::label(const std::string& name) const {
    for (size_t i = 0; i < labels.size(); ++i)
        if (labels[i] == name) return (int)i;
    return -1;
}
int TaskLabels::label_like(const std::string& part) const {
    for (size_t i = 1; i < labels.size(); ++i)
        if (labels[i].find(part) != std::string::npos) return (int)i;
    return -1;
}

namespace {

std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

// The box (a part of the CT grid) in which the masks are made
struct Box {
    int64_t lo[3], hi[3];  // CT grid indices, inclusive
    Shape3 shape() const { return {hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1}; }
};

Volume<uint8_t> empty_box(const Box& b, const tsc::Affine& a) { return Volume<uint8_t>(b.shape(), a); }

// dst (box) |= (task label map == label)
void or_label(Volume<uint8_t>& dst, const Box& b, const TaskLabels& t, int label) {
    if (label < 0) return;
    const Shape3 ts = t.seg.shape;
    int64_t lo[3], hi[3];  // overlap of the box and the task's crop, CT grid indices
    for (int d = 0; d < 3; ++d) {
        lo[d] = std::max(b.lo[d], t.origin[d]);
        hi[d] = std::min(b.hi[d], t.origin[d] + ts[d] - 1);
        if (lo[d] > hi[d]) return;
    }
    for (int64_t i = lo[0]; i <= hi[0]; ++i)
        for (int64_t j = lo[1]; j <= hi[1]; ++j)
            for (int64_t k = lo[2]; k <= hi[2]; ++k)
                if (t.seg(i - t.origin[0], j - t.origin[1], k - t.origin[2]) == label) dst(i - b.lo[0], j - b.lo[1], k - b.lo[2]) = 1;
}

void or_mask(Volume<uint8_t>& dst, const Volume<uint8_t>& add) {
    for (size_t i = 0; i < dst.data.size(); ++i)
        if (add.data[i]) dst.data[i] = 1;
}

// the grid axis that runs cranio-caudally (largest world z component of its direction)
int superior_axis(const tsc::Affine& a) {
    int s = 0;
    for (int d = 1; d < 3; ++d)
        if (std::fabs(a[8 + d]) > std::fabs(a[8 + s])) s = d;
    return s;
}

void and_mask(Volume<uint8_t>& dst, const Volume<uint8_t>& keep) {
    for (size_t i = 0; i < dst.data.size(); ++i)
        if (!keep.data[i]) dst.data[i] = 0;
}

int64_t count(const Volume<uint8_t>& m) {
    int64_t n = 0;
    for (auto v : m.data) n += v != 0;
    return n;
}

double voxel_ml(const Volume<uint8_t>& m) {
    auto z = tsc::zooms_from_affine(m.affine);
    return z[0] * z[1] * z[2] / 1000.0;
}

// 1-D squared Euclidean distance transform (Felzenszwalb & Huttenlocher), samples h mm apart. The samples are at
// positions o .. o + n - 1 of the whole line (o > 0 for a line of a box), so that the arithmetic is the same as on the
// whole line.
void edt1d(const double* f, double* d, int64_t n, int64_t o, double h, std::vector<int64_t>& v, std::vector<double>& z) {
    v.resize((size_t)n);
    z.resize((size_t)n + 1);
    const double h2 = h * h;
    int64_t k = 0;
    v[0] = 0;
    z[0] = -1e300;
    z[1] = 1e300;
    for (int64_t q = 1; q < n; ++q) {
        double s;
        for (;;) {
            const int64_t p = v[(size_t)k];
            const int64_t Q = q + o, P = p + o;
            s = ((f[q] + (double)Q * Q * h2) - (f[p] + (double)P * P * h2)) / (2.0 * h2 * (double)(q - p));
            if (s <= z[(size_t)k] && k > 0) { --k; continue; }
            break;
        }
        ++k;
        v[(size_t)k] = q;
        z[(size_t)k] = s;
        z[(size_t)k + 1] = 1e300;
    }
    k = 0;
    for (int64_t q = 0; q < n; ++q) {
        while (z[(size_t)k + 1] < (double)(q + o)) ++k;
        const int64_t p = v[(size_t)k];
        d[q] = h2 * (double)(q - p) * (double)(q - p) + f[p];
    }
}

// Grows a 0/1 mask by `mm` (exact Euclidean distance on the anisotropic grid). m is the box `box` of the CT grid; it
// must hold the grown mask (then the result is the same as on the whole grid).
void dilate_mm(Volume<uint8_t>& m, const Box& box, double mm) {
    if (mm <= 0) return;
    const Shape3 s = m.shape;
    const auto zoom = tsc::zooms_from_affine(m.affine);
    std::vector<float> d(m.data.size());
    for (size_t i = 0; i < d.size(); ++i) d[i] = m.data[i] ? 0.0f : 1e20f;
    for (int axis = 0; axis < 3; ++axis) {
        const int64_t n = s[axis];
        const int64_t stride = axis == 2 ? 1 : axis == 1 ? s[2] : s[1] * s[2];
        const int a1 = axis == 0 ? 1 : 0, a2 = axis == 2 ? 1 : 2;  // the two other axes
        const int64_t lines = s[a1] * s[a2];
        tsc::parallel_for(lines, [&](int64_t b, int64_t e, int) {
            std::vector<double> f((size_t)n), out((size_t)n), z;
            std::vector<int64_t> v;
            for (int64_t l = b; l < e; ++l) {
                int64_t idx[3];
                idx[a1] = l / s[a2];
                idx[a2] = l % s[a2];
                idx[axis] = 0;
                const int64_t base = (idx[0] * s[1] + idx[1]) * s[2] + idx[2];
                for (int64_t q = 0; q < n; ++q) f[(size_t)q] = d[(size_t)(base + q * stride)];
                edt1d(f.data(), out.data(), n, box.lo[axis], zoom[axis], v, z);
                for (int64_t q = 0; q < n; ++q) d[(size_t)(base + q * stride)] = (float)out[(size_t)q];
            }
        });
    }
    const float r2 = (float)(mm * mm);
    for (size_t i = 0; i < d.size(); ++i) m.data[i] = d[i] <= r2 ? 1 : 0;
}

// 26-connected pieces of a 0/1 mask: voxels, centroid (world mm, RAS), largest first
struct Piece {
    std::vector<int64_t> vox;
    double c[3] = {0, 0, 0};
};
std::vector<Piece> pieces(const Volume<uint8_t>& m, const Box& b) {
    const Shape3 s = m.shape;
    std::vector<uint8_t> seen(m.data.size(), 0);
    std::vector<Piece> out;
    std::vector<int64_t> stack;
    for (int64_t start = 0; start < (int64_t)m.data.size(); ++start) {
        if (!m.data[(size_t)start] || seen[(size_t)start]) continue;
        Piece P;
        double si = 0, sj = 0, sk = 0;
        seen[(size_t)start] = 1;
        stack.assign(1, start);
        while (!stack.empty()) {
            const int64_t v = stack.back();
            stack.pop_back();
            const int64_t i = v / (s[1] * s[2]), j = (v / s[2]) % s[1], k = v % s[2];
            P.vox.push_back(v);
            si += (double)(i + b.lo[0]); sj += (double)(j + b.lo[1]); sk += (double)(k + b.lo[2]);
            for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj)
                    for (int dk = -1; dk <= 1; ++dk) {
                        const int64_t a = i + di, b = j + dj, c = k + dk;
                        if (a < 0 || b < 0 || c < 0 || a >= s[0] || b >= s[1] || c >= s[2]) continue;
                        const int64_t w = (a * s[1] + b) * s[2] + c;
                        if (m.data[(size_t)w] && !seen[(size_t)w]) {
                            seen[(size_t)w] = 1;
                            stack.push_back(w);
                        }
                    }
        }
        const double n = (double)P.vox.size();
        const double ijk[3] = {si / n, sj / n, sk / n};
        for (int r = 0; r < 3; ++r)
            P.c[r] = m.affine[r * 4] * ijk[0] + m.affine[r * 4 + 1] * ijk[1] + m.affine[r * 4 + 2] * ijk[2] + m.affine[r * 4 + 3];
        out.push_back(std::move(P));
    }
    std::stable_sort(out.begin(), out.end(), [](const Piece& x, const Piece& y) { return x.vox.size() > y.vox.size(); });
    return out;
}

// tsc_core meshes are RAS; lps: x, y negated (a rotation, so the winding is unchanged)
void to_frame(Mesh& m, bool lps) {
    if (lps)
        for (size_t v = 0; v < m.num_vertices(); ++v) {
            m.vertices[v * 3] = -m.vertices[v * 3];
            m.vertices[v * 3 + 1] = -m.vertices[v * 3 + 1];
        }
}

// v x y z / f a b c only (what the simulator's MeshDataTypes::loadOBJFile reads); same format as the exporter
std::string obj_text(const Mesh& m, const std::string& comment, const std::string& version) {
    std::string s = "# " + comment + "\n# " + version + "\n";
    s.reserve(s.size() + m.num_vertices() * 32 + m.num_triangles() * 24);
    char buf[96];
    for (size_t v = 0; v < m.num_vertices(); ++v) {
        std::snprintf(buf, sizeof buf, "v %.3f %.3f %.3f\n", m.vertices[v * 3], m.vertices[v * 3 + 1], m.vertices[v * 3 + 2]);
        s += buf;
    }
    for (size_t t = 0; t < m.indices.size(); t += 3) {
        std::snprintf(buf, sizeof buf, "f %u %u %u\n", m.indices[t] + 1, m.indices[t + 1] + 1, m.indices[t + 2] + 1);
        s += buf;
    }
    return s;
}

// A small cube (edge mm) just outside the liver's bounding box: stands in for an organ that has no voxels,
// because the simulator loads a fixed list of files and stops on a missing one.
Mesh placeholder_cube(const Mesh& liver, float mm) {
    float lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    for (size_t v = 0; v < liver.num_vertices(); ++v)
        for (int k = 0; k < 3; ++k) {
            const float x = liver.vertices[v * 3 + k];
            if (v == 0 || x < lo[k]) lo[k] = x;
            if (v == 0 || x > hi[k]) hi[k] = x;
        }
    const float x0 = hi[0] + 5.0f, y0 = lo[1], z0 = lo[2];
    Mesh m;
    for (int i = 0; i < 8; ++i) {
        m.vertices.push_back(x0 + ((i & 1) ? mm : 0));
        m.vertices.push_back(y0 + ((i & 2) ? mm : 0));
        m.vertices.push_back(z0 + ((i & 4) ? mm : 0));
    }
    const uint32_t f[12][3] = {{0, 2, 1}, {1, 2, 3}, {4, 5, 6}, {5, 7, 6}, {0, 1, 4}, {1, 5, 4},
                               {2, 6, 3}, {3, 6, 7}, {0, 4, 2}, {2, 4, 6}, {1, 3, 5}, {3, 7, 5}};
    for (auto& t : f) m.indices.insert(m.indices.end(), {t[0], t[1], t[2]});
    return m;
}

const TaskLabels& need(const std::map<std::string, TaskLabels>& tasks, const std::string& name) {
    auto it = tasks.find(name);
    if (it == tasks.end()) throw std::runtime_error("liver bundle: task " + name + " has not been run");
    return it->second;
}

}  // namespace

TaskLabels crop_to_labels(Volume<uint8_t>&& seg, const std::vector<std::string>& labels) {
    TaskLabels t;
    t.labels = labels;
    t.full = seg.shape;
    int64_t lo[3], hi[3];
    tsc::bbox_where(seg.data.data(), seg.shape, [](uint8_t v) { return v != 0; }, lo, hi);
    if (hi[0] < 0)
        for (int d = 0; d < 3; ++d) lo[d] = hi[d] = 0;
    Volume<uint8_t> c(Shape3{hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1}, seg.affine);
    for (int64_t i = 0; i < c.shape[0]; ++i)
        for (int64_t j = 0; j < c.shape[1]; ++j)
            std::copy_n(&seg(i + lo[0], j + lo[1], lo[2]), c.shape[2], &c(i, j, 0));
    for (int d = 0; d < 3; ++d) t.origin[d] = lo[d];
    t.seg = std::move(c);
    seg = Volume<uint8_t>();
    return t;
}

BundleResult build_liver_bundle(const std::map<std::string, TaskLabels>& tasks, const BundleOptions& a) {
    const TaskLabels& T = need(tasks, "total");
    const TaskLabels& S = need(tasks, "liver_segments");
    const TaskLabels& V = need(tasks, "custom:950:liver");
    const TaskLabels& LV = need(tasks, "liver_vessels");
    const Shape3 G = T.grid();
    for (const TaskLabels* t : {&S, &V, &LV})
        if (t->grid() != G || t->seg.affine != T.seg.affine) throw std::runtime_error("liver bundle: the task label maps are on different grids");
    const tsc::Affine& A = T.seg.affine;

    // ---- the box: liver + margin (+1 voxel), gallbladder, liver_segments, liver_vessels (portal and vein are clipped to
    // liver + margin). Outside it every mask is empty, so the masks, meshes and pieces are those of the whole grid.
    Box box{{G[0], G[1], G[2]}, {-1, -1, -1}};
    auto add = [&](const TaskLabels& t, int label) {  // label < 0: every labelled voxel of t
        int64_t lo[3], hi[3];
        tsc::bbox_where(t.seg.data.data(), t.seg.shape, [label](uint8_t v) { return label < 0 ? v != 0 : v == label; }, lo, hi);
        if (hi[0] < 0) return false;
        for (int d = 0; d < 3; ++d) {
            box.lo[d] = std::min(box.lo[d], lo[d] + t.origin[d]);
            box.hi[d] = std::max(box.hi[d], hi[d] + t.origin[d]);
        }
        return true;
    };
    const int sax = superior_axis(A);
    int64_t slab_lo = 0, slab_hi = -1;  // IVC: liver's cranio-caudal extent + ivc_margin_mm, CT grid indices along sax
    if (T.label("liver") >= 0 && add(T, T.label("liver"))) {
        const int64_t g = (int64_t)std::ceil(a.ivc_margin_mm / tsc::zooms_from_affine(A)[sax]);
        slab_lo = std::max<int64_t>(0, box.lo[sax] - g);
        slab_hi = std::min<int64_t>(G[sax] - 1, box.hi[sax] + g);
        const auto zoom = tsc::zooms_from_affine(A);
        for (int d = 0; d < 3; ++d) {
            const int64_t g = (int64_t)std::ceil(a.margin_mm / zoom[d]) + 1;
            box.lo[d] = std::max<int64_t>(0, box.lo[d] - g);
            box.hi[d] = std::min<int64_t>(G[d] - 1, box.hi[d] + g);
        }
    }
    if (T.label("gallbladder") >= 0) add(T, T.label("gallbladder"));
    add(S, -1);
    add(LV, -1);
    if (const int l = T.label("inferior_vena_cava"); l >= 0 && slab_hi >= slab_lo) {  // IVC voxels inside the slab
        const Shape3 ts = T.seg.shape;
        for (int64_t i = 0; i < ts[0]; ++i)
            for (int64_t j = 0; j < ts[1]; ++j)
                for (int64_t k = 0; k < ts[2]; ++k) {
                    if (T.seg(i, j, k) != l) continue;
                    const int64_t g[3] = {i + T.origin[0], j + T.origin[1], k + T.origin[2]};
                    if (g[sax] < slab_lo || g[sax] > slab_hi) continue;
                    for (int d = 0; d < 3; ++d) {
                        box.lo[d] = std::min(box.lo[d], g[d]);
                        box.hi[d] = std::max(box.hi[d], g[d]);
                    }
                }
    }
    if (a.splenic_vein && T.label("portal_vein_and_splenic_vein") >= 0) add(T, T.label("portal_vein_and_splenic_vein"));
    if (box.hi[0] < 0) box = Box{{0, 0, 0}, {0, 0, 0}};  // nothing labelled: every mask is empty
    const bool lps = a.lps;
    const std::string frame_note = lps ? ", LPS mm" : ", RAS mm";

    struct Item {
        BundleOrgan O;
        Mesh mesh;
        bool required = false;
    };
    std::vector<Item> items;  // file order
    auto emit = [&](const std::string& stem, const std::string& source, const Volume<uint8_t>& mask, const MeshOptions& mo, bool optional) {
        Item it;
        it.O.stem = stem;
        it.O.source = source;
        const int64_t n = count(mask);
        it.O.voxel_ml = n * voxel_ml(mask);
        if (n > 0) it.mesh = tsc::mask_to_mesh(mask, 1, mo, box.lo);
        if (it.mesh.indices.empty()) {
            it.O.note = n == 0 ? "no voxels" : fmt("only pieces < %g mL", mo.min_piece_ml);
            it.O.note += optional ? ", placeholder written" : ", EMPTY file written!";
            it.mesh = Mesh();
        } else {
            it.O.mesh_ml = tsc::mesh_volume_ml(it.mesh);
            to_frame(it.mesh, lps);
        }
        it.required = !optional;
        items.push_back(std::move(it));
    };

    // ---- masks
    Volume<uint8_t> liver = empty_box(box, A);
    or_label(liver, box, T, T.label("liver"));
    Volume<uint8_t> gb = empty_box(box, A);
    or_label(gb, box, T, T.label("gallbladder"));
    Volume<uint8_t> clip = liver;
    dilate_mm(clip, box, a.margin_mm);
    auto vessel_mask = [&](bool portal, std::string& lname) {
        const int label = portal ? V.label_like("portal") : V.label_like("hepatic");
        if (label < 0) throw std::runtime_error("liver bundle: vessel labels not found in custom:950:liver");
        lname = V.labels[(size_t)label];
        Volume<uint8_t> m = empty_box(box, A);
        or_label(m, box, V, label);
        and_mask(m, clip);
        return m;
    };
    std::string ln_portal, ln_vein;
    Volume<uint8_t> portal = vessel_mask(true, ln_portal);
    Volume<uint8_t> vein = vessel_mask(false, ln_vein);
    const std::string clipnote = fmt(" (clip liver+%gmm)", a.margin_mm);
    // IVC: whole cross-sections from the liver's top - ivc_margin_mm to its bottom + ivc_margin_mm
    Volume<uint8_t> ivc = empty_box(box, A);
    or_label(ivc, box, T, T.label("inferior_vena_cava"));
    {
        const Shape3 bs = box.shape();
        for (int64_t i = 0; i < bs[0]; ++i)
            for (int64_t j = 0; j < bs[1]; ++j)
                for (int64_t k = 0; k < bs[2]; ++k) {
                    const int64_t g = (sax == 0 ? i : sax == 1 ? j : k) + box.lo[sax];
                    if (g < slab_lo || g > slab_hi) ivc(i, j, k) = 0;
                }
    }
    const std::string slabnote = fmt(" (liver top/bottom +%gmm)", a.ivc_margin_mm);
    std::string portal_src = "model950:" + ln_portal + clipnote, vein_src = "model950:" + ln_vein + clipnote;
    if (a.splenic_vein) {  // option: TS portal trunk and splenic vein, uncut
        or_label(portal, box, T, T.label("portal_vein_and_splenic_vein"));
        portal_src += " + total:portal_vein_and_splenic_vein (uncut)";
    }
    if (a.ivc_into_vein) {  // option for BLISS, which reads no ivc.obj
        or_mask(vein, ivc);
        vein_src += " + total:inferior_vena_cava" + slabnote;
    }

    emit("liver", "total:liver", liver, tsc::mesh_options_for("liver", true), false);
    emit("gb", "total:gallbladder", gb, tsc::mesh_options_for("gallbladder", true), true);
    emit("portal", portal_src, portal, tsc::mesh_options_for("portal_vein_intrahepatic", false), false);
    emit("vein", vein_src, vein, tsc::mesh_options_for("hepatic_veins", false), true);
    if (!a.ivc_into_vein) emit("ivc", "total:inferior_vena_cava" + slabnote, ivc, tsc::mesh_options_for("inferior_vena_cava", true), true);

    // ---- tumor: 26-connected pieces of TS liver_tumor; tumor.obj = union of the kept pieces
    BundleResult R;
    Volume<uint8_t> tumor_all = empty_box(box, A);
    or_label(tumor_all, box, LV, LV.label("liver_tumor"));
    Volume<uint8_t> tumor_kept = empty_box(box, A);
    {
        const double vml = voxel_ml(tumor_all);
        size_t kept = 0;
        auto ps = pieces(tumor_all, box);
        for (auto& p : ps) {
            BundleTumor C;
            C.ml = p.vox.size() * vml;
            const double sgn = lps ? -1.0 : 1.0;
            C.pos[0] = sgn * p.c[0];
            C.pos[1] = sgn * p.c[1];
            C.pos[2] = p.c[2];
            C.keep = C.ml >= a.tumor_min_ml;
            if (C.keep) {
                ++kept;
                for (int64_t v : p.vox) tumor_kept.data[(size_t)v] = 1;
            }
            R.tumors.push_back(C);
        }
        Item it;
        it.O.stem = "tumor";
        it.O.source = fmt("liver_vessels:liver_tumor (%zu of %zu candidates kept)", kept, R.tumors.size());
        it.O.voxel_ml = count(tumor_kept) * voxel_ml(tumor_kept);
        if (kept) {
            MeshOptions mo = tsc::mesh_options_for("liver_tumor", false);
            mo.min_piece_ml = 0;
            it.mesh = tsc::mask_to_mesh(tumor_kept, 1, mo, box.lo);
            it.O.mesh_ml = tsc::mesh_volume_ml(it.mesh);
            to_frame(it.mesh, lps);
        } else {
            it.O.note = R.tumors.empty() ? "no candidates, placeholder written" : "no candidate kept, placeholder written";
        }
        items.push_back(std::move(it));
    }

    for (int i = 1; i <= 8; ++i) {
        const std::string f = "S" + std::to_string(i), name = "liver_segment_" + std::to_string(i);
        Volume<uint8_t> m = empty_box(box, A);
        or_label(m, box, S, S.label(name));
        emit(f, "liver_segments:" + name, m, tsc::mesh_options_for(name, true), true);
    }

    // ---- files (Engine::write)
    const Mesh* liver_mesh = &items[0].mesh;
    for (auto& it : items) {
        BundleOrgan& O = it.O;
        const std::string file = O.stem + ".obj";
        if (!it.mesh.indices.empty()) {
            R.files.push_back({file, obj_text(it.mesh, O.stem + ": " + O.source + frame_note, a.version)});
            O.written = true;
            O.vertices = it.mesh.num_vertices();
            O.triangles = it.mesh.num_triangles();
        } else if (it.required) {
            R.files.push_back({file, obj_text(Mesh(), O.stem + " (empty)", a.version)});
            O.written = true;
        } else if (!liver_mesh->indices.empty()) {
            R.files.push_back({file, obj_text(placeholder_cube(*liver_mesh, 5.0f),
                                              O.stem + ": PLACEHOLDER 5 mm cube beside the liver (" + O.note + ")" + frame_note, a.version)});
            O.written = true;
            O.placeholder = true;
        }
        R.organs.push_back(O);
    }
    R.files.push_back({"preset_choice.txt", "high\n"});

    // ---- log (the exporter's export_log.txt sections)
    R.log += "tumor candidates (pieces of liver_vessels:liver_tumor):\n";
    for (size_t i = 0; i < R.tumors.size(); ++i) {
        const auto& C = R.tumors[i];
        R.log += fmt("  %zu: %7.2f mL  at (%.0f, %.0f, %.0f) mm  %s\n", i + 1, C.ml, C.pos[0], C.pos[1], C.pos[2], C.keep ? "kept" : "not kept");
    }
    R.log += "files (voxel volume of the mask, volume of the written mesh):\n";
    for (auto& O : R.organs) {
        const std::string f = O.stem + ".obj";
        if (O.triangles)
            R.log += fmt("  %-10s %9.1f mL  mesh %9.1f mL  %8zu vertices %8zu triangles  <- %s\n", f.c_str(), O.voxel_ml, O.mesh_ml,
                         O.vertices, O.triangles, O.source.c_str());
        else
            R.log += fmt("  %-10s %9.1f mL  %s  <- %s\n", f.c_str(), O.voxel_ml, O.note.c_str(), O.source.c_str());
    }

    // ---- display label map (not part of the exporter): segments inside the liver, then gb, vessels, kept tumor
    tumor_all = clip = Volume<uint8_t>();  // memory
    Volume<uint8_t> seg = empty_box(box, A);  // liver_segments in the box: 1..8
    for (int i = 1; i <= 8; ++i) {
        Volume<uint8_t> m = empty_box(box, A);
        or_label(m, box, S, S.label("liver_segment_" + std::to_string(i)));
        for (size_t v = 0; v < m.data.size(); ++v)
            if (m.data[v]) seg.data[v] = (uint8_t)i;
    }
    R.display_names.assign(19, "");
    R.display_names[0] = "background";
    R.display_names[1] = "liver";
    R.display_names[2] = "portal_vein";
    R.display_names[3] = "hepatic_vein";
    R.display_names[4] = "gallbladder";
    R.display_names[5] = "liver_tumor";
    R.display_names[6] = "inferior_vena_cava";
    for (int i = 1; i <= 8; ++i) R.display_names[(size_t)(10 + i)] = "liver_segment_" + std::to_string(i);
    R.display = Volume<uint8_t>(G, A);  // 0 outside the box
    const Shape3 bs = box.shape();
    for (int64_t i = 0; i < bs[0]; ++i)
        for (int64_t j = 0; j < bs[1]; ++j)
            for (int64_t k = 0; k < bs[2]; ++k) {
                const size_t v = (size_t)liver.idx(i, j, k);
                uint8_t o = 0;
                if (liver.data[v]) o = seg.data[v] ? (uint8_t)(10 + seg.data[v]) : 1;
                if (gb.data[v]) o = 4;
                if (portal.data[v]) o = 2;
                if (ivc.data[v]) o = 6;
                if (vein.data[v]) o = 3;
                if (tumor_kept.data[v]) o = 5;
                R.display(i + box.lo[0], j + box.lo[1], k + box.lo[2]) = o;
            }
    return R;
}

}  // namespace browseg
