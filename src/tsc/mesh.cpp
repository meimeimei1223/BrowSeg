#include "mesh.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace tsc {

namespace {

void keep_largest_component(std::vector<uint8_t>& m, const Shape3& s) {
    std::vector<int32_t> lab(m.size(), 0);
    std::vector<int64_t> stack;
    int32_t cur = 0, best = 0;
    int64_t best_n = 0;
    const int64_t s12 = s[1] * s[2];
    for (int64_t start = 0; start < (int64_t)m.size(); ++start) {
        if (!m[start] || lab[start]) continue;
        ++cur;
        int64_t n = 0;
        stack.push_back(start);
        lab[start] = cur;
        while (!stack.empty()) {
            int64_t p = stack.back();
            stack.pop_back();
            ++n;
            int64_t i = p / s12, j = (p / s[2]) % s[1], k = p % s[2];
            const int64_t nb[6] = {i > 0 ? p - s12 : -1, i + 1 < s[0] ? p + s12 : -1, j > 0 ? p - s[2] : -1,
                                   j + 1 < s[1] ? p + s[2] : -1, k > 0 ? p - 1 : -1, k + 1 < s[2] ? p + 1 : -1};
            for (int64_t q : nb)
                if (q >= 0 && m[q] && !lab[q]) {
                    lab[q] = cur;
                    stack.push_back(q);
                }
        }
        if (n > best_n) {
            best_n = n;
            best = cur;
        }
    }
    for (size_t i = 0; i < m.size(); ++i) m[i] = lab[i] == best ? 1 : 0;
}

void remove_small_pieces(std::vector<uint8_t>& m, const Shape3& s, int64_t min_vox) {
    std::vector<uint8_t> seen(m.size(), 0);
    std::vector<int64_t> stack, members;
    const int64_t s12 = s[1] * s[2];
    for (int64_t start = 0; start < (int64_t)m.size(); ++start) {
        if (!m[start] || seen[start]) continue;
        members.clear();
        stack.push_back(start);
        seen[start] = 1;
        while (!stack.empty()) {
            const int64_t p = stack.back();
            stack.pop_back();
            members.push_back(p);
            const int64_t i = p / s12, j = (p / s[2]) % s[1], k = p % s[2];
            for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj)
                    for (int dk = -1; dk <= 1; ++dk) {
                        const int64_t ii = i + di, jj = j + dj, kk = k + dk;
                        if (ii < 0 || jj < 0 || kk < 0 || ii >= s[0] || jj >= s[1] || kk >= s[2]) continue;
                        const int64_t q = (ii * s[1] + jj) * s[2] + kk;
                        if (m[q] && !seen[q]) {
                            seen[q] = 1;
                            stack.push_back(q);
                        }
                    }
        }
        if ((int64_t)members.size() < min_vox)
            for (int64_t q : members) m[q] = 0;
    }
}

void gauss_blur_axis(std::vector<float>& f, const Shape3& s, int axis, double sigma) {
    if (sigma < 0.3) return;
    const int r = std::max(1, (int)std::ceil(3 * sigma));
    std::vector<float> k(2 * r + 1);
    double sum = 0;
    for (int i = -r; i <= r; ++i) sum += (k[i + r] = (float)std::exp(-0.5 * i * i / (sigma * sigma)));
    for (auto& v : k) v = (float)(v / sum);
    int64_t stride = 1;
    for (int d = 2; d > axis; --d) stride *= s[d];
    const int64_t len = s[axis];
    const int64_t lines = numel(s) / len;
    parallel_for(lines, [&](int64_t b, int64_t e, int) {
        std::vector<float> buf((size_t)len), out((size_t)len);
        for (int64_t l = b; l < e; ++l) {
            // decompose line index into base offset
            int64_t base;
            if (axis == 2) base = l * len;
            else if (axis == 1) base = (l / s[2]) * s[1] * s[2] + (l % s[2]);
            else base = l;
            for (int64_t i = 0; i < len; ++i) buf[i] = f[base + i * stride];
            for (int64_t i = 0; i < len; ++i) {
                float acc = 0;
                for (int t = -r; t <= r; ++t) {
                    int64_t q = i + t;
                    if (q >= 0 && q < len) acc += k[t + r] * buf[q];
                }
                out[i] = acc;
            }
            for (int64_t i = 0; i < len; ++i) f[base + i * stride] = out[i];
        }
    }, 64);
}

}  // namespace

Mesh mask_to_mesh(const Volume<uint8_t>& vol, int label, const MeshOptions& opt, const int64_t* origin) {
    Mesh mesh;
    const Shape3& S = vol.shape;
    // bbox of the label
    int64_t lo[3], hi[3];
    bbox_where(vol.data.data(), S, [label](uint8_t v) { return v == label; }, lo, hi);
    if (hi[0] < 0) return mesh;
    Shape3 bs{hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1};
    std::vector<uint8_t> m(checked_count(numel(bs)));
    for (int64_t i = 0; i < bs[0]; ++i)
        for (int64_t j = 0; j < bs[1]; ++j)
            for (int64_t k = 0; k < bs[2]; ++k) m[(size_t)((i * bs[1] + j) * bs[2] + k)] = vol(i + lo[0], j + lo[1], k + lo[2]) == label;
    const auto sp = zooms_from_affine(vol.affine);
    if (opt.largest_component) keep_largest_component(m, bs);
    if (opt.min_piece_ml > 0) remove_small_pieces(m, bs, (int64_t)std::ceil(opt.min_piece_ml * 1000.0 / (sp[0] * sp[1] * sp[2])));
    int f[3];
    double sig[3];
    int pad[3];
    Shape3 gs;
    for (int d = 0; d < 3; ++d) {
        f[d] = std::max(1, (int)std::lround(opt.target_res_mm / sp[d]));
        sig[d] = opt.smooth_sigma_mm / (sp[d] * f[d]);
        pad[d] = (int)std::ceil(3 * sig[d]) + 2;
        gs[d] = (bs[d] + f[d] - 1) / f[d] + 2 * pad[d];
    }
    // block average into the padded grid
    std::vector<float> g(checked_count(numel(gs)), 0.f);
    const float inv = 1.0f / (float)(f[0] * f[1] * f[2]);
    for (int64_t i = 0; i < bs[0]; ++i)
        for (int64_t j = 0; j < bs[1]; ++j)
            for (int64_t k = 0; k < bs[2]; ++k)
                if (m[(size_t)((i * bs[1] + j) * bs[2] + k)]) {
                    int64_t gi = i / f[0] + pad[0], gj = j / f[1] + pad[1], gk = k / f[2] + pad[2];
                    g[(size_t)((gi * gs[1] + gj) * gs[2] + gk)] += inv;
                }
    m.clear();
    for (int d = 0; d < 3; ++d) gauss_blur_axis(g, gs, d, sig[d]);

    // ---- Surface Nets
    const float iso = 0.5f;
    auto G = [&](int64_t i, int64_t j, int64_t k) { return g[(size_t)((i * gs[1] + j) * gs[2] + k)]; };
    const Shape3 cs{gs[0] - 1, gs[1] - 1, gs[2] - 1};
    std::vector<int32_t> cell(checked_count(numel(cs)), -1);
    std::vector<float> verts;  // grid coordinates first
    static const int corner[8][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    static const int edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    for (int64_t i = 0; i < cs[0]; ++i)
        for (int64_t j = 0; j < cs[1]; ++j)
            for (int64_t k = 0; k < cs[2]; ++k) {
                float v[8];
                int bits = 0;
                for (int c = 0; c < 8; ++c) {
                    v[c] = G(i + corner[c][0], j + corner[c][1], k + corner[c][2]);
                    if (v[c] > iso) bits |= 1 << c;
                }
                if (bits == 0 || bits == 255) continue;
                double p[3] = {0, 0, 0};
                int n = 0;
                for (auto& e : edges) {
                    const bool a = (bits >> e[0]) & 1, b = (bits >> e[1]) & 1;
                    if (a == b) continue;
                    const double t = (iso - v[e[0]]) / (double)(v[e[1]] - v[e[0]]);
                    for (int d = 0; d < 3; ++d) p[d] += corner[e[0]][d] + t * (corner[e[1]][d] - corner[e[0]][d]);
                    ++n;
                }
                cell[(size_t)((i * cs[1] + j) * cs[2] + k)] = (int32_t)(verts.size() / 3);
                verts.push_back((float)(i + p[0] / n));
                verts.push_back((float)(j + p[1] / n));
                verts.push_back((float)(k + p[2] / n));
            }
    auto C = [&](int64_t i, int64_t j, int64_t k) -> int32_t {
        if (i < 0 || j < 0 || k < 0 || i >= cs[0] || j >= cs[1] || k >= cs[2]) return -1;
        return cell[(size_t)((i * cs[1] + j) * cs[2] + k)];
    };
    auto quad = [&](int32_t a, int32_t b, int32_t c, int32_t d, bool flip) {
        if (a < 0 || b < 0 || c < 0 || d < 0) return;
        if (!flip) {
            mesh.indices.insert(mesh.indices.end(), {(uint32_t)a, (uint32_t)b, (uint32_t)c, (uint32_t)a, (uint32_t)c, (uint32_t)d});
        } else {
            mesh.indices.insert(mesh.indices.end(), {(uint32_t)a, (uint32_t)c, (uint32_t)b, (uint32_t)a, (uint32_t)d, (uint32_t)c});
        }
    };
    for (int64_t i = 0; i < gs[0]; ++i)
        for (int64_t j = 0; j < gs[1]; ++j)
            for (int64_t k = 0; k < gs[2]; ++k) {
                const bool in0 = G(i, j, k) > iso;
                if (i + 1 < gs[0] && in0 != (G(i + 1, j, k) > iso))  // x edge: plane (y,z)
                    quad(C(i, j - 1, k - 1), C(i, j, k - 1), C(i, j, k), C(i, j - 1, k), !in0);
                if (j + 1 < gs[1] && in0 != (G(i, j + 1, k) > iso))  // y edge: plane (z,x)
                    quad(C(i - 1, j, k - 1), C(i - 1, j, k), C(i, j, k), C(i, j, k - 1), !in0);
                if (k + 1 < gs[2] && in0 != (G(i, j, k + 1) > iso))  // z edge: plane (x,y)
                    quad(C(i - 1, j - 1, k), C(i, j - 1, k), C(i, j, k), C(i - 1, j, k), !in0);
            }
    g.clear();
    cell.clear();

    // ---- Taubin smoothing (uniform Laplacian) in grid coordinates
    const size_t nv = verts.size() / 3;
    if (opt.taubin_iterations > 0 && nv > 0) {
        // vertex adjacency as CSR (unique neighbours per vertex)
        std::vector<uint64_t> edges;
        edges.reserve(mesh.indices.size() * 2);
        for (size_t t = 0; t < mesh.indices.size(); t += 3)
            for (int e = 0; e < 3; ++e) {
                const uint64_t a = mesh.indices[t + e], b = mesh.indices[t + (e + 1) % 3];
                edges.push_back(a << 32 | b);
                edges.push_back(b << 32 | a);
            }
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        std::vector<uint32_t> start(nv + 1, 0), adj(edges.size());
        for (size_t i = 0; i < edges.size(); ++i) {
            start[(edges[i] >> 32) + 1]++;
            adj[i] = (uint32_t)(edges[i] & 0xffffffffu);
        }
        for (size_t v = 0; v < nv; ++v) start[v + 1] += start[v];
        edges.clear();
        edges.shrink_to_fit();
        std::vector<float> tmp(verts.size());
        for (int it = 0; it < 2 * opt.taubin_iterations; ++it) {
            const float lam = (it % 2 == 0) ? 0.5f : -0.53f;
            parallel_for((int64_t)nv, [&](int64_t b, int64_t e, int) {
                for (int64_t v = b; v < e; ++v) {
                    float c[3] = {0, 0, 0};
                    const uint32_t s0 = start[v], s1 = start[v + 1];
                    for (uint32_t q = s0; q < s1; ++q)
                        for (int d = 0; d < 3; ++d) c[d] += verts[adj[q] * 3 + d];
                    const float cnt = (float)(s1 - s0);
                    for (int d = 0; d < 3; ++d) {
                        const float p = verts[v * 3 + d];
                        tmp[v * 3 + d] = s1 == s0 ? p : p + lam * (c[d] / cnt - p);
                    }
                }
            }, 1024);
            verts.swap(tmp);
        }
    }

    // ---- grid -> original voxel -> world (RAS mm)
    mesh.vertices.resize(verts.size());
    const Affine& A = vol.affine;
    for (size_t v = 0; v < nv; ++v) {
        double o[3];
        for (int d = 0; d < 3; ++d) o[d] = (double)(lo[d] + (origin ? origin[d] : 0)) + ((double)verts[v * 3 + d] - pad[d]) * f[d] + (f[d] - 1) * 0.5;
        for (int r = 0; r < 3; ++r)
            mesh.vertices[v * 3 + r] = (float)(A[r * 4 + 0] * o[0] + A[r * 4 + 1] * o[1] + A[r * 4 + 2] * o[2] + A[r * 4 + 3]);
    }
    compute_normals(mesh);
    return mesh;
}

void compute_normals(Mesh& m) {
    m.normals.assign(m.vertices.size(), 0.f);
    for (size_t t = 0; t < m.indices.size(); t += 3) {
        const float* a = &m.vertices[m.indices[t] * 3];
        const float* b = &m.vertices[m.indices[t + 1] * 3];
        const float* c = &m.vertices[m.indices[t + 2] * 3];
        const float u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, w[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const float n[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
        for (int e = 0; e < 3; ++e)
            for (int d = 0; d < 3; ++d) m.normals[m.indices[t + e] * 3 + d] += n[d];
    }
    for (size_t v = 0; v < m.normals.size(); v += 3) {
        float l = std::sqrt(m.normals[v] * m.normals[v] + m.normals[v + 1] * m.normals[v + 1] + m.normals[v + 2] * m.normals[v + 2]);
        if (l > 0)
            for (int d = 0; d < 3; ++d) m.normals[v + d] /= l;
    }
}

double mesh_volume_ml(const Mesh& m) {
    double vol = 0;
    for (size_t t = 0; t < m.indices.size(); t += 3) {
        const float* a = &m.vertices[m.indices[t] * 3];
        const float* b = &m.vertices[m.indices[t + 1] * 3];
        const float* c = &m.vertices[m.indices[t + 2] * 3];
        vol += (double)a[0] * (b[1] * c[2] - b[2] * c[1]) - (double)a[1] * (b[0] * c[2] - b[2] * c[0]) +
               (double)a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    return vol / 6.0 / 1000.0;
}

bool write_stl(const Mesh& m, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    char header[80] = "tsc_liver mesh (RAS mm)";
    f.write(header, 80);
    uint32_t n = (uint32_t)m.num_triangles();
    f.write((const char*)&n, 4);
    for (size_t t = 0; t < m.indices.size(); t += 3) {
        const float* a = &m.vertices[m.indices[t] * 3];
        const float* b = &m.vertices[m.indices[t + 1] * 3];
        const float* c = &m.vertices[m.indices[t + 2] * 3];
        float u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, w[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        float nn[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
        float l = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
        if (l > 0)
            for (auto& x : nn) x /= l;
        f.write((const char*)nn, 12);
        f.write((const char*)a, 12);
        f.write((const char*)b, 12);
        f.write((const char*)c, 12);
        uint16_t attr = 0;
        f.write((const char*)&attr, 2);
    }
    return true;
}

bool write_obj(const Mesh& m, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    std::fprintf(f, "# tsc_liver mesh, RAS mm\n");
    for (size_t v = 0; v < m.num_vertices(); ++v)
        std::fprintf(f, "v %.4f %.4f %.4f\n", m.vertices[v * 3], m.vertices[v * 3 + 1], m.vertices[v * 3 + 2]);
    for (size_t v = 0; v < m.normals.size() / 3; ++v)
        std::fprintf(f, "vn %.4f %.4f %.4f\n", m.normals[v * 3], m.normals[v * 3 + 1], m.normals[v * 3 + 2]);
    const bool hn = !m.normals.empty();
    for (size_t t = 0; t < m.indices.size(); t += 3) {
        uint32_t a = m.indices[t] + 1, b = m.indices[t + 1] + 1, c = m.indices[t + 2] + 1;
        if (hn) std::fprintf(f, "f %u//%u %u//%u %u//%u\n", a, a, b, b, c, c);
        else std::fprintf(f, "f %u %u %u\n", a, b, c);
    }
    std::fclose(f);
    return true;
}

}  // namespace tsc
