// Label mask -> triangle surface mesh (world RAS mm coordinates).
// Pipeline: optional largest connected component -> block-average to ~target
// resolution -> Gaussian smoothing -> Surface Nets iso-surface (0.5) ->
// Taubin smoothing.
#pragma once

#include <string>

#include "common.h"

namespace tsc {

struct Mesh {
    std::vector<float> vertices;    // xyz triplets, RAS mm
    std::vector<uint32_t> indices;  // triangles
    std::vector<float> normals;     // per-vertex (computed by compute_normals)
    size_t num_vertices() const { return vertices.size() / 3; }
    size_t num_triangles() const { return indices.size() / 3; }
};

struct MeshOptions {
    double target_res_mm = 1.0;   // block-average the mask to about this resolution
    double smooth_sigma_mm = 1.5; // Gaussian smoothing of the mask before iso-surfacing
    int taubin_iterations = 20;
    bool largest_component = true;
    double min_piece_ml = 0.0;    // drop 26-connected pieces smaller than this (0 = keep all)
};

// origin: index of mask(0, 0, 0) on the grid of mask.affine, when the mask is a crop of that grid (nullptr = 0, 0, 0);
// the vertices are then the same as for the uncropped mask.
Mesh mask_to_mesh(const Volume<uint8_t>& mask, int label, const MeshOptions& opt = {}, const int64_t* origin = nullptr);

// Settings by structure: thin tubular structures (vessels) keep the native resolution
// and get only light smoothing, otherwise they collapse; organs use the defaults.
inline MeshOptions mesh_options_for(const std::string& name, bool single_structure) {
    MeshOptions o;
    o.largest_component = single_structure;
    // thin intrahepatic vessel trees (TS liver_vessels and its portal / hepatic split)
    const bool thin_tree = name == "liver_vessels" || name == "portal_vein_intrahepatic" || name == "hepatic_veins";
    // large vessels / trunks: keep the native resolution but smooth like an organ surface
    const bool trunk = !thin_tree && (name.find("vein") != std::string::npos || name.find("vena") != std::string::npos ||
                                      name.find("arter") != std::string::npos || name.find("aorta") != std::string::npos);
    if (thin_tree) {
        // compared with the IRCAD V6 VTK pipeline (marching cubes + windowed sinc) on the same
        // liver_vessels mask: no pre-blur + 30 Taubin steps keeps 95% of the voxel volume and
        // removes the slice terracing; 60+ steps start folding faces
        o.target_res_mm = 0.0;
        o.smooth_sigma_mm = 0.0;
        o.taubin_iterations = 30;
        o.largest_component = false;
        o.min_piece_ml = 0.03;  // like V6: drop specks, keep real disconnected branches
    } else if (trunk) {
        o.target_res_mm = 0.0;
        o.smooth_sigma_mm = 1.0;
        o.taubin_iterations = 20;
        o.largest_component = false;
        o.min_piece_ml = 0.03;
    }
    return o;
}
void compute_normals(Mesh& m);
double mesh_volume_ml(const Mesh& m);

bool write_stl(const Mesh& m, const std::string& path);
bool write_obj(const Mesh& m, const std::string& path);

}  // namespace tsc
