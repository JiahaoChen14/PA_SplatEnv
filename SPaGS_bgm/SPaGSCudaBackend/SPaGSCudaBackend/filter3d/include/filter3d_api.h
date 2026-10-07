#pragma once

#include <torch/extension.h>

namespace SPaGS::filter3d {

    void update_3d_filter_wrapper(
        const torch::Tensor& positions,
        const torch::Tensor& cam_position,
        torch::Tensor& filter_3d,
        torch::Tensor& visibility_mask,
        const float near_plane,
        const float distance2filter);

}
