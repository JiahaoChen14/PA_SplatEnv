#include "filter3d_api.h"
#include "filter3d.h"
#include "helper_math.h"
#include <functional>

void SPaGS::filter3d::update_3d_filter_wrapper(
    const torch::Tensor& positions,
    const torch::Tensor& cam_position,
    torch::Tensor& filter_3d,
    torch::Tensor& visibility_mask,
    const float near_plane,
    const float distance2filter)
{
    const uint n_points = positions.size(0);

    update_3d_filter(
        reinterpret_cast<const float3*>(positions.data_ptr<float>()),
        reinterpret_cast<const float3*>(cam_position.data_ptr<float>()),
        filter_3d.data_ptr<float>(),
        visibility_mask.data_ptr<bool>(),
        n_points,
        near_plane,
        distance2filter);
}
