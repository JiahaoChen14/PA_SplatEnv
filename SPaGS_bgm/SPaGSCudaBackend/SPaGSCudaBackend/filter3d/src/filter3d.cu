#include "filter3d.h"
#include "filter3d_config.h"
#include "filter3d_kernels.cuh"
#include "helper_math.h"
#include "utils.h"

void SPaGS::filter3d::update_3d_filter(
    const float3* positions,
    const float3* cam_position,
    float* filter_3d,
    bool* visibility_mask,
    const uint n_points,
    const float near,
    const float distance2filter)
{
    update_3d_filter_cu<<<div_round_up(n_points, config::block_size_update_3d_filter), config::block_size_update_3d_filter>>>(
        positions,
        cam_position,
        filter_3d,
        visibility_mask,
        n_points,
        near,
        distance2filter);
    CHECK_CUDA(config::debug, "update_3d_filter_cu");
}
