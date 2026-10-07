#include "filter3d_kernels.cuh"
#include "helper_math.h"

namespace SPaGS::filter3d {

    __global__ void update_3d_filter_cu(
        const float3* positions,
        const float3* cam_position,
        float* filter_3d,
        bool* visibility_mask,
        const uint n_points,
        const float near,
        const float distance2filter)
    {
        const uint point_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (point_idx >= n_points) return;
        const float3 position_world = positions[point_idx];
        const float distance = length(position_world - cam_position[0]);
        if (distance < near) return;
        const float filter_3d_new = distance2filter * distance;
        if (filter_3d[point_idx] < filter_3d_new) return;
        filter_3d[point_idx] = filter_3d_new;
        visibility_mask[point_idx] = true;
    }

}
