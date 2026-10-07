#pragma once

#include "helper_math.h"

namespace SPaGS::filter3d {

    void update_3d_filter(
        const float3* positions,
        const float3* cam_position,
        float* filter_3d,
        bool* visibility_mask,
        const uint n_points,
        const float near,
        const float distance2filter);

}
