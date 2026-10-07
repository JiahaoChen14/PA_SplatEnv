#pragma once

#include "helper_math.h"
#include "kernel_utils.cuh"
#include "alpha_blend_first_k/config.h"
#include <cooperative_groups.h>

namespace SPaGS::rasterization::alpha_blend_first_k::kernels::backward {

    __global__ void preprocess_cu(
        const float3* positions,
        const float3* sh_rest,
        const uint* primitive_n_touched_tiles,
        const bool* primitive_rgb_clamp_info,
        const float* densification_info_helper,
        float3* grad_positions,
        float3* grad_sh_0,
        float3* grad_sh_rest,
        float* densification_info,
        const uint n_primitives,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const bool use_distance_scaling)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;

        // sh/position gradients from view-dependent color
        const float3 position_world = positions[primitive_idx];
        const float3 drgb_dposition = convert_sh_to_rgb_backward(
            sh_rest,
            primitive_rgb_clamp_info,
            grad_sh_0,
            grad_sh_rest,
            position_world,
            n_primitives,
            primitive_idx,
            active_sh_bases,
            total_sh_bases
        );
        if (densification_info != nullptr) {
            const float3 dL_dposition = grad_positions[primitive_idx] + drgb_dposition;
            grad_positions[primitive_idx] = dL_dposition;

            // create densification_info from position gradients
            float distance_scale = 1.0f;
            if (use_distance_scaling) {
                const float4 M1 = c_M[0];
                const float4 M2 = c_M[1];
                const float4 M3 = c_M[2];
                const float3 position_view = make_float3(
                    dot(make_float3(M1), position_world) + M1.w,
                    dot(make_float3(M2), position_world) + M2.w,
                    dot(make_float3(M3), position_world) + M3.w
                );
                distance_scale = length(position_view) * 0.5f;
            }
            densification_info[primitive_idx] += 1.0f;
            densification_info[n_primitives + primitive_idx] += length(dL_dposition * distance_scale);
            if (densification_info_helper != nullptr) densification_info[2 * n_primitives + primitive_idx] += distance_scale * densification_info_helper[primitive_idx];
        }
        else {
            grad_positions[primitive_idx] += drgb_dposition;
        }
    }

    template <int K>
    __global__ void blend_cu(
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float4* primitive_MT1,
        const float4* primitive_MT2,
        const float4* primitive_MT3,
        const uint* pixel_primitive_indices_core,
        const float4* pixel_grad_info_core,
        const float* grad_image,
        const float* grad_alphas,
        const float* alphas,
        float3* grad_positions,
        float3* grad_scales,
        float4* grad_rotations,
        float* grad_opacities,
        float3* grad_sh_0,
        float* densification_info,
        const uint width,
        const uint height,
        const uint n_fragments,
        const uint n_pixels)
    {
        const uint fragment_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (fragment_idx >= n_fragments) return;
        const uint pixel_idx = fragment_idx / K;
        const uint core_idx = fragment_idx % K;
        const uint primitive_idx = pixel_primitive_indices_core[core_idx * n_pixels + pixel_idx];
        if (primitive_idx == __UINT32_MAX__) return;
        const float pixel_x = __uint2float_rn(pixel_idx % width) + 0.5f;
        const float pixel_y = __uint2float_rn(pixel_idx / width) + 0.5f;
        constexpr float PI = 3.14159265358979323846f;
        constexpr float PI2 = 2.0f * PI;
        const float pixel_phi = PI2 * (pixel_x / __uint2float_rn(width) - 0.5f);
        const float pixel_theta = PI * (pixel_y / __uint2float_rn(height) - 0.5f);
        const float cos_phi = cosf(pixel_phi);
        const float sin_phi = sinf(pixel_phi);
        const float cos_theta = cosf(pixel_theta);
        const float sin_theta = sinf(pixel_theta);
        const float3 ray_direction = normalize(make_float3(sin_phi * cos_theta, sin_theta, -cos_phi * cos_theta));
        const float3 ray_plane_normal_x = normalize(make_float3(cos_phi, 0.0f, sin_phi));
        const float3 ray_plane_normal_y = normalize(cross(ray_direction, ray_plane_normal_x));
        // load gradient data
        const float3 grad_pixel = make_float3(
            grad_image[pixel_idx],
            grad_image[n_pixels + pixel_idx],
            grad_image[2 * n_pixels + pixel_idx]
        );
        const float4 precomputed_grad = pixel_grad_info_core[core_idx * n_pixels + pixel_idx];
        const float3 dL_drgb = precomputed_grad.w * grad_pixel;
        float dL_dalpha = dot(make_float3(precomputed_grad), grad_pixel);
        // load primitive data
        const float4 MT1 = primitive_MT1[primitive_idx];
        const float4 MT2 = primitive_MT2[primitive_idx];
        const float4 MT3 = primitive_MT3[primitive_idx];
        const float opacity = opacities[primitive_idx];
        const float4 plane_x_diag = MT1 * ray_plane_normal_x.x + MT3 * ray_plane_normal_x.z;
        const float4 plane_y_diag = MT1 * ray_plane_normal_y.x + MT2 * ray_plane_normal_y.y + MT3 * ray_plane_normal_y.z;
        const float3 plane_x_diag_normal = make_float3(plane_x_diag);
        const float3 plane_y_diag_normal = make_float3(plane_y_diag);
        const float3 m = plane_x_diag.w * plane_y_diag_normal - plane_x_diag_normal * plane_y_diag.w;
        const float3 d = cross(plane_x_diag_normal, plane_y_diag_normal);
        const float numerator_rho2 = dot(m, m);
        const float denominator_rcp = 1.0f / dot(d, d);
        const float G = expf(-0.5f * numerator_rho2 * denominator_rcp);

        // grad from background color
        const float alpha = fminf(opacity * G, config::max_fragment_alpha);
        dL_dalpha += grad_alphas[pixel_idx] * (1.0f - alphas[pixel_idx]) * (1.0f / fmaxf(1.0f - alpha, config::one_minus_alpha_eps));

        // color gradient
        if (dL_drgb.x != 0.0f) atomicAdd(&grad_sh_0[primitive_idx].x, dL_drgb.x);
        if (dL_drgb.y != 0.0f) atomicAdd(&grad_sh_0[primitive_idx].y, dL_drgb.y);
        if (dL_drgb.z != 0.0f) atomicAdd(&grad_sh_0[primitive_idx].z, dL_drgb.z);

        // opacity gradient
        if (dL_dalpha != 0.0f) {
            const float dL_dopacity = dL_dalpha * G;
            atomicAdd(&grad_opacities[primitive_idx], dL_dopacity);

            const float dL_dG = dL_dalpha * opacity;
            const float dL_drho2 = dL_dG * -0.5f * G;
            const float dL_dnum = dL_drho2 * denominator_rcp;
            const float dL_ddenom = dL_drho2 * -numerator_rho2 * denominator_rcp * denominator_rcp;
            const float3 dL_dm = dL_dnum * 2.0f * m;
            const float3 dL_dd = dL_ddenom * 2.0f * d;

            const float4 dL_dplane_x_diag = make_float4(
                -plane_y_diag.w * dL_dm.x - plane_y_diag.z * dL_dd.y + plane_y_diag.y * dL_dd.z,
                -plane_y_diag.w * dL_dm.y + plane_y_diag.z * dL_dd.x - plane_y_diag.x * dL_dd.z,
                -plane_y_diag.w * dL_dm.z - plane_y_diag.y * dL_dd.x + plane_y_diag.x * dL_dd.y,
                dot(plane_y_diag_normal, dL_dm)
            );
            const float4 dL_dplane_y_diag = make_float4(
                plane_x_diag.w * dL_dm.x + plane_x_diag.z * dL_dd.y - plane_x_diag.y * dL_dd.z,
                plane_x_diag.w * dL_dm.y - plane_x_diag.z * dL_dd.x + plane_x_diag.x * dL_dd.z,
                plane_x_diag.w * dL_dm.z + plane_x_diag.y * dL_dd.x - plane_x_diag.x * dL_dd.y,
                -dot(plane_x_diag_normal, dL_dm)
            );

            const float4 dL_dMT1 = dL_dplane_x_diag * ray_plane_normal_x.x + dL_dplane_y_diag * ray_plane_normal_y.x;
            const float4 dL_dMT2 = dL_dplane_y_diag * ray_plane_normal_y.y;
            const float4 dL_dMT3 = dL_dplane_x_diag * ray_plane_normal_x.z + dL_dplane_y_diag * ray_plane_normal_y.z;
            const float3 dL_dMT_c1 = make_float3(dL_dMT1.x, dL_dMT2.x, dL_dMT3.x);
            const float3 dL_dMT_c2 = make_float3(dL_dMT1.y, dL_dMT2.y, dL_dMT3.y);
            const float3 dL_dMT_c3 = make_float3(dL_dMT1.z, dL_dMT2.z, dL_dMT3.z);
            const float3 dL_dMT_c4 = make_float3(dL_dMT1.w, dL_dMT2.w, dL_dMT3.w);

            const float4 M1 = c_M[0];
            const float4 M2 = c_M[1];
            const float4 M3 = c_M[2];
            const float3 M_c1 = make_float3(M1.x, M2.x, M3.x);
            const float3 M_c2 = make_float3(M1.y, M2.y, M3.y);
            const float3 M_c3 = make_float3(M1.z, M2.z, M3.z);

            // position gradient
            const float3 dL_dposition = make_float3(
                dot(M_c1, dL_dMT_c4),
                dot(M_c2, dL_dMT_c4),
                dot(M_c3, dL_dMT_c4)
            );
            atomicAdd(&grad_positions[primitive_idx].x, dL_dposition.x);
            atomicAdd(&grad_positions[primitive_idx].y, dL_dposition.y);
            atomicAdd(&grad_positions[primitive_idx].z, dL_dposition.z);

            // load/re-compute scale and rotation
            const float3 scale = scales[primitive_idx];
            const float4 quaternion = rotations[primitive_idx];
            const Mat3x3 R = convert_quaterion_to_rotation_matrix(quaternion);

            // scale gradient
            const float3 R_c1 = make_float3(R.r11, R.r21, R.r31);
            const float3 R_c2 = make_float3(R.r12, R.r22, R.r32);
            const float3 R_c3 = make_float3(R.r13, R.r23, R.r33);
            const float3 dL_dscale = make_float3(
                dot(make_float3(dot(make_float3(M1), R_c1), dot(make_float3(M2), R_c1), dot(make_float3(M3), R_c1)), dL_dMT_c1),
                dot(make_float3(dot(make_float3(M1), R_c2), dot(make_float3(M2), R_c2), dot(make_float3(M3), R_c2)), dL_dMT_c2),
                dot(make_float3(dot(make_float3(M1), R_c3), dot(make_float3(M2), R_c3), dot(make_float3(M3), R_c3)), dL_dMT_c3)
            );
            atomicAdd(&grad_scales[primitive_idx].x, dL_dscale.x);
            atomicAdd(&grad_scales[primitive_idx].y, dL_dscale.y);
            atomicAdd(&grad_scales[primitive_idx].z, dL_dscale.z);

            // rotation gradient
            const Mat3x3 dL_dR = {
                dot(M_c1, dL_dMT_c1) * scale.x, dot(M_c1, dL_dMT_c2) * scale.y, dot(M_c1, dL_dMT_c3) * scale.z,
                dot(M_c2, dL_dMT_c1) * scale.x, dot(M_c2, dL_dMT_c2) * scale.y, dot(M_c2, dL_dMT_c3) * scale.z,
                dot(M_c3, dL_dMT_c1) * scale.x, dot(M_c3, dL_dMT_c2) * scale.y, dot(M_c3, dL_dMT_c3) * scale.z
            };
            const float4 dL_drotation = convert_quaterion_to_rotation_matrix_backward(quaternion, dL_dR);
            atomicAdd(&grad_rotations[primitive_idx].x, dL_drotation.x);
            atomicAdd(&grad_rotations[primitive_idx].y, dL_drotation.y);
            atomicAdd(&grad_rotations[primitive_idx].z, dL_drotation.z);
            atomicAdd(&grad_rotations[primitive_idx].w, dL_drotation.w);

            // densification info
            if (densification_info != nullptr) atomicAdd(&densification_info[primitive_idx], fabsf(dL_dposition.x) + fabsf(dL_dposition.y) + fabsf(dL_dposition.z));

        }
    }

}
