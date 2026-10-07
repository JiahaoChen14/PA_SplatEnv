#pragma once

#include "helper_math.h"

__device__ __constant__ float4 c_M[3];
__device__ __constant__ float3 c_cam_position;

struct Mat3x3 {
    float r11, r12, r13;
    float r21, r22, r23;
    float r31, r32, r33;
};

template<typename T>
__device__ void swap(
    T& a,
    T& b)
{
    T temp = a;
    a = b;
    b = temp;
}

__forceinline__ __device__ Mat3x3 convert_quaterion_to_rotation_matrix(
    const float4& quaternion)
{
    auto [r, x, y, z] = quaternion;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float rx = r * x, ry = r * y, rz = r * z;
    return {
        1.0f - 2.0f * (yy + zz), 2.0f * (xy - rz), 2.0f * (xz + ry),
        2.0f * (xy + rz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - rx),
        2.0f * (xz - ry), 2.0f * (yz + rx), 1.0f - 2.0f * (xx + yy)
    };
}

__forceinline__ __device__ float4 convert_quaterion_to_rotation_matrix_backward(
    const float4& quaternion,
    const Mat3x3& dL_dR)
{
    auto [r, x, y, z] = quaternion;
    const float dL_dR_r21_sub_r12 = dL_dR.r21 - dL_dR.r12;
    const float dL_dR_r21_add_r12 = dL_dR.r21 + dL_dR.r12;
    const float dL_dR_r13_sub_r31 = dL_dR.r13 - dL_dR.r31;
    const float dL_dR_r13_add_r31 = dL_dR.r13 + dL_dR.r31;
    const float dL_dR_r32_sub_r23 = dL_dR.r32 - dL_dR.r23;
    const float dL_dR_r32_add_r23 = dL_dR.r32 + dL_dR.r23;
    return {
        2.0f * (x * dL_dR_r32_sub_r23 + y * dL_dR_r13_sub_r31 + z * dL_dR_r21_sub_r12),
        2.0f * (r * dL_dR_r32_sub_r23 - 2.0f * x * (dL_dR.r22 + dL_dR.r33) + y * dL_dR_r21_add_r12 + z * dL_dR_r13_add_r31),
        2.0f * (r * dL_dR_r13_sub_r31 + x * dL_dR_r21_add_r12 - 2.0f * y * (dL_dR.r11 + dL_dR.r33) + z * dL_dR_r32_add_r23),
        2.0f * (r * dL_dR_r21_sub_r12 + x * dL_dR_r13_add_r31 + y * dL_dR_r32_add_r23 - 2.0f * z * (dL_dR.r11 + dL_dR.r22))
    };
}

__forceinline__ __device__ bool transform_and_cull(
    const float3* scales,
    const float4* rotations,
    const float3& position_world,
    const float& opacity,
    uint& n_touched_tiles,
    uint4& screen_bounds,
    float4& MT1,
    float4& MT2,
    float4& MT3,
    float& distance,
    const uint primitive_idx,
    const uint width,
    const uint height,
    const uint grid_width,
    const uint grid_height,
    const uint tile_width,
    const uint tile_height,
    const float near,
    const float min_alpha_threshold_rcp,
    const float scale_modifier)
{
    // early near culling
    const float4 M1 = c_M[0];
    const float4 M2 = c_M[1];
    const float4 M3 = c_M[2];
    const float3 position_view = make_float3(
        dot(make_float3(M1), position_world) + M1.w,
        dot(make_float3(M2), position_world) + M2.w,
        dot(make_float3(M3), position_world) + M3.w
    );
    distance = length(position_view);
    if (distance < near) return true;

    // load scale, rotation, and opacity
    const float3 scale = scales[primitive_idx];
    const float4 quaternion = rotations[primitive_idx];
    const Mat3x3 R = convert_quaterion_to_rotation_matrix(quaternion);

    // compute MT
    const float3 u = make_float3(R.r11 * scale.x, R.r21 * scale.x, R.r31 * scale.x) * scale_modifier;
    const float3 v = make_float3(R.r12 * scale.y, R.r22 * scale.y, R.r32 * scale.y) * scale_modifier;
    const float3 w = make_float3(R.r13 * scale.z, R.r23 * scale.z, R.r33 * scale.z) * scale_modifier;
    MT1 = make_float4(dot(make_float3(M1), u), dot(make_float3(M1), v), dot(make_float3(M1), w), position_view.x);
    MT2 = make_float4(dot(make_float3(M2), u), dot(make_float3(M2), v), dot(make_float3(M2), w), position_view.y);
    MT3 = make_float4(dot(make_float3(M3), u), dot(make_float3(M3), v), dot(make_float3(M3), w), position_view.z);

    // ### START BOUNDING BOX CALCULATION ###
    constexpr float FLT_EPS = 1e-8f;
    constexpr float PI = 3.14159265358979323846f;
    constexpr float PI_RCP = 1.0f / PI;
    constexpr float PI_RCP2 = 2.0f * PI_RCP;
    constexpr float BB_EPS = 1e-12f;

    // construct camera2tangent matrix
    const float3 tangent_plane_normal = normalize(position_view);

    // precomputed variables
    const float mx2 = tangent_plane_normal.x * tangent_plane_normal.x;
    const float mz2 = tangent_plane_normal.z * tangent_plane_normal.z;
    const float mx2mz2 = mx2 + mz2;
    const float sr_mx2mz2 = sqrtf(mx2mz2);
    const float3 reference = (fabsf(position_view.x) < FLT_EPS && fabsf(position_view.z) < FLT_EPS) ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(0.0f, 1.0f, 0.0f);

    const float3 tangent_plane_tangent = normalize(cross(tangent_plane_normal, reference));
    const float3 tangent_plane_bitangent = normalize(cross(tangent_plane_tangent, tangent_plane_normal));

    // projection matrix
    // const float proj_a = 1.0f; // ((far + NEAR) / (far - near)), because we assume an infinite far plane
    const float proj_b = -2.0f * near; // ((-2.0f * far * near) / (far - near))

    // MT helpers
    const float3 MT_c1 = make_float3(MT1.x, MT2.x, MT3.x);
    const float3 MT_c2 = make_float3(MT1.y, MT2.y, MT3.y);
    const float3 MT_c3 = make_float3(MT1.z, MT2.z, MT3.z);
    const float3 MT_c4 = make_float3(MT1.w, MT2.w, MT3.w);

    // splat to tangent plane transform
    const float4 T_tangent_r1 = make_float4(dot(tangent_plane_tangent, MT_c1), dot(tangent_plane_tangent, MT_c2), dot(tangent_plane_tangent, MT_c3), dot(tangent_plane_tangent, MT_c4));
    const float4 T_tangent_r2 = make_float4(dot(tangent_plane_bitangent, MT_c1), dot(tangent_plane_bitangent, MT_c2), dot(tangent_plane_bitangent, MT_c3), dot(tangent_plane_bitangent, MT_c4));
    const float4 T_tangent_r3 = make_float4(dot(tangent_plane_normal, MT_c1), dot(tangent_plane_normal, MT_c2), dot(tangent_plane_normal, MT_c3), dot(tangent_plane_normal, MT_c4) + proj_b);
    const float4 T_tangent_r4 = make_float4(dot(tangent_plane_normal, MT_c1), dot(tangent_plane_normal, MT_c2), dot(tangent_plane_normal, MT_c3), dot(tangent_plane_normal, MT_c4));
    
    // correct cutoff for the used opacity threshold
    const float rho_cutoff = 2.0f * logf(opacity * min_alpha_threshold_rcp);
    const float4 t = make_float4(rho_cutoff, rho_cutoff, rho_cutoff, -1.0f);
    const float d = dot(t, T_tangent_r4 * T_tangent_r4);
    if (d == 0.0f) return true;
    const float4 f = (1.0f / d) * t;

    // compute bounding box "on" tangent plane
    const float3 center_tangent = make_float3(dot(f, T_tangent_r1 * T_tangent_r4), dot(f, T_tangent_r2 * T_tangent_r4), dot(f, T_tangent_r3 * T_tangent_r4));
    const float3 extent_tangent = sqrtf(fmaxf(center_tangent * center_tangent - make_float3(dot(f, T_tangent_r1 * T_tangent_r1), dot(f, T_tangent_r2 * T_tangent_r2), dot(f, T_tangent_r3 * T_tangent_r3)), BB_EPS));
    if ((center_tangent.z + extent_tangent.z) >= 1.0f || (center_tangent.z - extent_tangent.z) <= -1.0f) return true;
    if ((extent_tangent.x < BB_EPS) || (extent_tangent.y < BB_EPS) || (extent_tangent.z < BB_EPS)) return true;
    
    // compute relevant points on tangent plane
    const float2 min_bounds = make_float2(center_tangent.x - extent_tangent.x, center_tangent.y - extent_tangent.y);
    const float2 max_bounds = make_float2(center_tangent.x + extent_tangent.x, center_tangent.y + extent_tangent.y);
    if ((min_bounds.x >= 0.0f) || (min_bounds.y >= 0.0f) || (max_bounds.x <= 0.0f) || (max_bounds.y <= 0.0f)) return true;

    // reused helper
    const float width_f = __uint2float_rn(width);
    const float height_f = __uint2float_rn(height);
    const float2 half_wh = make_float2(0.5f * width_f, 0.5f * height_f);
    const float pole_determinator_ty = (tangent_plane_normal.y > 0.0f) ? max_bounds.y : min_bounds.y;
    const float pole_determinator = sr_mx2mz2 - tangent_plane_normal.y * pole_determinator_ty;
    const float t_x_abs = fmaxf(-min_bounds.x, max_bounds.x);
    const float t_x_abs2p1 = t_x_abs * t_x_abs + 1.0f;
    const float max_by2 = max_bounds.y * max_bounds.y;
    const float min_by2 = min_bounds.y * min_bounds.y;

    float px_min, px_max, py_min, py_max;

    // bbx cover pole
    if (pole_determinator <= 0.0f) {
        // X bounds on images plane (pole)
        // x_max
        px_max = width_f;

        // x_min
        px_min = 0.0f;

        // Y bounds on images plane (pole)
        const float inner_ty = (tangent_plane_normal.y > 0.0f) ? min_bounds.y : max_bounds.y;
        const float inner_cy = sr_mx2mz2 * inner_ty + tangent_plane_normal.y;
        const float bound_temp_a = tangent_plane_normal.y / fmaxf(sr_mx2mz2, FLT_EPS);
        const float bound_deter_left_r = (bound_temp_a + min_bounds.y) / (bound_temp_a + max_bounds.y);
        const float bound_deter_left = bound_deter_left_r * bound_deter_left_r;
        const float bound_deter_right = (t_x_abs2p1 + min_by2) / (t_x_abs2p1 + max_by2);
        const float ext_ty = (bound_deter_left > bound_deter_right) ? max_bounds.y : min_bounds.y;
        const float sin_py_main_ref = (tangent_plane_normal.y * inner_cy <= 0.0f) ? inner_cy * rsqrtf(inner_ty * inner_ty + 1.0f) : (sr_mx2mz2 * ext_ty + tangent_plane_normal.y) * rsqrtf(t_x_abs2p1 + ext_ty * ext_ty);
        // The only different determination part for different reference vector
        const float sin_py = (reference.z == 1.0f) ? tangent_plane_normal.y * rsqrtf(inner_ty * inner_ty + t_x_abs2p1) : sin_py_main_ref;
        
        // Final py
        const float py_universal = half_wh.y * (1.0f + PI_RCP2 * asinf(clamp(sin_py, -1.0f, 1.0f)));

        // y_max
        py_max = (tangent_plane_normal.y > 0.0f) ? height_f : py_universal;

        // y_min
        py_min = (tangent_plane_normal.y > 0.0f) ? py_universal : 0.0f;
    }
    else {
        // X bounds on images plane (not pole)
        const float ref_lon = atan2f(tangent_plane_normal.x, -tangent_plane_normal.z);
        
        // x_max
        px_max = half_wh.x * (1.0f + PI_RCP * (ref_lon + atan2f(max_bounds.x, pole_determinator))) + width_f;

        // x_min
        px_min = half_wh.x * (1.0f + PI_RCP * (ref_lon + atan2f(min_bounds.x, pole_determinator))) + width_f;


        // Y bounds on images plane (not pole)
        // y_max
        const float cy_max = sr_mx2mz2 * max_bounds.y + tangent_plane_normal.y;
        float t_x_abs_max2p1 = (cy_max > 0.0f) ? 1.0f : t_x_abs2p1;
        py_max = half_wh.y * (1.0f + PI_RCP2 * asinf(clamp(cy_max * rsqrtf(t_x_abs_max2p1 + max_by2), -1.0f, 1.0f)));

        // y_min
        const float cy_min = sr_mx2mz2 * min_bounds.y + tangent_plane_normal.y;
        float t_x_abs_min2p1 = (cy_min > 0.0f) ? t_x_abs2p1 : 1.0f;
        py_min = half_wh.y * (1.0f + PI_RCP2 * asinf(clamp(cy_min * rsqrtf(t_x_abs_min2p1 + min_by2), -1.0f, 1.0f)));
    }
    
    if (px_max < px_min) return true;
    if ((px_max - px_min < 1.0f) || (py_max - py_min < 1.0f)) return true;

    // final screen-space bounding box in tile coordinates
    screen_bounds = make_uint4(
        __float2uint_rd(px_min / tile_width), // x_min
        __float2uint_ru(px_max / tile_width), // x_max
        __float2uint_rd(py_min / tile_height), // y_min
        __float2uint_ru(py_max / tile_height) // y_max
    );
    // ### END BOUNDING BOX CALCULATION ###

    // compute number of potentially influenced tiles
    n_touched_tiles = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
    return n_touched_tiles == 0;
}

template <bool train_mode>
__forceinline__ __device__ float3 convert_sh_to_rgb(
    const float3* sh_0,
    const float3* sh_rest,
    [[maybe_unused]] bool* rgb_clamp_info,
    const float3& position_world,
    const uint n_primitives,
    const uint primitive_idx,
    const uint active_sh_bases,
    const uint total_sh_bases)
{
    // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
    float3 result = 0.5f + 0.28209479177387814f * sh_0[primitive_idx];
    if (active_sh_bases > 1) {
        const float3* coefficients_ptr = sh_rest + primitive_idx * total_sh_bases;
        auto [x, y, z] = normalize(position_world - c_cam_position);
        result = result + (-0.48860251190291987f * y) * coefficients_ptr[0]
                        + (0.48860251190291987f * z) * coefficients_ptr[1]
                        + (-0.48860251190291987f * x) * coefficients_ptr[2];
        if (active_sh_bases > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            result = result + (1.0925484305920792f * xy) * coefficients_ptr[3]
                            + (-1.0925484305920792f * yz) * coefficients_ptr[4]
                            + (0.94617469575755997f * zz - 0.31539156525251999f) * coefficients_ptr[5]
                            + (-1.0925484305920792f * xz) * coefficients_ptr[6]
                            + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * coefficients_ptr[7];
            if (active_sh_bases > 9) {
                result = result + (0.59004358992664352f * y * (-3.0f * xx + yy)) * coefficients_ptr[8]
                                + (2.8906114426405538f * xy * z) * coefficients_ptr[9]
                                + (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * coefficients_ptr[10]
                                + (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * coefficients_ptr[11]
                                + (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * coefficients_ptr[12]
                                + (1.4453057213202769f * z * (xx - yy)) * coefficients_ptr[13]
                                + (0.59004358992664352f * x * (-xx + 3.0f * yy)) * coefficients_ptr[14];
            }
        }
    }
    if constexpr (train_mode) {
        rgb_clamp_info[primitive_idx] = result.x < 0;
        rgb_clamp_info[n_primitives + primitive_idx] = result.y < 0;
        rgb_clamp_info[2 * n_primitives + primitive_idx] = result.z < 0;
    }
    return {
        fmaxf(0.0f, result.x),
        fmaxf(0.0f, result.y),
        fmaxf(0.0f, result.z)
    };
}

__forceinline__ __device__ float3 convert_sh_to_rgb_backward(
    const float3* sh_rest,
    const bool* rgb_clamp_info,
    float3* grad_sh_0,
    float3* grads_sh_rest,
    const float3& position_world,
    const uint n_primitives,
    const uint primitive_idx,
    const uint active_sh_bases,
    const uint total_sh_bases)
{
    const int coefficients_base_idx = primitive_idx * total_sh_bases;
    const float3* coefficients_ptr = sh_rest + coefficients_base_idx;
    float3* grad_coefficients_ptr = grads_sh_rest + coefficients_base_idx;

    const float3 grad_rgb_raw = grad_sh_0[primitive_idx];
    const float3 grad_rgb = make_float3(
        rgb_clamp_info[primitive_idx] ? 0.0f : grad_rgb_raw.x,
        rgb_clamp_info[n_primitives + primitive_idx] ? 0.0f : grad_rgb_raw.y,
        rgb_clamp_info[2 * n_primitives + primitive_idx] ? 0.0f : grad_rgb_raw.z
    );

    grad_sh_0[primitive_idx] = 0.28209479177387814f * grad_rgb;
    float3 drgb_dposition = make_float3(0.0f);
    if (active_sh_bases > 1) {
        auto [x_raw, y_raw, z_raw] = position_world - c_cam_position;
        auto [x, y, z] = normalize(make_float3(x_raw, y_raw, z_raw));
        grad_coefficients_ptr[0] = (-0.48860251190291987f * y) * grad_rgb;
        grad_coefficients_ptr[1] = (0.48860251190291987f * z) * grad_rgb;
        grad_coefficients_ptr[2] = (-0.48860251190291987f * x) * grad_rgb;
        float3 grad_direction_x = -0.48860251190291987f * coefficients_ptr[2];
        float3 grad_direction_y = -0.48860251190291987f * coefficients_ptr[0];
        float3 grad_direction_z = 0.48860251190291987f * coefficients_ptr[1];
        if (active_sh_bases > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            grad_coefficients_ptr[3] = (1.0925484305920792f * xy) * grad_rgb;
            grad_coefficients_ptr[4] = (-1.0925484305920792f * yz) * grad_rgb;
            grad_coefficients_ptr[5] = (0.94617469575755997f * zz - 0.31539156525251999f) * grad_rgb;
            grad_coefficients_ptr[6] = (-1.0925484305920792f * xz) * grad_rgb;
            grad_coefficients_ptr[7] = (0.54627421529603959f * xx - 0.54627421529603959f * yy) * grad_rgb;
            grad_direction_x = grad_direction_x + (1.0925484305920792f * y) * coefficients_ptr[3]
                                                + (-1.0925484305920792f * z) * coefficients_ptr[6]
                                                + (1.0925484305920792f * x) * coefficients_ptr[7];
            grad_direction_y = grad_direction_y + (1.0925484305920792f * x) * coefficients_ptr[3]
                                                + (-1.0925484305920792f * z) * coefficients_ptr[4]
                                                + (-1.0925484305920792f * y) * coefficients_ptr[7];
            grad_direction_z = grad_direction_z + (-1.0925484305920792f * y) * coefficients_ptr[4]
                                                + (1.8923493915151202f * z) * coefficients_ptr[5]
                                                + (-1.0925484305920792f * x) * coefficients_ptr[6];
            if (active_sh_bases > 9) {
                grad_coefficients_ptr[8] = (0.59004358992664352f * y * (-3.0f * xx + yy)) * grad_rgb;
                grad_coefficients_ptr[9] = (2.8906114426405538f * xy * z) * grad_rgb;
                grad_coefficients_ptr[10] = (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * grad_rgb;
                grad_coefficients_ptr[11] = (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * grad_rgb;
                grad_coefficients_ptr[12] = (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * grad_rgb;
                grad_coefficients_ptr[13] = (1.4453057213202769f * z * (xx - yy)) * grad_rgb;
                grad_coefficients_ptr[14] = (0.59004358992664352f * x * (-xx + 3.0f * yy)) * grad_rgb;
                grad_direction_x = grad_direction_x + (-3.5402615395598609f * xy) * coefficients_ptr[8]
                                                    + (2.8906114426405538f * yz) * coefficients_ptr[9]
                                                    + (0.45704579946446572f - 2.2852289973223288f * zz) * coefficients_ptr[12]
                                                    + (2.8906114426405538f * xz) * coefficients_ptr[13]
                                                    + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * coefficients_ptr[14];
                grad_direction_y = grad_direction_y + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * coefficients_ptr[8]
                                                    + (2.8906114426405538f * xz) * coefficients_ptr[9]
                                                    + (0.45704579946446572f - 2.2852289973223288f * zz) * coefficients_ptr[10]
                                                    + (-2.8906114426405538f * yz) * coefficients_ptr[13]
                                                    + (3.5402615395598609f * xy) * coefficients_ptr[14];
                grad_direction_z = grad_direction_z + (2.8906114426405538f * xy) * coefficients_ptr[9]
                                                    + (-4.5704579946446566f * yz) * coefficients_ptr[10]
                                                    + (5.597644988851731f * zz - 1.1195289977703462f) * coefficients_ptr[11]
                                                    + (-4.5704579946446566f * xz) * coefficients_ptr[12]
                                                    + (1.4453057213202769f * xx - 1.4453057213202769f * yy) * coefficients_ptr[13];
            }
        }

        const float3 grad_direction = make_float3(
            dot(grad_direction_x, grad_rgb),
            dot(grad_direction_y, grad_rgb),
            dot(grad_direction_z, grad_rgb)
        );
        const float xx_raw = x_raw * x_raw, yy_raw = y_raw * y_raw, zz_raw = z_raw * z_raw;
        const float xy_raw = x_raw * y_raw, xz_raw = x_raw * z_raw, yz_raw = y_raw * z_raw;
        const float norm_sq = xx_raw + yy_raw + zz_raw;
        drgb_dposition = make_float3(
            (yy_raw + zz_raw) * grad_direction.x - xy_raw * grad_direction.y - xz_raw * grad_direction.z,
            -xy_raw * grad_direction.x + (xx_raw + zz_raw) * grad_direction.y - yz_raw * grad_direction.z,
            -xz_raw * grad_direction.x - yz_raw * grad_direction.y + (xx_raw + yy_raw) * grad_direction.z
        ) * rsqrtf(norm_sq * norm_sq * norm_sq);
    }
    return drgb_dposition;
}
