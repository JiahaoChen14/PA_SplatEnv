#include "shared_kernels.cuh"
#include "helper_math.h"
#include "utils.h"
#include <cstdint>
#include <cooperative_groups.h>

namespace SPaGS::rasterization::shared_kernels {

    template <typename KeyT>
    __global__ void create_instances_cu(
        const uint* primitive_n_touched_tiles,
        const uint* primitive_offsets,
        const uint4* primitive_screen_bounds,
        KeyT* instance_keys,
        uint* instance_primitive_indices,
        const uint grid_width,
        const uint n_primitives)
    {
        constexpr uint block_size = 256u;
        constexpr uint n_sequential_threshold = 8u;

        auto block = cooperative_groups::this_thread_block();
        auto warp = cooperative_groups::tiled_partition<32u>(block);
        uint primitive_idx = cooperative_groups::this_grid().thread_rank();

        bool active = true;
        if (primitive_idx >= n_primitives) {
            active = false;
            primitive_idx = n_primitives - 1;
        }
        const uint tile_count_init = primitive_n_touched_tiles[primitive_idx];
        if (tile_count_init == 0) active = false;
        if (__ballot_sync(0xffffffffu, active) == 0) return;

        uint current_write_offset = (primitive_idx == 0) ? 0 : primitive_offsets[primitive_idx - 1];
        const uint4 screen_bounds_init = primitive_screen_bounds[primitive_idx];
        const uint screen_bounds_width_init = screen_bounds_init.y - screen_bounds_init.x;

        if (active) {
            for (uint instance_idx = 0; instance_idx < tile_count_init && instance_idx < n_sequential_threshold; instance_idx++) {
                const uint y = screen_bounds_init.z + (instance_idx / screen_bounds_width_init);
                const uint x = screen_bounds_init.x + (instance_idx % screen_bounds_width_init);
                const KeyT tile_idx = y * grid_width + (x % grid_width);
                instance_keys[current_write_offset] = tile_idx;
                instance_primitive_indices[current_write_offset] = primitive_idx;
                current_write_offset++;
            }
        }

        const uint lane_idx = cooperative_groups::this_thread_block().thread_rank() % 32u;
        const uint warp_idx = cooperative_groups::this_thread_block().thread_rank() / 32u;
        const uint lane_mask_allprev_excl = 0xffffffffu >> (32u - lane_idx);

        const int compute_cooperatively = active && tile_count_init > n_sequential_threshold;
        const uint remaining_threads = __ballot_sync(0xffffffffu, compute_cooperatively);
        if (remaining_threads == 0) return;

        __shared__ uint4 collected_screen_bounds[block_size];
        collected_screen_bounds[block.thread_rank()] = screen_bounds_init;

        uint n_remaining_threads = __popc(remaining_threads);
        for (int n = 0; n < n_remaining_threads && n < 32; n++) {
            int i = __fns(remaining_threads, 0, n + 1); // find lane index of next remaining thread

            uint primitive_idx_coop = __shfl_sync(0xffffffffu, primitive_idx, i);
            uint current_write_offset_coop = __shfl_sync(0xffffffffu, current_write_offset, i);

            const uint4 screen_bounds = collected_screen_bounds[warp.meta_group_rank() * 32 + i];

            const uint screen_bounds_width = screen_bounds.y - screen_bounds.x;
            const uint tile_count = screen_bounds_width * (screen_bounds.w - screen_bounds.z);
            const uint remaining_tile_count = tile_count - n_sequential_threshold;

            const int n_iterations = div_round_up(remaining_tile_count, 32u);
            for (int it = 0; it < n_iterations; it++) {
                const int instance_idx = it * 32 + lane_idx + n_sequential_threshold;
                const int active_curr_it = instance_idx < tile_count;

                const uint y = screen_bounds.z + (instance_idx / screen_bounds_width);
                const uint x = screen_bounds.x + (instance_idx % screen_bounds_width);

                const uint write = active_curr_it && true;

                const uint write_ballot = __ballot_sync(0xffffffffu, write);
                const uint n_writes = __popc(write_ballot);

                const uint write_offset_it = __popc(write_ballot & lane_mask_allprev_excl);
                const uint write_offset = current_write_offset_coop + write_offset_it;

                if (write) {
                    const KeyT tile_idx = y * grid_width + (x % grid_width);
                    instance_keys[write_offset] = tile_idx;
                    instance_primitive_indices[write_offset] = primitive_idx_coop;
                }
                current_write_offset_coop += n_writes;
            }

            __syncwarp();
        }
    }

    __global__ void create_instances_cu(
        const uint* primitive_n_touched_tiles,
        const uint* primitive_offsets,
        const uint4* primitive_screen_bounds,
        const float* primitive_depths,
        uint64_t* instance_keys,
        uint* instance_primitive_indices,
        const uint grid_width,
        const uint n_primitives)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;
        const uint4 screen_bounds = primitive_screen_bounds[primitive_idx];
        uint offset = (primitive_idx == 0) ? 0 : primitive_offsets[primitive_idx - 1];
        const uint64_t depth_key = __float_as_uint(primitive_depths[primitive_idx]);
        for (uint y = screen_bounds.z; y < screen_bounds.w; ++y) {
            for (uint x = screen_bounds.x; x < screen_bounds.y; ++x) {
                const uint64_t tile_idx = y * grid_width + (x % grid_width);
                instance_keys[offset] = (tile_idx << 32) | depth_key;
                instance_primitive_indices[offset] = primitive_idx;
                offset++;
            }
        }
    }

    template <typename KeyT>
    __global__ void extract_instance_ranges_cu(
        const KeyT* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const KeyT instance_tile_idx = instance_keys[instance_idx];
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const KeyT previous_instance_tile_idx = instance_keys[instance_idx - 1];
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }
    
    __global__ void extract_instance_ranges_cu(
        const uint64_t* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const uint64_t instance_key = instance_keys[instance_idx];
        const uint instance_tile_idx = instance_key >> 32;
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const uint64_t previous_instance_key = instance_keys[instance_idx - 1];
            const uint previous_instance_tile_idx = previous_instance_key >> 32;
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    template __global__ void create_instances_cu<uint>(
        const uint*, const uint*, const uint4*, uint*, uint*, const uint, const uint);
    template __global__ void create_instances_cu<ushort>(
        const uint*, const uint*, const uint4*, ushort*, uint*, const uint, const uint);
    template __global__ void extract_instance_ranges_cu<uint>(
        const uint*, uint2*, const uint);
    template __global__ void extract_instance_ranges_cu<ushort>(
        const ushort*, uint2*, const uint);

}
