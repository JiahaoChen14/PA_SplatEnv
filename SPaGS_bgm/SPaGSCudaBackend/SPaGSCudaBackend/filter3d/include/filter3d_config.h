#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace SPaGS::filter3d::config {
    DEF bool debug = false;
    DEF uint block_size_update_3d_filter = 256;
}

namespace config = SPaGS::filter3d::config;

#undef DEF
