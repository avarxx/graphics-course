#ifndef GRASS_PARAMS_H_INCLUDED
#define GRASS_PARAMS_H_INCLUDED

#include "cpp_glsl_compat.h"

// Keep this in sync with the placement buffer in the C++ renderer
#define GRASS_MAX_INSTANCES 1500000

// One instance is exactly one blade, as required by the task
#define GRASS_SEGMENTS 5
#define GRASS_BLADES_PER_TUFT 1


// Push constants of the GPU placement compute shader (not needed by the draw passes)
struct GrassPlacementPush
{
  shader_vec4 grid;  // x = cells per side, y = cell size (m), z = radius (m), w = blades per cell
  shader_vec4 extra; // x = seed, y = 1 if per-blade frustum culling is enabled
};


#endif // GRASS_PARAMS_H_INCLUDED
