#ifndef DECAL_DATA_H_INCLUDED
#define DECAL_DATA_H_INCLUDED

#include "cpp_glsl_compat.h"


// One projective decal: an oriented box. The decal texture is projected along the box's local
// Y axis, texture u goes along local X and v along local Z.
struct DecalGpu
{
  shader_mat4 boxToWorld; // unit cube [-0.5, 0.5]^3 -> world
  shader_mat4 worldToBox; // inverse of the above
  shader_vec4 params0;    // x = texture layer, y = opacity, z = albedo intensity,
                          // w = normal intensity
  shader_vec4 params1;    // x = roughness intensity, y = angle fade start (cos), z, w unused
  shader_vec4 tint;       // rgb = albedo multiplier
};

#define DECAL_MAX_COUNT 256


#endif // DECAL_DATA_H_INCLUDED
