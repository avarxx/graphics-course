#ifndef TERRAIN_PARAMS_H_INCLUDED
#define TERRAIN_PARAMS_H_INCLUDED

#include "cpp_glsl_compat.h"


// NOTE: only vec4/mat4 members to keep std140 layout identical in C++ and GLSL
struct TerrainParams
{
  shader_mat4 viewProj;
  shader_vec4 cameraPos;    // xyz = camera position, w = time
  shader_vec4 sunDir;       // xyz = direction TOWARDS the sun, w = sun intensity
  shader_vec4 sunColor;     // rgb = sun color, w = ambient intensity
  shader_vec4 skyColor;     // rgb = sky color (linear), w = exposure
  shader_vec4 groundColor;  // rgb = ambient color from below, w = fog density
  shader_vec4 terrainSize;  // x = world size, y = height scale, z = sea level, w = heightmap res
  shader_vec4 layerHeights; // sand->grass, grass->rock, rock->snow, blend width (all normalized)
  shader_vec4 detail;       // x = meters per texture tile, y = detail normal strength,
                            // z = slope -> rock strength, w = height blend sharpness
  shader_vec4 debug;        // x = debug view mode, y = use detail normals, z = roughness scale
};


#endif // TERRAIN_PARAMS_H_INCLUDED
