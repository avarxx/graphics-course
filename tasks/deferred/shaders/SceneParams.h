#ifndef SCENE_PARAMS_H_INCLUDED
#define SCENE_PARAMS_H_INCLUDED

#include "cpp_glsl_compat.h"


// Per-frame constants shared by every pass (depth prepass, G-buffer, decals, lighting, forward).
// NOTE: only vec4/mat4 members to keep std140 layout identical in C++ and GLSL
struct SceneParams
{
  shader_mat4 viewProj;
  shader_mat4 invViewProjRel; // inverse of proj * (view without translation): NDC -> world
                              // position RELATIVE to the camera (better float precision)
  shader_vec4 cameraPos;      // xyz = camera position, w = time
  shader_vec4 screen;         // xy = resolution, zw = 1 / resolution
  shader_vec4 sunDir;         // xyz = direction TOWARDS the sun, w = sun intensity
  shader_vec4 sunColor;       // rgb = sun color, w = ambient intensity
  shader_vec4 skyColor;       // rgb = sky color (linear), w = exposure
  shader_vec4 groundColor;    // rgb = ambient color from below, w = fog density
  shader_vec4 terrainSize;    // x = world size, y = height scale, z = sea level, w = heightmap res
  shader_vec4 layerHeights;   // sand->grass, grass->rock, rock->snow, blend width (all normalized)
  shader_vec4 detail;         // x = meters per texture tile, y = detail normal strength,
                              // z = slope -> rock strength, w = height blend sharpness
  shader_vec4 debug;          // x = debug view, y = use detail normals, z = roughness scale,
                              // w = cavity AO strength
  shader_vec4 gbuffer;        // x = 1 if the compact G-buffer layout is used, 0 for the naive one
  shader_vec4 lighting;       // x = ambient specular strength (sky reflections)
};

// Debug views (SceneParams::debug.x)
#define DEBUG_VIEW_SHADED 0
#define DEBUG_VIEW_ALBEDO 1
#define DEBUG_VIEW_NORMAL 2
#define DEBUG_VIEW_ROUGHNESS 3
#define DEBUG_VIEW_METALLIC 4
#define DEBUG_VIEW_AO 5
#define DEBUG_VIEW_DEPTH 6
#define DEBUG_VIEW_POSITION 7
#define DEBUG_VIEW_LAYERS 8
#define DEBUG_VIEW_HEIGHT 9


#endif // SCENE_PARAMS_H_INCLUDED
