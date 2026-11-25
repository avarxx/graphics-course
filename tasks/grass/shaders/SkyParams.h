#ifndef SKY_PARAMS_H_INCLUDED
#define SKY_PARAMS_H_INCLUDED

#include "cpp_glsl_compat.h"


// Parameters of the procedural sky that is baked into a cubemap (shaders/sky_bake.comp).
// NOTE: only vec4 members to keep std140 layout identical in C++ and GLSL
struct SkyParams
{
  shader_vec4 sunDir;       // xyz = normalized direction TOWARDS the sun, w = sun intensity
  shader_vec4 sunColor;     // rgb = sun color, w = cubemap face size in texels
  shader_vec4 horizonColor; // rgb = light blue (used at t = 0), w = sky brightness multiplier
  shader_vec4 zenithColor;  // rgb = dark blue (used at t = 1), w = sunlight scattering strength
  shader_vec4 sunGlow;      // x = narrow glow weight, y = wide halo weight,
                            // z = narrow glow sharpness (9000), w = halo falloff (42)
  shader_vec4 clouds;       // x = coverage [0, 1], y = density (extinction per meter at d = 1),
                            // z = feature size (meters), w = 1 if clouds are enabled
  shader_vec4 layer;        // x = layer bottom (m), y = layer top (m), z = primary ray steps,
                            // w = shadow (light) ray steps
  shader_vec4 lighting;     // x = ambient strength, y = forward scattering g,
                            // z = detail erosion strength, w = aerial perspective distance (m)
  shader_vec4 offset;       // xyz = noise offset (wind / "seed"), w = first face of this dispatch
};


#endif // SKY_PARAMS_H_INCLUDED
