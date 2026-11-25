#ifndef LUMINANCE_COMMON_GLSL_INCLUDED
#define LUMINANCE_COMMON_GLSL_INCLUDED

// Shared declarations of the passes over the HDR image (min/max and histogram)

#include "TonemapParams.h"
#include "luminance.glsl"

layout(local_size_x = LUM_GROUP_SIZE, local_size_y = LUM_GROUP_SIZE, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D hdrImage;

layout(binding = 1, std430) buffer TonemapStateBuffer
{
  TonemapState state;
};

layout(push_constant) uniform Push
{
  TonemapPush pc;
};

bool loadLogLuminance(out float logLum)
{
  const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
  const bool inside = all(lessThan(pixel, ivec2(pc.frame.xy)));
  logLum = inside ? logLuminance(texelFetch(hdrImage, pixel, 0).rgb) : 0.0;
  return inside;
}

#endif // LUMINANCE_COMMON_GLSL_INCLUDED
