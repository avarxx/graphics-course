#ifndef TONEMAP_PARAMS_H_INCLUDED
#define TONEMAP_PARAMS_H_INCLUDED

#include "cpp_glsl_compat.h"


// Number of histogram bins between the minimum and the maximum log2 luminance of the frame
#define HIST_BINS 100
// Compute passes over the HDR image use 16x16 work groups
#define LUM_GROUP_SIZE 16

// GPU-side state of the automatic exposure (one storage buffer, std430).
// The first part is cleared every frame, the adaptation state persists between frames.
struct TonemapState
{
  // --- cleared every frame -------------------------------------------------------------------
  shader_uint minLogBits; // min log2(luminance), as an order-preserving uint (see luminance.glsl)
  shader_uint maxLogBits; // max log2(luminance)
  shader_uint pad0;
  shader_uint pad1;
  shader_uint histogram[HIST_BINS];
  // --- persistent ----------------------------------------------------------------------------
  shader_float adaptedLogLum; // temporally smoothed log2 luminance the exposure is based on
  shader_float targetLogLum;  // value chosen from this frame's histogram
  shader_float exposure;      // multiplier applied to the HDR color
  shader_uint initialized;    // 0 until the first frame: then adapted = target immediately
};

// Byte offsets for vkCmdFillBuffer
#define TONEMAP_STATE_MAX_OFFSET 4
#define TONEMAP_STATE_CLEAR_END (16 + 4 * HIST_BINS)

// Push constants of the luminance / exposure compute passes
struct TonemapPush
{
  shader_vec4 frame; // xy = image size, z = delta time (s), w = exposure method
  shader_vec4 range; // x = low percentile, y = high percentile, z = min log2 lum, w = max
  shader_vec4 adapt; // x = adaptation speed to brighter, y = to darker, z = key value, w = unused
};

// Push constants of the post-processing pass
struct PostPush
{
  shader_vec4 flags;    // x = tone mapping on, y = operator, z = sRGB encoding on, w = LUT on
  shader_vec4 settings; // x = manual exposure, y = LUT strength, z = dithering on, w = debug view
};

// Exposure methods (TonemapPush::frame.w)
#define EXPOSURE_TRIMMED_MEAN 0
#define EXPOSURE_MEDIAN 1
#define EXPOSURE_MEAN 2

// Tone mapping operators (PostPush::flags.y)
#define TONEMAP_CLAMP 0
#define TONEMAP_REINHARD 1
#define TONEMAP_ACES 2
#define TONEMAP_HABLE 3


#endif // TONEMAP_PARAMS_H_INCLUDED
