#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// HDR (B10G11R11_UFLOAT) -> LDR backbuffer (B8G8R8A8_UNORM):
//   exposure -> tone mapping curve -> sRGB encoding (gamma) -> 3D LUT color grading -> dithering.
// With everything disabled it only copies the image: the UNORM attachment clamps to [0, 1] and
// quantizes to 256 levels by itself, no shader code is needed for that.

#include "TonemapParams.h"
#include "luminance.glsl"

layout(binding = 0) uniform sampler2D hdrImage;

layout(binding = 1, std430) readonly buffer TonemapStateBuffer
{
  TonemapState state;
};

layout(binding = 2) uniform sampler3D gradingLut;

layout(push_constant) uniform Push
{
  PostPush pc;
};

layout(location = 0) in vec2 vUv;

layout(location = 0) out vec4 out_fragColor;

// --- Tone mapping curves (input: exposed linear HDR, output: linear [0, 1]) ---------------------

// Reinhard on luminance with a white point (Reinhard et al. 2002, eq. 4): the hue is kept,
// only the brightness is compressed. Luminance W maps exactly to 1.
vec3 tonemapReinhard(vec3 x)
{
  const float white = 6.0;
  const float l = luminance(x);
  const float ld = l * (1.0 + l / (white * white)) / (1.0 + l);
  return clamp(x * (ld / max(l, 1e-6)), 0.0, 1.0);
}

// Narkowicz's fit of the ACES filmic curve
vec3 tonemapACES(vec3 x)
{
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// Hable's "Uncharted 2" filmic curve
vec3 hablePartial(vec3 x)
{
  const float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 tonemapHable(vec3 x)
{
  const float exposureBias = 2.0;
  const vec3 white = vec3(11.2);
  return clamp(hablePartial(x * exposureBias) / hablePartial(white), 0.0, 1.0);
}

// --- sRGB transfer function (exact piecewise form, not pow(1/2.2)) ------------------------------
vec3 linearToSrgb(vec3 c)
{
  c = clamp(c, 0.0, 1.0);
  const vec3 lo = c * 12.92;
  const vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
  return mix(hi, lo, lessThanEqual(c, vec3(0.0031308)));
}

// Triangular-PDF noise in (-1, 1): +-1/255 dithering hides 8-bit banding of the smooth sky
float tpdfNoise(vec2 p, float seed)
{
  const float a = fract(sin(dot(p + seed, vec2(12.9898, 78.233))) * 43758.5453);
  const float b = fract(sin(dot(p + seed, vec2(39.3468, 11.1353))) * 24634.6345);
  return a + b - 1.0;
}

void main()
{
  const vec3 hdr = texelFetch(hdrImage, ivec2(gl_FragCoord.xy), 0).rgb;

  // Debug views of the G-buffer are shown as-is
  if (pc.settings.w > 0.5)
  {
    out_fragColor = vec4(hdr, 1.0);
    return;
  }

  vec3 color = hdr;

  if (pc.flags.x > 0.5)
  {
    color *= state.exposure;
    const int op = int(pc.flags.y + 0.5);
    if (op == TONEMAP_REINHARD)
      color = tonemapReinhard(color);
    else if (op == TONEMAP_ACES)
      color = tonemapACES(color);
    else if (op == TONEMAP_HABLE)
      color = tonemapHable(color);
    else
      color = clamp(color, 0.0, 1.0);
  }
  else
    color *= pc.settings.x;

  // The backbuffer is UNORM: the values written are what the monitor receives, so the linear
  // color has to be encoded with the sRGB transfer function here
  if (pc.flags.z > 0.5)
    color = linearToSrgb(color);

  // Color grading: the LUT maps display-encoded colors to display-encoded colors (it was built
  // from screenshots). Texel k stores the average of the colors in [k/32, (k+1)/32), whose center
  // is (k + 0.5)/32 -- exactly the texture coordinate of that texel's center, so the color itself
  // is the right coordinate, and trilinear filtering interpolates between neighboring cells.
  if (pc.flags.w > 0.5)
  {
    const vec3 graded = texture(gradingLut, clamp(color, 0.0, 1.0)).rgb;
    color = mix(color, graded, pc.settings.y);
  }

  if (pc.settings.z > 0.5)
    color += tpdfNoise(gl_FragCoord.xy, 0.0) / 255.0;

  out_fragColor = vec4(color, 1.0);
}
