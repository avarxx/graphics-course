#ifndef GBUFFER_GLSL_INCLUDED
#define GBUFFER_GLSL_INCLUDED

// G-buffer encoding. Two layouts exist so that their cost can be compared in the profiler:
//
//                     compact (14 B/px)                     naive (28 B/px)
//   RT0  albedo     R8G8B8A8_SRGB   rgb = albedo, a = AO     R16G16B16A16_SFLOAT  rgb = albedo
//   RT1  normal     R16G16_SNORM    octahedral normal        R16G16B16A16_SFLOAT  xyz = normal
//   RT2  material   R8G8_UNORM      r = rough, g = metal     R16G16B16A16_SFLOAT  r, m, ao
//   depth           D32_SFLOAT (reverse-Z)                   D32_SFLOAT (reverse-Z)
//
// The world position is never stored: it is restored from depth (see reconstructWorldPos).

#include "surface.glsl"

vec2 signNotZero(vec2 v)
{
  return vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

// Octahedral normal encoding: unit vector -> [-1, 1]^2
// (Cigolle et al., "A Survey of Efficient Representations for Independent Unit Vectors", 2014)
vec2 octEncode(vec3 n)
{
  const vec2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
  return n.z <= 0.0 ? (1.0 - abs(p.yx)) * signNotZero(p) : p;
}

vec3 octDecode(vec2 e)
{
  vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
  if (n.z < 0.0)
    n.xy = (1.0 - abs(n.yx)) * signNotZero(n.xy);
  return normalize(n);
}

// The scene is mostly "up-facing" (terrain), so we rotate Y into the octahedron's +Z pole:
// the whole upper hemisphere then lands in the inner diamond, where the mapping is continuous.
// That keeps the hardware alpha blending of decal normals well-behaved.
vec2 encodeNormal(vec3 n)
{
  return octEncode(n.xzy);
}

vec3 decodeNormal(vec2 e)
{
  return octDecode(e).xzy;
}

struct GBufferTexel
{
  vec4 rt0;
  vec4 rt1;
  vec4 rt2;
};

GBufferTexel packGBuffer(Surface s, bool compact)
{
  GBufferTexel t;
  if (compact)
  {
    t.rt0 = vec4(s.albedo, s.ao);
    t.rt1 = vec4(encodeNormal(s.normal), 0.0, 0.0);
    t.rt2 = vec4(s.roughness, s.metallic, 0.0, 0.0);
  }
  else
  {
    t.rt0 = vec4(s.albedo, 1.0);
    t.rt1 = vec4(s.normal, 0.0);
    t.rt2 = vec4(s.roughness, s.metallic, s.ao, 0.0);
  }
  return t;
}

Surface unpackGBuffer(GBufferTexel t, bool compact)
{
  Surface s;
  s.albedo = t.rt0.rgb;
  s.roughness = t.rt2.r;
  s.metallic = t.rt2.g;
  if (compact)
  {
    s.ao = t.rt0.a;
    s.normal = decodeNormal(t.rt1.xy);
  }
  else
  {
    s.ao = t.rt2.b;
    s.normal = normalize(t.rt1.xyz);
  }
  return s;
}

// Screen uv in [0,1]^2 (origin at the top-left, like gl_FragCoord) + depth -> world position.
// In Vulkan the NDC y axis also points down, so ndc.xy = uv * 2 - 1 without any flips.
// inv_view_proj_rel is the inverse of proj * (view without translation), so the matrix only
// produces small camera-relative offsets and we add the camera position afterwards: this keeps
// the precision good even far away from the world origin.
vec3 reconstructWorldPos(vec2 uv, float depth, mat4 inv_view_proj_rel, vec3 camera_pos)
{
  const vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
  const vec4 rel = inv_view_proj_rel * clip;
  return camera_pos + rel.xyz / rel.w;
}

#endif // GBUFFER_GLSL_INCLUDED
