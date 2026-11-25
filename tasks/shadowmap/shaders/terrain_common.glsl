#ifndef TERRAIN_COMMON_GLSL_INCLUDED
#define TERRAIN_COMMON_GLSL_INCLUDED

// R32_SFLOAT is not guaranteed to be filterable, so we do bilinear filtering by hand.
float sampleHeight(sampler2D height_map, vec2 uv)
{
  const ivec2 size = textureSize(height_map, 0);
  const vec2 p = clamp(uv, 0.0, 1.0) * vec2(size - 1);
  const ivec2 i0 = min(ivec2(floor(p)), size - 2);
  const vec2 f = p - vec2(i0);

  const float a = texelFetch(height_map, i0, 0).r;
  const float b = texelFetch(height_map, i0 + ivec2(1, 0), 0).r;
  const float c = texelFetch(height_map, i0 + ivec2(0, 1), 0).r;
  const float d = texelFetch(height_map, i0 + ivec2(1, 1), 0).r;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// uv in [0,1]^2 -> world space XZ
vec2 uvToWorldXZ(vec2 uv, float world_size)
{
  return (uv - 0.5) * world_size;
}

vec2 worldXZToUv(vec2 xz, float world_size)
{
  return xz / world_size + 0.5;
}

// Normal of the heightfield reconstructed from neighbouring heightmap texels
vec3 terrainNormal(sampler2D height_map, vec2 uv, float world_size, float height_scale)
{
  const float texel = 1.0 / float(textureSize(height_map, 0).x - 1);
  const float hL = sampleHeight(height_map, uv - vec2(texel, 0));
  const float hR = sampleHeight(height_map, uv + vec2(texel, 0));
  const float hD = sampleHeight(height_map, uv - vec2(0, texel));
  const float hU = sampleHeight(height_map, uv + vec2(0, texel));
  const float worldStep = 2.0 * texel * world_size;
  const float dhdx = (hR - hL) * height_scale / worldStep;
  const float dhdz = (hU - hD) * height_scale / worldStep;
  return normalize(vec3(-dhdx, 1.0, -dhdz));
}

#endif // TERRAIN_COMMON_GLSL_INCLUDED
