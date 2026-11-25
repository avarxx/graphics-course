#ifndef SHADING_GLSL_INCLUDED
#define SHADING_GLSL_INCLUDED

// Lighting of a single surface point. Used by the deferred lighting pass and by the forward
// reference path, so both produce exactly the same image.
// Expects `params` (SceneParams) and `skyCube` (samplerCube with the baked sky) to be declared.

#include "surface.glsl"
#include "pbr.glsl"

// --- Sky cubemap lookups -------------------------------------------------------------------------

// Radiance of the sky in the given direction (full resolution: what is seen through the pixels
// without geometry). Explicit LOD 0: implicit derivatives are undefined in the non-uniform
// "is this pixel sky?" branch, and a 2048 face is about one texel per pixel at a 60-90 deg FOV.
vec3 skyRadiance(vec3 dir)
{
  return textureLod(skyCube, dir, 0.0).rgb;
}

// Diffuse sky light for a surface with normal N. A very blurry mip (4x4 texels per face)
// approximates the irradiance; the lookup is tilted up so that side-facing surfaces see the sky
// above the horizon rather than the gradient below it (the ground bounce is added separately).
vec3 skyDiffuse(vec3 N)
{
  if (params.sky.x < 0.5)
    return params.skyColor.rgb;
  const vec3 dir = normalize(vec3(N.x, max(N.y, 0.25), N.z));
  return textureLod(skyCube, dir, max(params.sky.y - 3.0, 0.0)).rgb;
}

// Specular sky reflection: rougher surfaces read blurrier mips
vec3 skySpecular(vec3 R, float roughness)
{
  if (params.sky.x < 0.5)
    return params.skyColor.rgb;
  const vec3 dir = normalize(vec3(R.x, max(R.y, 0.02), R.z));
  return textureLod(skyCube, dir, roughness * max(params.sky.y - 4.0, 0.0)).rgb;
}

// Color the distant terrain fades into: the (blurred) sky right above the horizon in the view
// direction, so the fog matches the sky behind the island
vec3 fogColor(vec3 viewDir)
{
  if (params.sky.x < 0.5)
    return params.skyColor.rgb;
  const vec3 dir = normalize(vec3(viewDir.x, max(viewDir.y, 0.0) + 0.03, viewDir.z));
  return textureLod(skyCube, dir, params.sky.w).rgb;
}

// Fraction of the sunlight that passes through the cloud layer (alpha of the baked cubemap).
// A blurry mip (~3 degrees per texel) averages the clouds around the sun, so that the light
// dims smoothly when a cloud passes rather than flickering with the noise detail.
float sunTransmittance(vec3 L)
{
  const float t = L.y > 0.0 ? textureLod(skyCube, L, max(params.sky.y - 6.0, 0.0)).a : 1.0;
  return mix(1.0, t, params.sky.z);
}

// --- Surface lighting ----------------------------------------------------------------------------

// Linear HDR radiance leaving the surface towards the camera (before exposure and tone mapping)
vec3 shadeSurfaceHDR(Surface s, vec3 wPos)
{
  const vec3 V = normalize(params.cameraPos.xyz - wPos);
  const vec3 L = normalize(params.sunDir.xyz);
  const vec3 radiance = params.sunColor.rgb * params.sunDir.w * sunTransmittance(L);

  // Cook-Torrance (GGX) for the sun + ambient light from the sky
  vec3 color = shadeDirectional(s.albedo, s.roughness, s.metallic, s.normal, V, L, radiance);
  color += shadeAmbient(
    s.albedo * (1.0 - s.metallic), s.ao, s.normal, skyDiffuse(s.normal), params.groundColor.rgb,
    params.sunColor.w);

  const float roughness = clamp(s.roughness, 0.04, 1.0);
  const vec3 R = normalize(mix(reflect(-V, s.normal), s.normal, roughness * roughness));
  color += params.lighting.x * shadeAmbientSpecular(
    s.albedo, s.roughness, s.metallic, s.ao, s.normal, V,
    skySpecular(R, roughness), params.groundColor.rgb, params.sunColor.w);

  // Underwater tint: the island is surrounded by the sea
  const float seaHeight = params.terrainSize.z * params.terrainSize.y;
  if (wPos.y < seaHeight)
  {
    const float depth = seaHeight - wPos.y;
    const vec3 waterColor = vec3(0.02, 0.12, 0.18) * (params.sunDir.w * 0.3 + params.sunColor.w);
    const float k = 1.0 - exp(-depth * 0.35);
    color = mix(color * vec3(0.6, 0.85, 0.9), waterColor, clamp(k, 0.0, 0.97));
  }

  // Distance fog towards the sky color behind the surface
  const float dist = length(params.cameraPos.xyz - wPos);
  const float fog = 1.0 - exp(-dist * params.groundColor.w);
  color = mix(color, fogColor(-V), clamp(fog, 0.0, 1.0));

  return color;
}

// Debug views show the raw G-buffer contents instead of the lit color
bool debugViewColor(Surface s, vec3 wPos, out vec3 color)
{
  const int mode = int(params.debug.x + 0.5);
  const float dist = length(params.cameraPos.xyz - wPos);
  color = vec3(0.0);
  if (mode == DEBUG_VIEW_ALBEDO || mode == DEBUG_VIEW_LAYERS || mode == DEBUG_VIEW_HEIGHT)
    color = s.albedo;
  else if (mode == DEBUG_VIEW_NORMAL)
    color = s.normal * 0.5 + 0.5;
  else if (mode == DEBUG_VIEW_ROUGHNESS)
    color = vec3(s.roughness);
  else if (mode == DEBUG_VIEW_METALLIC)
    color = vec3(s.metallic);
  else if (mode == DEBUG_VIEW_AO)
    color = vec3(s.ao);
  else if (mode == DEBUG_VIEW_DEPTH)
    color = vec3(1.0 - exp(-dist / 250.0));
  else if (mode == DEBUG_VIEW_POSITION)
  {
    // 10 m world-space grid: lines must stay glued to the surface while the camera moves
    const vec3 cell = fract(wPos / 10.0);
    const vec3 line = smoothstep(vec3(0.0), vec3(0.04), cell) * smoothstep(vec3(0.0), vec3(0.04), 1.0 - cell);
    color = mix(vec3(1.0, 0.85, 0.2), fract(wPos / 100.0), min(line.x, min(line.y, line.z)));
  }
  else
    return false;
  return true;
}

// Color written into the HDR target: linear radiance (exposure, tone mapping and gamma are
// applied by the post-processing pass), or the raw debug value
vec3 shadeSurface(Surface s, vec3 wPos)
{
  vec3 debugColor;
  if (debugViewColor(s, wPos, debugColor))
    return debugColor;
  return shadeSurfaceHDR(s, wPos);
}

#endif // SHADING_GLSL_INCLUDED
