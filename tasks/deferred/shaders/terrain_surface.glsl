#ifndef TERRAIN_SURFACE_GLSL_INCLUDED
#define TERRAIN_SURFACE_GLSL_INCLUDED

// Terrain material evaluation, shared by the G-buffer pass and the forward reference path.
// Expects `params` (SceneParams), `heightMap`, `albedoArray` and `roughHeightArray` to be declared.

#include "surface.glsl"
#include "terrain_common.glsl"

const int LAYER_COUNT = 4;

Surface evalTerrainSurface(vec3 wPos, vec2 uv)
{
  const float worldSize = params.terrainSize.x;
  const float heightScale = params.terrainSize.y;

  // --- Geometry ---------------------------------------------------------------------------------
  const float hNorm = sampleHeight(heightMap, uv);
  const vec3 geomN = terrainNormal(heightMap, uv, worldSize, heightScale);

  // --- Layer weights from altitude and slope ----------------------------------------------------
  const vec4 t = params.layerHeights;
  const float bw = max(t.w, 1e-3);
  const float s0 = smoothstep(t.x - bw, t.x + bw, hNorm);
  const float s1 = smoothstep(t.y - bw, t.y + bw, hNorm);
  const float s2 = smoothstep(t.z - bw, t.z + bw, hNorm);

  float w[LAYER_COUNT];
  w[0] = 1.0 - s0;
  w[1] = s0 - s1;
  w[2] = s1 - s2;
  w[3] = s2;

  // Steep slopes are rocky: snow and grass do not stay there
  const float slope = 1.0 - geomN.y;
  const float rocky = smoothstep(0.18, 0.38, slope) * params.detail.z;
  const float moved = (w[1] + w[3] * 0.8 + w[0] * 0.4) * rocky;
  w[1] *= 1.0 - rocky;
  w[3] *= 1.0 - 0.8 * rocky;
  w[0] *= 1.0 - 0.4 * rocky;
  w[2] += moved;

  // --- Material sampling ------------------------------------------------------------------------
  const vec2 texUv = wPos.xz / params.detail.x;
  const float du = 1.0 / float(textureSize(roughHeightArray, 0).x);

  vec3 albedo[LAYER_COUNT];
  float rough[LAYER_COUNT];
  float disp[LAYER_COUNT];
  vec2 dispGrad[LAYER_COUNT];
  for (int i = 0; i < LAYER_COUNT; ++i)
  {
    albedo[i] = texture(albedoArray, vec3(texUv, i)).rgb;
    const vec2 rh = texture(roughHeightArray, vec3(texUv, i)).rg;
    rough[i] = rh.r;
    disp[i] = rh.g;
    const float hx = texture(roughHeightArray, vec3(texUv + vec2(du, 0), i)).g;
    const float hy = texture(roughHeightArray, vec3(texUv + vec2(0, du), i)).g;
    dispGrad[i] = vec2(hx - rh.g, hy - rh.g);
  }

  // Height-based blending: the layer whose displacement "sticks out" wins at transitions
  const float sharp = params.detail.w;
  float wSum = 0.0;
  for (int i = 0; i < LAYER_COUNT; ++i)
  {
    w[i] = max(w[i], 0.0) * pow(disp[i] + 0.05, sharp);
    wSum += w[i];
  }
  for (int i = 0; i < LAYER_COUNT; ++i)
    w[i] /= max(wSum, 1e-5);

  Surface s;
  s.albedo = vec3(0.0);
  s.roughness = 0.0;
  s.metallic = 0.0;
  vec2 grad = vec2(0.0);
  float height = 0.0;
  for (int i = 0; i < LAYER_COUNT; ++i)
  {
    s.albedo += w[i] * albedo[i];
    s.roughness += w[i] * rough[i];
    grad += w[i] * dispGrad[i];
    height += w[i] * disp[i];
  }
  s.roughness = clamp(s.roughness * params.debug.z, 0.04, 1.0);

  // Cavity occlusion: low points of the displacement maps (cracks between stones etc.)
  // receive less ambient light. Strength 0 reproduces the forward renderer of the previous task.
  s.ao = mix(1.0, clamp(0.35 + 1.3 * height, 0.0, 1.0), params.debug.w);

  // --- Detail normal from the Height maps (bump mapping, no geometry change) --------------------
  s.normal = geomN;
  if (params.debug.y > 0.5)
  {
    // Height texture gradient in "meters of bump per meter of surface"
    const float bumpDepth = 0.06 * params.detail.y;
    const vec2 slopeXZ = grad * bumpDepth / (du * params.detail.x);
    const vec3 T = normalize(vec3(1, 0, 0) - geomN * geomN.x);
    const vec3 B = normalize(vec3(0, 0, 1) - geomN * geomN.z);
    s.normal = normalize(geomN - slopeXZ.x * T - slopeXZ.y * B);
  }

  // Terrain-specific debug views are routed through the albedo channel
  const int mode = int(params.debug.x + 0.5);
  if (mode == DEBUG_VIEW_LAYERS)
    s.albedo = vec3(w[0] + w[3], w[1] + w[3], w[2] + w[3]);
  else if (mode == DEBUG_VIEW_HEIGHT)
    s.albedo = vec3(hNorm);

  return s;
}

#endif // TERRAIN_SURFACE_GLSL_INCLUDED
