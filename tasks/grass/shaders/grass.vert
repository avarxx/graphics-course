#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// One grass blade is a narrow strip of GRASS_SEGMENTS quads. Every instance comes from the
// placement compute shader through a storage buffer; the strip is bent by the wind here, so the
// base stays glued to the terrain while the tip follows the wind wave.

#include "SceneParams.h"
#include "GrassParams.h"

layout(location = 0) in vec2 vLocal; // x = side (-1 / +1), y = height fraction [0, 1]

struct GrassInstance
{
  vec4 posHeight; // xyz = base world position, w = height
  vec4 shape;     // x = width, y = yaw (radians), z = lean, w = wind phase
  vec4 color;     // rgb = albedo (linear)
};

layout(std430, binding = 0) readonly buffer Instances
{
  GrassInstance instances[];
};

layout(binding = 1) uniform SceneUbo
{
  SceneParams params;
};

layout(location = 0) out VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 color;
  float t;
}
vOut;

out gl_PerVertex
{
  invariant vec4 gl_Position;
};

void main()
{
  const GrassInstance inst = instances[gl_InstanceIndex];

  const vec3 base = inst.posHeight.xyz;
  const float height = inst.posHeight.w;
  const float width = inst.shape.x;
  const float yaw = inst.shape.y;
  const float lean = inst.shape.z;
  const float phase = inst.shape.w;
  const float t = clamp(vLocal.y, 0.0, 1.0);

  // Camera-facing ribbon: the blade is always seen from its wide side, but the random yaw
  // twists it a little so that the field does not look like a fan of identical billboards.
  const vec3 toCamera = params.cameraPos.xyz - base;
  const vec3 up = vec3(0.0, 1.0, 0.0);
  vec3 right = cross(up, toCamera);
  right = length(right) > 1e-5 ? normalize(right) : vec3(1.0, 0.0, 0.0);
  const vec3 forward = normalize(cross(right, up));
  right = normalize(right * cos(yaw) + forward * sin(yaw));

  // Wind: two travelling waves along the wind direction, with a per-blade phase. The wave is
  // sampled in world space, so gusts visibly run over the field instead of every blade moving
  // in sync.
  const vec2 windDir = normalize(vec2(params.grassWind.x, params.grassWind.y));
  const float time = params.cameraPos.w;
  const float along = dot(base.xz, windDir);
  const float wave = 0.65 * sin(time * params.grassWind.w + phase + along * 0.11) +
    0.35 * sin(time * params.grassWind.w * 1.73 + phase * 0.7 + along * 0.037);
  const float bendAmount = params.grassWind.z * height * (0.25 + 0.75 * wave);

  const vec3 leanDir = vec3(cos(yaw * 3.0 + phase), 0.0, sin(yaw * 3.0 + phase));
  const float curve = t * t;

  // Base is fixed, tip bends: the bend grows quadratically with height along the blade.
  const vec3 offset = up * (t * height) +
    (vec3(windDir.x, 0.0, windDir.y) * bendAmount + leanDir * (lean * height)) * curve;

  // The strip narrows towards the tip, ending almost in a point.
  const float widthScale = 1.0 - 0.92 * t;
  const vec3 side = right * (vLocal.x * width * 0.5 * widthScale);

  vOut.wPos = base + side + offset;
  vOut.color = inst.color.rgb * mix(0.55, 1.0, t);
  vOut.t = t;

  // Tangent of the bent curve, used to build a normal that follows the blade
  const vec3 tangent = normalize(
    up * height +
    (vec3(windDir.x, 0.0, windDir.y) * bendAmount + leanDir * (lean * height)) * (2.0 * t));

  const vec3 n = cross(right, tangent);
  vOut.wNorm = length(n) > 1e-5 ? normalize(n) : up;

  gl_Position = params.viewProj * vec4(vOut.wPos, 1.0);
}
