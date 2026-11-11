#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "rock_common.glsl"

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNormal;

struct RockInstance
{
  mat4 model;
  vec4 color;
};

// Instance transforms of all visible rocks, grouped by model
layout(std430, binding = 0) readonly buffer Instances
{
  RockInstance instances[];
};

layout(location = 0) out VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 color;
} vOut;

// Shared by the depth prepass and the G-buffer pass (depth test EQUAL), see terrain.vert
out gl_PerVertex { invariant vec4 gl_Position; };

void main()
{
  const RockInstance inst = instances[uint(pc.rock.x) + gl_InstanceIndex];
  vOut.wPos = (inst.model * vec4(vPos, 1.0)).xyz;
  vOut.wNorm = normalize(mat3(inst.model) * vNormal);
  vOut.color = inst.color.rgb;
  gl_Position = params.viewProj * vec4(vOut.wPos, 1.0);
}
