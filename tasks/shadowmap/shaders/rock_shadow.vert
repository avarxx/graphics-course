#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// Depth-only pass for the sun shadow map: same instance buffer as the G-buffer pass,
// but projected with the light matrix.

#include "rock_common.glsl"

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNormal;

struct RockInstance
{
  mat4 model;
  vec4 color;
};

layout(std430, binding = 0) readonly buffer Instances
{
  RockInstance instances[];
};

out gl_PerVertex { vec4 gl_Position; };

void main()
{
  const RockInstance inst = instances[uint(pc.rock.x) + gl_InstanceIndex];
  const vec3 wPos = (inst.model * vec4(vPos, 1.0)).xyz;
  gl_Position = params.shadowViewProj * vec4(wPos, 1.0);
}
