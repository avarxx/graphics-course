#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// Draws the bounding box of every decal (instanced, 36 vertices each, no vertex buffer).
// The fragment shader then decides which G-buffer pixels actually lie inside the box.

#include "decal_common.glsl"

layout(location = 0) flat out uint vDecal;

out gl_PerVertex { vec4 gl_Position; };

const vec3 CORNERS[8] = vec3[8](
  vec3(-0.5, -0.5, -0.5), vec3(0.5, -0.5, -0.5), vec3(0.5, 0.5, -0.5), vec3(-0.5, 0.5, -0.5),
  vec3(-0.5, -0.5, 0.5), vec3(0.5, -0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(-0.5, 0.5, 0.5));

const int INDICES[36] = int[36](
  0, 2, 1, 0, 3, 2,  // -z
  4, 5, 6, 4, 6, 7,  // +z
  0, 1, 5, 0, 5, 4,  // -y
  3, 6, 2, 3, 7, 6,  // +y
  0, 4, 7, 0, 7, 3,  // -x
  1, 2, 6, 1, 6, 5); // +x

void main()
{
  vDecal = gl_InstanceIndex;
  const vec3 corner = CORNERS[INDICES[gl_VertexIndex]];
  gl_Position = params.viewProj * (decals[gl_InstanceIndex].boxToWorld * vec4(corner, 1.0));
}
