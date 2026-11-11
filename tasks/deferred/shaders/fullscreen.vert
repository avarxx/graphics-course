#version 450
#extension GL_ARB_separate_shader_objects : enable

// One triangle that covers the whole screen, no vertex buffer needed:
// vertices (-1,-1), (3,-1), (-1,3) in NDC.

layout(location = 0) out vec2 vUv;

out gl_PerVertex { vec4 gl_Position; };

void main()
{
  const vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  vUv = uv;
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
