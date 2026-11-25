#ifndef SURFACE_GLSL_INCLUDED
#define SURFACE_GLSL_INCLUDED

// Everything the lighting needs to know about a point of an opaque surface.
// This is exactly what goes into the G-buffer (plus depth, from which the position is restored).
struct Surface
{
  vec3 albedo;     // linear base color
  vec3 normal;     // world space, normalized
  float roughness; // perceptual roughness
  float metallic;
  float ao;        // ambient occlusion
};

#endif // SURFACE_GLSL_INCLUDED
