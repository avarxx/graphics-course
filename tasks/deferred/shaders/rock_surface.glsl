#ifndef ROCK_SURFACE_GLSL_INCLUDED
#define ROCK_SURFACE_GLSL_INCLUDED

#include "rock_common.glsl"
#include "surface.glsl"

layout(location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 color;
} surf;

Surface evalRockSurface()
{
  Surface s;
  s.albedo = surf.color;
  s.normal = normalize(surf.wNorm);
  s.roughness = pc.rock.y;
  s.metallic = pc.rock.z;
  s.ao = 1.0;
  return s;
}

#endif // ROCK_SURFACE_GLSL_INCLUDED
