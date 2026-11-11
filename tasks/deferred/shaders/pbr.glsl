#ifndef PBR_GLSL_INCLUDED
#define PBR_GLSL_INCLUDED

const float PI = 3.14159265359;

float distributionGGX(float NdotH, float roughness)
{
  const float a = roughness * roughness;
  const float a2 = a * a;
  const float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-6);
}

float geometrySchlickGGX(float NdotX, float roughness)
{
  const float r = roughness + 1.0;
  const float k = r * r / 8.0;
  return NdotX / (NdotX * (1.0 - k) + k);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Cook-Torrance BRDF for a single directional light, returns outgoing radiance
vec3 shadeDirectional(
  vec3 albedo, float roughness, float metallic, vec3 N, vec3 V, vec3 L, vec3 radiance)
{
  roughness = clamp(roughness, 0.04, 1.0);
  const vec3 H = normalize(V + L);
  const float NdotL = max(dot(N, L), 0.0);
  const float NdotV = max(dot(N, V), 1e-4);
  const float NdotH = max(dot(N, H), 0.0);

  const vec3 F0 = mix(vec3(0.04), albedo, metallic);
  const vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
  const float D = distributionGGX(NdotH, roughness);
  const float G = geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);

  const vec3 specular = D * G * F / max(4.0 * NdotV * NdotL, 1e-4);
  const vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

  return (kD * albedo / PI + specular) * radiance * NdotL;
}

// Simple hemispherical ambient lighting
vec3 shadeAmbient(vec3 albedo, float ao, vec3 N, vec3 sky, vec3 ground, float intensity)
{
  const vec3 irradiance = mix(ground, sky, N.y * 0.5 + 0.5);
  return albedo * irradiance * intensity * ao;
}

// Analytic approximation of the pre-integrated environment BRDF
// (Karis, "Physically Based Shading on Mobile", 2014)
vec3 envBrdfApprox(vec3 F0, float roughness, float NdotV)
{
  const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
  const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
  const vec4 r = roughness * c0 + c1;
  const float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
  const vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;
  return F0 * AB.x + AB.y;
}

// Specular reflection of the same sky/ground hemisphere. Without it perfectly smooth surfaces
// (e.g. puddle decals) only ever show the sun highlight and look black otherwise.
vec3 shadeAmbientSpecular(
  vec3 albedo, float roughness, float metallic, float ao, vec3 N, vec3 V,
  vec3 sky, vec3 ground, float intensity)
{
  roughness = clamp(roughness, 0.04, 1.0);
  const float NdotV = max(dot(N, V), 1e-4);
  // Rough surfaces reflect a blurry environment: lean the lookup direction towards the normal
  const vec3 R = normalize(mix(reflect(-V, N), N, roughness * roughness));
  const vec3 radiance = mix(ground, sky, smoothstep(-0.2, 0.2, R.y));
  const vec3 F0 = mix(vec3(0.04), albedo, metallic);
  return envBrdfApprox(F0, roughness, NdotV) * radiance * intensity * ao;
}

// Narkowicz ACES fit
vec3 tonemapACES(vec3 x)
{
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

#endif // PBR_GLSL_INCLUDED
