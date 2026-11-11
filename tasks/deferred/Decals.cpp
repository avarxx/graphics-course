#include "Decals.hpp"

#include <array>
#include <cmath>
#include <cstring>

#include <etna/GlobalContext.hpp>
#include <glm/ext.hpp>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include "HeightmapGenerator.hpp"
#include "TextureLoader.hpp"


namespace
{

// ---------------------------------------------------------------------------------------------
// Procedural decal textures. Generated on the CPU at startup, so the decals work without any
// downloaded assets. Every kind provides albedo + coverage mask, a height field (turned into a
// tangent-space normal map) and roughness.
// ---------------------------------------------------------------------------------------------

constexpr std::uint32_t DECAL_TEX_SIZE = 512;

float smooth01(float edge0, float edge1, float x)
{
  return glm::smoothstep(edge0, edge1, x);
}

float lattice(int x, int y, std::uint32_t seed)
{
  return prng::to01(prng::hash(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), seed));
}

float valueNoise(glm::vec2 p, std::uint32_t seed)
{
  const glm::vec2 i = glm::floor(p);
  const glm::vec2 f = p - i;
  const glm::vec2 u = f * f * (3.0f - 2.0f * f);
  const int x = static_cast<int>(i.x);
  const int y = static_cast<int>(i.y);
  const float a = lattice(x, y, seed);
  const float b = lattice(x + 1, y, seed);
  const float c = lattice(x, y + 1, seed);
  const float d = lattice(x + 1, y + 1, seed);
  return glm::mix(glm::mix(a, b, u.x), glm::mix(c, d, u.x), u.y);
}

float fbm(glm::vec2 p, std::uint32_t seed, int octaves = 5)
{
  float sum = 0.0f;
  float amp = 0.5f;
  float norm = 0.0f;
  for (int i = 0; i < octaves; ++i)
  {
    sum += amp * valueNoise(p, seed + static_cast<std::uint32_t>(i) * 101u);
    norm += amp;
    p = p * 2.03f + glm::vec2(17.1f, 9.3f);
    amp *= 0.5f;
  }
  return sum / norm;
}

// Distance to the nearest Voronoi edge (F2 - F1), used for mud cracks
float voronoiEdge(glm::vec2 p, std::uint32_t seed)
{
  const glm::vec2 cell = glm::floor(p);
  float f1 = 1e9f;
  float f2 = 1e9f;
  for (int j = -1; j <= 1; ++j)
    for (int i = -1; i <= 1; ++i)
    {
      const glm::vec2 c = cell + glm::vec2(static_cast<float>(i), static_cast<float>(j));
      const int cx = static_cast<int>(c.x);
      const int cy = static_cast<int>(c.y);
      const glm::vec2 site = c + glm::vec2(lattice(cx, cy, seed), lattice(cx, cy, seed + 7u));
      const float d = glm::length(p - site);
      if (d < f1)
      {
        f2 = f1;
        f1 = d;
      }
      else if (d < f2)
        f2 = d;
    }
  return f2 - f1;
}

struct DecalTexel
{
  glm::vec3 albedo{0.5f}; // linear
  float mask = 0.0f;
  float height = 0.0f;
  float roughness = 0.8f;
};

// Irregular round blob: 1 in the middle, soft noisy falloff towards radius ~0.45 (in uv)
float blob(glm::vec2 uv, std::uint32_t seed, float radius, float softness, float wobble)
{
  const glm::vec2 c = uv - 0.5f;
  const float r = glm::length(c) / radius;
  const float n = fbm(uv * 3.0f, seed) - 0.5f;
  return 1.0f - smooth01(1.0f - softness, 1.0f, r + wobble * n);
}

DecalTexel evalDecal(DecalKind kind, glm::vec2 uv)
{
  DecalTexel t;
  const glm::vec2 c = uv - 0.5f;
  const float r = glm::length(c) * 2.0f;

  switch (kind)
  {
  case DecalKind::Puddle: {
    // Standing water: very smooth, dark, with a wet (darker, semi-rough) rim around it
    const float water = blob(uv, 11u, 0.36f, 0.06f, 0.55f);
    const float wet = blob(uv, 11u, 0.46f, 0.35f, 0.55f);
    t.mask = wet;
    const float ripple = fbm(uv * 40.0f, 3u, 3);
    t.albedo = glm::mix(glm::vec3(0.07f, 0.06f, 0.05f), glm::vec3(0.025f, 0.035f, 0.04f), water);
    t.roughness = glm::mix(0.38f, 0.03f, water);
    t.height = water * 0.02f * ripple;
    break;
  }
  case DecalKind::CrackedMud: {
    // Dry clay plates separated by dark cracks (Voronoi edges)
    const glm::vec2 p = uv * 9.0f + glm::vec2(fbm(uv * 5.0f, 21u), fbm(uv * 5.0f, 22u)) * 0.8f;
    const float edge = voronoiEdge(p, 5u);
    const float crack = 1.0f - smooth01(0.02f, 0.09f, edge);
    const float grain = fbm(uv * 60.0f, 23u, 3);
    t.mask = blob(uv, 24u, 0.44f, 0.4f, 0.7f);
    t.albedo = glm::mix(
      glm::vec3(0.50f, 0.40f, 0.28f) * (0.85f + 0.3f * grain),
      glm::vec3(0.12f, 0.09f, 0.07f),
      crack);
    // Plates curl up at the edges a little
    t.height = (1.0f - crack) * (0.6f + 0.4f * smooth01(0.0f, 0.25f, edge)) + 0.1f * grain;
    t.roughness = glm::mix(0.88f, 1.0f, crack);
    break;
  }
  case DecalKind::Moss: {
    const float patches = fbm(uv * 4.0f, 31u);
    const float detail = fbm(uv * 48.0f, 32u, 4);
    t.mask = blob(uv, 33u, 0.45f, 0.5f, 0.8f) * smooth01(0.38f, 0.55f, patches + 0.25f * detail);
    t.albedo = glm::mix(glm::vec3(0.05f, 0.12f, 0.02f), glm::vec3(0.22f, 0.30f, 0.05f), detail);
    t.albedo =
      glm::mix(t.albedo, glm::vec3(0.35f, 0.33f, 0.08f), smooth01(0.7f, 0.9f, detail) * 0.6f);
    t.height = detail;
    t.roughness = 0.95f;
    break;
  }
  case DecalKind::Scorch: {
    // Burn mark: black center with radial soot streaks
    const float angle = std::atan2(c.y, c.x);
    const float streaks = fbm(glm::vec2(angle * 6.0f, r * 2.0f), 41u, 4);
    const float burn = 1.0f - smooth01(0.15f, 0.95f, r + 0.4f * (streaks - 0.5f));
    t.mask = burn;
    t.albedo = glm::mix(
      glm::vec3(0.10f, 0.07f, 0.05f),
      glm::vec3(0.012f, 0.011f, 0.010f),
      smooth01(0.2f, 0.8f, burn));
    t.height = 0.3f * fbm(uv * 30.0f, 42u, 3) * burn;
    t.roughness = 0.97f;
    break;
  }
  case DecalKind::PaintMark:
  default: {
    // Worn painted landing marker: ring + cross
    const float ring = 1.0f - smooth01(0.0f, 0.012f, std::abs(r - 0.78f) - 0.05f);
    const float crossH =
      1.0f - smooth01(0.0f, 0.012f, std::max(std::abs(c.y) - 0.035f, std::abs(c.x) - 0.3f));
    const float crossV =
      1.0f - smooth01(0.0f, 0.012f, std::max(std::abs(c.x) - 0.035f, std::abs(c.y) - 0.3f));
    const float paint = std::max(ring, std::max(crossH, crossV));
    const float wear = smooth01(0.3f, 0.45f, fbm(uv * 14.0f, 51u));
    t.mask = paint * wear;
    t.albedo = glm::vec3(0.80f, 0.78f, 0.72f) * (0.9f + 0.1f * fbm(uv * 80.0f, 52u, 2));
    t.height = paint;
    t.roughness = 0.45f;
    break;
  }
  }

  // Nothing may leak outside the texture: the box ends there
  const float border =
    smooth01(0.0f, 0.03f, std::min(std::min(uv.x, uv.y), std::min(1.0f - uv.x, 1.0f - uv.y)));
  t.mask = glm::clamp(t.mask * border, 0.0f, 1.0f);
  return t;
}

std::uint8_t to8(float v)
{
  return static_cast<std::uint8_t>(glm::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

float linearToSrgb(float x)
{
  x = glm::clamp(x, 0.0f, 1.0f);
  return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

// Bump strength (in "texels of height per texel") for each kind
constexpr std::array<float, static_cast<std::size_t>(DecalKind::Count)> BUMP_STRENGTH{
  6.0f, 14.0f, 10.0f, 4.0f, 3.0f};

void buildDecalLayer(DecalKind kind, ImageRGBA8& albedo, ImageRGBA8& normalRough)
{
  constexpr std::uint32_t N = DECAL_TEX_SIZE;
  albedo = tex::make_solid(N, N, {0, 0, 0, 0});
  normalRough = tex::make_solid(N, N, {128, 128, 255, 255});

  std::vector<DecalTexel> texels(static_cast<std::size_t>(N) * N);
  for (std::uint32_t y = 0; y < N; ++y)
    for (std::uint32_t x = 0; x < N; ++x)
    {
      const float invN = 1.0f / static_cast<float>(N);
      const glm::vec2 uv{
        (static_cast<float>(x) + 0.5f) * invN, (static_cast<float>(y) + 0.5f) * invN};
      texels[static_cast<std::size_t>(y) * N + x] = evalDecal(kind, uv);
    }

  const float strength = BUMP_STRENGTH[static_cast<std::size_t>(kind)];
  auto heightAt = [&](int x, int y) {
    x = glm::clamp(x, 0, static_cast<int>(N) - 1);
    y = glm::clamp(y, 0, static_cast<int>(N) - 1);
    return texels[static_cast<std::size_t>(y) * N + static_cast<std::size_t>(x)].height;
  };

  for (std::uint32_t y = 0; y < N; ++y)
    for (std::uint32_t x = 0; x < N; ++x)
    {
      const auto& t = texels[static_cast<std::size_t>(y) * N + x];
      albedo.at(x, y) = {
        to8(linearToSrgb(t.albedo.r)),
        to8(linearToSrgb(t.albedo.g)),
        to8(linearToSrgb(t.albedo.b)),
        to8(t.mask)};

      // Tangent space: +x along u, +y along v (texture rows), +z out of the surface
      const int ix = static_cast<int>(x);
      const int iy = static_cast<int>(y);
      const float dhdu = (heightAt(ix + 1, iy) - heightAt(ix - 1, iy)) * 0.5f * strength;
      const float dhdv = (heightAt(ix, iy + 1) - heightAt(ix, iy - 1)) * 0.5f * strength;
      const glm::vec3 n = glm::normalize(glm::vec3(-dhdu, -dhdv, 1.0f));
      normalRough.at(x, y) = {
        to8(n.x * 0.5f + 0.5f), to8(n.y * 0.5f + 0.5f), to8(n.z * 0.5f + 0.5f), to8(t.roughness)};
    }
}

const char* KIND_NAMES[] = {"Puddle", "Cracked mud", "Moss", "Scorch mark", "Paint marker"};

float terrainHeight(const TerrainInfo& terrain, glm::vec2 xz)
{
  if (terrain.heights == nullptr || terrain.heights->empty())
    return 0.0f;
  const glm::vec2 uv = glm::clamp(xz / terrain.worldSize + 0.5f, glm::vec2(0.0f), glm::vec2(1.0f));
  return HeightmapGenerator::sample(*terrain.heights, uv) * terrain.heightScale;
}

// Steepness of the terrain at xz: 0 = flat, 1 = vertical. Underwater spots are rejected.
float terrainSlope(const TerrainInfo& terrain, glm::vec2 xz)
{
  constexpr float STEP = 2.0f;
  const float hx0 = terrainHeight(terrain, xz - glm::vec2(STEP, 0));
  const float hx1 = terrainHeight(terrain, xz + glm::vec2(STEP, 0));
  const float hz0 = terrainHeight(terrain, xz - glm::vec2(0, STEP));
  const float hz1 = terrainHeight(terrain, xz + glm::vec2(0, STEP));
  const glm::vec3 n = glm::normalize(glm::vec3(hx0 - hx1, 2.0f * STEP, hz0 - hz1));
  const bool dry = terrainHeight(terrain, xz) > (terrain.seaLevel + 0.015f) * terrain.heightScale;
  return dry ? 1.0f - n.y : 10.0f;
}

// Flattest dry spot in a square around xz: decals look best on gentle ground
glm::vec2 findFlatSpot(const TerrainInfo& terrain, glm::vec2 xz, float radius)
{
  glm::vec2 best = xz;
  float bestScore = 1e9f;
  constexpr int N = 8;
  for (int j = -N; j <= N; ++j)
    for (int i = -N; i <= N; ++i)
    {
      const glm::vec2 p = xz +
        glm::vec2(static_cast<float>(i), static_cast<float>(j)) * (radius / static_cast<float>(N));
      // Prefer flat ground, but do not wander away from the requested spot without a reason
      const float score = terrainSlope(terrain, p) + 0.05f * glm::length(p - xz) / radius;
      if (score < bestScore)
      {
        bestScore = score;
        best = p;
      }
    }
  return best;
}

} // namespace

void DecalSystem::load(etna::OneShotCmdMgr& cmd_mgr)
{
  constexpr auto KINDS = static_cast<std::size_t>(DecalKind::Count);
  std::vector<ImageRGBA8> albedoLayers(KINDS);
  std::vector<ImageRGBA8> normalLayers(KINDS);
  for (std::size_t k = 0; k < KINDS; ++k)
    buildDecalLayer(static_cast<DecalKind>(k), albedoLayers[k], normalLayers[k]);

  albedoArray =
    tex::create_texture(cmd_mgr, "decal_albedo", vk::Format::eR8G8B8A8Srgb, albedoLayers);
  normalArray =
    tex::create_texture(cmd_mgr, "decal_normal_rough", vk::Format::eR8G8B8A8Unorm, normalLayers);

  auto& ctx = etna::get_context();
  buffer.emplace(ctx.getMainWorkCount(), [&ctx](std::size_t i) {
    return ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = sizeof(DecalGpu) * DECAL_MAX_COUNT,
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .allocationCreate =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .name = fmt::format("decals{}", i),
    });
  });

  placeDefaults();
}

void DecalSystem::placeDefaults()
{
  // Decals in front of the starting camera. The island depends on the seed, so the positions
  // below are only hints: on the first update every decal moves to the flattest dry spot nearby.
  decals.clear();
  snapPending = true;
  auto add = [this](DecalKind kind, glm::vec2 xz, glm::vec2 size, float yaw) {
    Decal d;
    d.kind = kind;
    d.xz = xz;
    d.size = size;
    d.yaw = yaw;
    decals.push_back(d);
  };
  add(DecalKind::PaintMark, {0.0f, 200.0f}, {30.0f, 30.0f}, 0.0f);
  add(DecalKind::Puddle, {-70.0f, 210.0f}, {26.0f, 20.0f}, 30.0f);
  add(DecalKind::Puddle, {70.0f, 230.0f}, {18.0f, 15.0f}, 110.0f);
  add(DecalKind::CrackedMud, {110.0f, 160.0f}, {40.0f, 32.0f}, 15.0f);
  add(DecalKind::Moss, {-120.0f, 150.0f}, {42.0f, 42.0f}, 70.0f);
  add(DecalKind::Scorch, {30.0f, 140.0f}, {26.0f, 26.0f}, 0.0f);
  add(DecalKind::CrackedMud, {-150.0f, 60.0f}, {50.0f, 50.0f}, 45.0f);
  add(DecalKind::Moss, {150.0f, 80.0f}, {45.0f, 35.0f}, 160.0f);
}

void DecalSystem::scatterRandom(const TerrainInfo& terrain, int count, std::uint32_t seed)
{
  for (int i = 0; i < count && decals.size() < DECAL_MAX_COUNT; ++i)
  {
    auto rnd = [&](std::uint32_t k) {
      return prng::to01(prng::hash(static_cast<std::uint32_t>(i), seed, k));
    };
    Decal d;
    // Several random candidates, keep the flattest dry one
    float bestSlope = 1e9f;
    for (std::uint32_t attempt = 0; attempt < 12; ++attempt)
    {
      const glm::vec2 xz =
        (glm::vec2(rnd(1 + attempt * 2), rnd(2 + attempt * 2)) - 0.5f) * terrain.worldSize * 0.8f;
      const float slope = terrainSlope(terrain, xz);
      if (slope < bestSlope)
      {
        bestSlope = slope;
        d.xz = xz;
      }
    }
    d.kind = static_cast<DecalKind>(
      prng::hash(static_cast<std::uint32_t>(i), seed, 99u) %
      static_cast<std::uint32_t>(DecalKind::Count));
    const float s = glm::mix(10.0f, 40.0f, rnd(30));
    d.size = glm::vec2(s, s * glm::mix(0.7f, 1.3f, rnd(31)));
    d.yaw = rnd(32) * 360.0f;
    d.opacity = glm::mix(0.6f, 1.0f, rnd(33));
    decals.push_back(d);
  }
}

void DecalSystem::update(const TerrainInfo& terrain)
{
  if (snapPending && terrain.heights != nullptr && !terrain.heights->empty())
  {
    for (auto& d : decals)
    {
      d.xz = findFlatSpot(terrain, d.xz, 45.0f);
      spdlog::debug(
        "Decal {} at ({:.1f}, {:.1f}, {:.1f})",
        KIND_NAMES[static_cast<int>(d.kind)],
        d.xz.x,
        terrainHeight(terrain, d.xz),
        d.xz.y);
    }
    snapPending = false;
  }

  auto* dst = reinterpret_cast<DecalGpu*>(buffer->get().data());
  activeCount = 0;
  if (!enabled)
    return;

  for (const auto& d : decals)
  {
    if (!d.enabled || activeCount >= DECAL_MAX_COUNT)
      continue;

    const glm::vec3 center{d.xz.x, terrainHeight(terrain, d.xz), d.xz.y};
    glm::mat4 boxToWorld = glm::translate(glm::mat4(1.0f), center);
    boxToWorld = glm::rotate(boxToWorld, glm::radians(d.yaw), glm::vec3(0, 1, 0));
    boxToWorld = glm::scale(boxToWorld, glm::vec3(d.size.x, d.depth, d.size.y));

    dst[activeCount++] = DecalGpu{
      .boxToWorld = boxToWorld,
      .worldToBox = glm::inverse(boxToWorld),
      .params0 =
        glm::vec4(static_cast<float>(d.kind), d.opacity, d.albedoIntensity, d.normalIntensity),
      .params1 = glm::vec4(d.roughnessIntensity, angleFade, 0.0f, 0.0f),
      .tint = glm::vec4(d.tint, 1.0f),
    };
  }
}

void DecalSystem::drawGui(
  const TerrainInfo& terrain, glm::vec3 camera_pos, glm::vec3 camera_forward)
{
  ImGui::Checkbox("Enable decals", &enabled);
  ImGui::SliderFloat("Global opacity", &globalOpacity, 0.0f, 1.0f);
  ImGui::SliderFloat("Slope fade (cos)", &angleFade, -1.0f, 0.95f);
  ImGui::Checkbox("Show decal boxes", &showBoxes);
  ImGui::Text("Active decals: %u / %zu", activeCount, decals.size());

  if (ImGui::Button("Add in front of camera") && decals.size() < DECAL_MAX_COUNT)
  {
    Decal d;
    glm::vec2 fwd{camera_forward.x, camera_forward.z};
    fwd = glm::length(fwd) > 1e-4f ? glm::normalize(fwd) : glm::vec2(0, -1);
    d.xz = glm::vec2(camera_pos.x, camera_pos.z) + fwd * 40.0f;
    d.kind = static_cast<DecalKind>(decals.size() % static_cast<std::size_t>(DecalKind::Count));
    decals.push_back(d);
  }
  ImGui::SameLine();
  if (ImGui::Button("Reset"))
    placeDefaults();
  ImGui::SameLine();
  if (ImGui::Button("Clear"))
    decals.clear();

  ImGui::SliderInt("Scatter count", &scatterCount, 1, 200);
  ImGui::InputInt("Scatter seed", &scatterSeed);
  if (ImGui::Button("Scatter random decals on the island"))
    scatterRandom(terrain, scatterCount, static_cast<std::uint32_t>(scatterSeed));

  int toRemove = -1;
  for (std::size_t i = 0; i < decals.size(); ++i)
  {
    auto& d = decals[i];
    ImGui::PushID(static_cast<int>(i));
    const bool open = ImGui::TreeNode("decal", "#%zu %s", i, KIND_NAMES[static_cast<int>(d.kind)]);
    ImGui::SameLine();
    ImGui::Checkbox("##on", &d.enabled);
    if (open)
    {
      int kind = static_cast<int>(d.kind);
      if (ImGui::Combo("Kind", &kind, KIND_NAMES, IM_ARRAYSIZE(KIND_NAMES)))
        d.kind = static_cast<DecalKind>(kind);
      ImGui::DragFloat2("Position XZ", &d.xz.x, 0.5f);
      ImGui::SliderFloat("Rotation", &d.yaw, 0.0f, 360.0f);
      ImGui::DragFloat2("Size", &d.size.x, 0.25f, 1.0f, 300.0f);
      ImGui::DragFloat("Box height", &d.depth, 0.25f, 1.0f, 200.0f);
      ImGui::SliderFloat("Opacity", &d.opacity, 0.0f, 1.0f);
      ImGui::SliderFloat("Albedo intensity", &d.albedoIntensity, 0.0f, 1.0f);
      ImGui::SliderFloat("Normal intensity", &d.normalIntensity, 0.0f, 2.0f);
      ImGui::SliderFloat("Roughness intensity", &d.roughnessIntensity, 0.0f, 1.0f);
      ImGui::ColorEdit3("Tint", &d.tint.x);
      if (ImGui::Button("Remove"))
        toRemove = static_cast<int>(i);
      ImGui::TreePop();
    }
    ImGui::PopID();
  }
  if (toRemove >= 0)
    decals.erase(decals.begin() + toRemove);
}
