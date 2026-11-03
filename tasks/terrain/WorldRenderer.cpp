#include "WorldRenderer.hpp"

#include <chrono>
#include <array>
#include <cstring>

#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <glm/ext.hpp>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include "TextureLoader.hpp"


WorldRenderer::WorldRenderer() = default;

void WorldRenderer::allocateResources(glm::uvec2 swapchain_resolution)
{
  resolution = swapchain_resolution;

  auto& ctx = etna::get_context();

  mainViewDepth = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
  });
}

void WorldRenderer::loadResources()
{
  auto& ctx = etna::get_context();

  oneShotCmdMgr = ctx.createOneShotCmdMgr();

  heightSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eNearest,
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .mipmapMode = vk::SamplerMipmapMode::eNearest,
    .name = "height_sampler",
  });
  materialSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .mipmapMode = vk::SamplerMipmapMode::eLinear,
    .name = "material_sampler",
  });

  heightMap = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{HeightmapGenerator::RESOLUTION, HeightmapGenerator::RESOLUTION, 1},
    .name = "height_map",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
  });

  terrainUbo.emplace(ctx.getMainWorkCount(), [&ctx](std::size_t i) {
    return ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = sizeof(TerrainParams),
      .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .allocationCreate =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .name = fmt::format("terrain_params{}", i),
    });
  });

  createTerrainGrid();
  loadTerrainTextures();
  regenerateHeightmap();
}

void WorldRenderer::createTerrainGrid()
{
  constexpr std::uint32_t N = GRID_CELLS;
  constexpr std::uint32_t VERTS = N + 1;

  std::vector<glm::vec2> vertices;
  vertices.reserve(static_cast<std::size_t>(VERTS) * VERTS);
  for (std::uint32_t j = 0; j < VERTS; ++j)
    for (std::uint32_t i = 0; i < VERTS; ++i)
      vertices.emplace_back(static_cast<float>(i) / N, static_cast<float>(j) / N);

  std::vector<std::uint32_t> indices;
  indices.reserve(static_cast<std::size_t>(N) * N * 6);
  for (std::uint32_t j = 0; j < N; ++j)
    for (std::uint32_t i = 0; i < N; ++i)
    {
      const std::uint32_t a = j * VERTS + i;
      const std::uint32_t b = a + VERTS; // +z
      const std::uint32_t c = a + 1;     // +x
      const std::uint32_t d = b + 1;     // +x +z
      indices.insert(indices.end(), {a, b, c, c, b, d});
    }
  gridIndexCount = static_cast<std::uint32_t>(indices.size());

  auto& ctx = etna::get_context();
  gridVertices = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = vertices.size() * sizeof(glm::vec2),
    .bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = "terrain_grid_vertices",
  });
  gridIndices = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = indices.size() * sizeof(std::uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = "terrain_grid_indices",
  });

  etna::BlockingTransferHelper transfer(etna::BlockingTransferHelper::CreateInfo{
    .stagingSize = indices.size() * sizeof(std::uint32_t),
  });
  transfer.uploadBuffer<glm::vec2>(*oneShotCmdMgr, gridVertices, 0, vertices);
  transfer.uploadBuffer<std::uint32_t>(*oneShotCmdMgr, gridIndices, 0, indices);
}

void WorldRenderer::loadTerrainTextures()
{
  struct LayerDesc
  {
    const char* asset;
    glm::u8vec4 fallbackAlbedo;
    std::uint8_t fallbackRoughness;
  };
  const std::array<LayerDesc, 4> layerDescs{{
    {"sand_03", {194, 178, 128, 255}, 230},
    {"sparse_grass", {86, 110, 50, 255}, 220},
    {"rock_06", {110, 104, 98, 255}, 180},
    {"snow_01", {235, 240, 245, 255}, 120},
  }};

  constexpr std::uint32_t SIZE = 1024;
  const std::filesystem::path root = TERRAIN_ASSETS_ROOT;

  std::vector<ImageRGBA8> albedoLayers;
  std::vector<ImageRGBA8> roughHeightLayers;
  loadedTextureLayers = 0;

  for (const auto& desc : layerDescs)
  {
    const auto dir = root / desc.asset;

    auto loadMap = [&](const char* map) -> std::optional<ImageRGBA8> {
      auto path = tex::find_image(dir, map);
      if (!path)
        return std::nullopt;
      auto img = tex::load_rgba8(*path);
      if (img)
        *img = tex::resize(*img, SIZE, SIZE);
      return img;
    };

    auto albedo = loadMap("Diffuse");
    auto rough = loadMap("Rough");
    auto height = loadMap("Displacement");

    if (!albedo || !rough || !height)
      spdlog::warn(
        "Terrain layer '{}' is missing some textures in {}, using fallback values. "
        "Re-run CMake configure with internet access to download them.",
        desc.asset,
        dir.string());
    else
      ++loadedTextureLayers;

    albedoLayers.push_back(albedo ? std::move(*albedo) : tex::make_solid(SIZE, SIZE, desc.fallbackAlbedo));

    ImageRGBA8 rh = tex::make_solid(SIZE, SIZE, {desc.fallbackRoughness, 128, 0, 255});
    for (std::uint32_t y = 0; y < SIZE; ++y)
      for (std::uint32_t x = 0; x < SIZE; ++x)
      {
        if (rough)
          rh.at(x, y).r = rough->at(x, y).r;
        if (height)
          rh.at(x, y).g = height->at(x, y).r;
      }
    roughHeightLayers.push_back(std::move(rh));
  }

  albedoArray = tex::create_texture(
    *oneShotCmdMgr, "terrain_albedo", vk::Format::eR8G8B8A8Srgb, albedoLayers);
  roughHeightArray = tex::create_texture(
    *oneShotCmdMgr, "terrain_rough_height", vk::Format::eR8G8B8A8Unorm, roughHeightLayers);
}

void WorldRenderer::regenerateHeightmap()
{
  const auto start = std::chrono::steady_clock::now();
  heights = HeightmapGenerator::generate(heightmapParams);
  const auto end = std::chrono::steady_clock::now();
  lastGenerationMs = std::chrono::duration<float, std::milli>(end - start).count();

  // The heightmap may still be in use by frames in flight
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());

  tex::upload_raw(
    *oneShotCmdMgr,
    heightMap,
    std::as_bytes(std::span<const float>(heights.data(), heights.size())));

  heightmapDirty = false;
}

void WorldRenderer::loadShaders()
{
  etna::create_program(
    "terrain", {TERRAIN_SHADERS_ROOT "terrain.vert.spv", TERRAIN_SHADERS_ROOT "terrain.frag.spv"});
}

void WorldRenderer::setupPipelines(vk::Format swapchain_format)
{
  auto& pipelineManager = etna::get_context().getPipelineManager();

  etna::VertexShaderInputDescription gridInputDesc{
    .bindings = {etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription =
        etna::VertexByteStreamFormatDescription{
          .stride = sizeof(glm::vec2),
          .attributes = {etna::VertexByteStreamFormatDescription::Attribute{
            .format = vk::Format::eR32G32Sfloat,
            .offset = 0,
          }},
        },
    }},
  };

  terrainPipeline = {};
  terrainPipeline = pipelineManager.createGraphicsPipeline(
    "terrain",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = gridInputDesc,
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eNone,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {swapchain_format},
          .depthAttachmentFormat = vk::Format::eD32Sfloat,
        },
    });
}

void WorldRenderer::debugInput(const Keyboard& kb)
{
  if (kb[KeyboardKey::kG] == ButtonState::Falling)
    heightmapDirty = true;
}

glm::vec3 WorldRenderer::sunDirection() const
{
  const float az = glm::radians(sunAzimuth);
  const float el = glm::radians(sunElevation);
  return glm::normalize(
    glm::vec3{std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)});
}

void WorldRenderer::update(const FramePacket& packet)
{
  ZoneScoped;

  if (heightmapDirty)
    regenerateHeightmap();

  const float aspect = float(resolution.x) / float(resolution.y);
  worldViewProj = packet.mainCam.projTm(aspect) * packet.mainCam.viewTm();
  cameraPos = packet.mainCam.position;
  currentTime = packet.currentTime;

  terrainParams.viewProj = worldViewProj;
  terrainParams.cameraPos = glm::vec4(cameraPos, currentTime);
  terrainParams.sunDir = glm::vec4(sunDirection(), sunIntensity);
  terrainParams.sunColor = glm::vec4(sunColor, ambientIntensity);
  terrainParams.skyColor = glm::vec4(skyColor, exposure);
  terrainParams.groundColor = glm::vec4(groundAmbient, fogDensity);
  terrainParams.terrainSize = glm::vec4(
    worldSize, heightScale, seaLevel, static_cast<float>(HeightmapGenerator::RESOLUTION));
  terrainParams.layerHeights = layerHeights;
  terrainParams.detail =
    glm::vec4(textureTiling, detailStrength, slopeRock, heightBlendSharpness);
  terrainParams.debug = glm::vec4(
    static_cast<float>(debugView), detailNormals ? 1.0f : 0.0f, roughnessScale, 0.0f);

  // This frame's buffer is guaranteed to not be in use by the GPU anymore
  std::memcpy(terrainUbo->get().data(), &terrainParams, sizeof(terrainParams));
}

void WorldRenderer::renderTerrain(vk::CommandBuffer cmd_buf)
{
  auto programInfo = etna::get_shader_program("terrain");

  auto set = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmd_buf,
    {
      etna::Binding{0, terrainUbo->get().genBinding()},
      etna::Binding{
        1,
        heightMap.genBinding(heightSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{
        2,
        albedoArray.genBinding(
          materialSampler.get(),
          vk::ImageLayout::eShaderReadOnlyOptimal,
          {.type = vk::ImageViewType::e2DArray})},
      etna::Binding{
        3,
        roughHeightArray.genBinding(
          materialSampler.get(),
          vk::ImageLayout::eShaderReadOnlyOptimal,
          {.type = vk::ImageViewType::e2DArray})},
    });

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, terrainPipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics,
    terrainPipeline.getVkPipelineLayout(),
    0,
    {set.getVkSet()},
    {});

  cmd_buf.bindVertexBuffers(0, {gridVertices.get()}, {0});
  cmd_buf.bindIndexBuffer(gridIndices.get(), 0, vk::IndexType::eUint32);
  cmd_buf.drawIndexed(gridIndexCount, 1, 0, 0, 0);
}

void WorldRenderer::renderWorld(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd_buf, renderWorld);

  // Sky color for the clear, run through the same exposure + tonemapping as the terrain
  const glm::vec3 x = skyColor * exposure;
  const glm::vec3 sky = glm::clamp(
    (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), glm::vec3(0.0f), glm::vec3(1.0f));

  {
    ETNA_PROFILE_GPU(cmd_buf, renderTerrain);

    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {{
        .image = target_image,
        .view = target_image_view,
        .clearColorValue = std::array<float, 4>{sky.r, sky.g, sky.b, 1.0f},
      }},
      {.image = mainViewDepth.get(), .view = mainViewDepth.getView({})});

    renderTerrain(cmd_buf);
  }
}

void WorldRenderer::drawGui()
{
  ImGui::Begin("Terrain");

  ImGui::Text(
    "%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

  if (ImGui::CollapsingHeader("Heightmap (fBm noise)", ImGuiTreeNodeFlags_DefaultOpen))
  {
    bool changed = false;
    changed |= ImGui::InputInt("Seed", &heightmapParams.seed);
    changed |= ImGui::SliderInt("Octaves", &heightmapParams.octaves, 1, 12);
    changed |= ImGui::SliderFloat("Frequency", &heightmapParams.frequency, 0.5f, 12.0f);
    changed |= ImGui::SliderFloat("Lacunarity", &heightmapParams.lacunarity, 1.5f, 3.0f);
    changed |= ImGui::SliderFloat("Gain", &heightmapParams.gain, 0.2f, 0.8f);
    changed |= ImGui::SliderFloat("Ridges", &heightmapParams.ridge, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Domain warp", &heightmapParams.warp, 0.0f, 1.5f);
    changed |= ImGui::SliderFloat("Exponent", &heightmapParams.exponent, 0.5f, 4.0f);
    changed |= ImGui::SliderFloat("Island radius", &heightmapParams.islandRadius, 0.3f, 1.4f);
    changed |= ImGui::SliderFloat("Coast falloff", &heightmapParams.islandFalloff, 0.05f, 1.0f);

    ImGui::Checkbox("Regenerate automatically", &autoRegenerate);
    if (ImGui::Button("Regenerate (G)") || (changed && autoRegenerate))
      heightmapDirty = true;
    ImGui::SameLine();
    if (ImGui::Button("Random seed"))
    {
      heightmapParams.seed = static_cast<int>(prng::pcg(static_cast<std::uint32_t>(heightmapParams.seed)) & 0xFFFFF);
      heightmapDirty = true;
    }
    ImGui::Text("Generated 1024x1024 R32_SFLOAT in %.1f ms", lastGenerationMs);
  }

  if (ImGui::CollapsingHeader("Terrain shape", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::SliderFloat("World size (m)", &worldSize, 100.0f, 2000.0f);
    ImGui::SliderFloat("Height scale (m)", &heightScale, 5.0f, 300.0f);
    ImGui::SliderFloat("Sea level", &seaLevel, 0.0f, 0.5f);
  }

  if (ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::Text("Loaded PBR texture sets: %d / 4", loadedTextureLayers);
    ImGui::SliderFloat("Sand -> grass", &layerHeights.x, 0.0f, 1.0f);
    ImGui::SliderFloat("Grass -> rock", &layerHeights.y, 0.0f, 1.0f);
    ImGui::SliderFloat("Rock -> snow", &layerHeights.z, 0.0f, 1.0f);
    ImGui::SliderFloat("Blend width", &layerHeights.w, 0.001f, 0.2f);
    ImGui::SliderFloat("Height blend sharpness", &heightBlendSharpness, 0.0f, 16.0f);
    ImGui::SliderFloat("Slope -> rock", &slopeRock, 0.0f, 1.0f);
    ImGui::SliderFloat("Texture tile (m)", &textureTiling, 0.5f, 30.0f);
    ImGui::Checkbox("Detail normals from Height maps", &detailNormals);
    ImGui::SliderFloat("Detail strength", &detailStrength, 0.0f, 5.0f);
    ImGui::SliderFloat("Roughness scale", &roughnessScale, 0.1f, 2.0f);
    const char* views[] = {"Shaded", "Detail normals", "Geometry normals", "Layer weights", "Height"};
    ImGui::Combo("Debug view", &debugView, views, IM_ARRAYSIZE(views));
  }

  if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::SliderFloat("Sun azimuth", &sunAzimuth, 0.0f, 360.0f);
    ImGui::SliderFloat("Sun elevation", &sunElevation, 1.0f, 90.0f);
    ImGui::SliderFloat("Sun intensity", &sunIntensity, 0.0f, 10.0f);
    ImGui::ColorEdit3("Sun color", &sunColor.x);
    ImGui::SliderFloat("Ambient", &ambientIntensity, 0.0f, 2.0f);
    ImGui::ColorEdit3("Sky color", &skyColor.x);
    ImGui::ColorEdit3("Ground bounce", &groundAmbient.x);
    ImGui::SliderFloat("Exposure", &exposure, 0.1f, 4.0f);
    ImGui::SliderFloat("Fog density", &fogDensity, 0.0f, 0.005f, "%.5f");
  }

  ImGui::Separator();
  ImGui::TextWrapped(
    "WASD + R/F (E/Q): move, Shift: faster, RMB: toggle mouse look, G: regenerate, "
    "B: reload shaders");

  ImGui::End();
}
