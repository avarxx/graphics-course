#include "WorldRenderer.hpp"

#include <chrono>
#include <array>
#include <cstring>

#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <glm/ext.hpp>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include "TextureLoader.hpp"


/*
 * Frame structure (deferred mode):
 *
 *   1. Depth prepass       terrain + rocks, depth only (reverse-Z: GREATER, write)
 *   2. G-buffer pass       terrain + rocks, depth test EQUAL without writes ->
 * albedo/normal/material
 *   3. Decal pass          decal boxes, read depth, alpha-blend into albedo/normal/roughness
 *   4. Lighting pass       fullscreen triangle: G-buffer + depth -> world position -> Cook-Torrance
 *
 * The forward path of the previous task is kept as a reference (same material and lighting code),
 * so that the correctness of the deferred image can be checked by switching between them.
 */

WorldRenderer::WorldRenderer() = default;

void WorldRenderer::allocateResources(glm::uvec2 swapchain_resolution)
{
  resolution = swapchain_resolution;
  gbuffers[0].allocate(resolution, GBufferLayout::Compact);
  gbuffers[1].allocate(resolution, GBufferLayout::Naive);
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
  decalSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .mipmapMode = vk::SamplerMipmapMode::eLinear,
    .name = "decal_sampler",
  });
  // G-buffer is read with texelFetch, the sampler is only there to satisfy the descriptor type
  pointSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eNearest,
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .mipmapMode = vk::SamplerMipmapMode::eNearest,
    .name = "point_sampler",
  });

  heightMap = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{HeightmapGenerator::RESOLUTION, HeightmapGenerator::RESOLUTION, 1},
    .name = "height_map",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
  });

  sceneUbo.emplace(ctx.getMainWorkCount(), [&ctx](std::size_t i) {
    return ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = sizeof(SceneParams),
      .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .allocationCreate =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .name = fmt::format("scene_params{}", i),
    });
  });

  createTerrainGrid();
  loadTerrainTextures();
  loadRocks();
  decals.load(*oneShotCmdMgr);
  regenerateHeightmap();

  gpuTimer.init(ctx.getMainWorkCount().multiBufferingCount());
}

void WorldRenderer::loadShaders()
{
  const auto shader = [](const char* name) {
    return std::filesystem::path(DEFERRED_SHADERS_ROOT) / name;
  };
  // Depth prepass reuses the vertex shaders of the main passes (and they are `invariant`),
  // so the EQUAL depth test of the G-buffer pass passes exactly for the visible fragments.
  etna::create_program("terrain_depth", {shader("terrain.vert.spv")});
  etna::create_program("rock_depth", {shader("rock.vert.spv")});
  etna::create_program(
    "terrain_gbuffer", {shader("terrain.vert.spv"), shader("terrain_gbuffer.frag.spv")});
  etna::create_program("rock_gbuffer", {shader("rock.vert.spv"), shader("rock_gbuffer.frag.spv")});
  etna::create_program(
    "deferred_lighting", {shader("fullscreen.vert.spv"), shader("deferred_lighting.frag.spv")});
  etna::create_program("decal", {shader("decal.vert.spv"), shader("decal.frag.spv")});
  etna::create_program(
    "terrain_forward", {shader("terrain.vert.spv"), shader("terrain_forward.frag.spv")});
  etna::create_program("rock_forward", {shader("rock.vert.spv"), shader("rock_forward.frag.spv")});
}

void WorldRenderer::setupPipelines(vk::Format swapchain_format)
{
  auto& pipelineManager = etna::get_context().getPipelineManager();

  const etna::VertexShaderInputDescription gridInput{
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
  const etna::VertexShaderInputDescription rockInput{
    .bindings = {etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription =
        etna::VertexByteStreamFormatDescription{
          .stride = sizeof(RockVertex),
          .attributes =
            {
              {.format = vk::Format::eR32G32B32Sfloat, .offset = 0},
              {.format = vk::Format::eR32G32B32Sfloat, .offset = sizeof(glm::vec3)},
            },
        },
    }},
  };

  const vk::PipelineRasterizationStateCreateInfo noCull{
    .polygonMode = vk::PolygonMode::eFill,
    .cullMode = vk::CullModeFlagBits::eNone,
    .frontFace = vk::FrontFace::eCounterClockwise,
    .lineWidth = 1.f,
  };

  // Reverse-Z: closer fragments have GREATER depth
  const vk::PipelineDepthStencilStateCreateInfo depthCloserWrite{
    .depthTestEnable = vk::True,
    .depthWriteEnable = vk::True,
    .depthCompareOp = vk::CompareOp::eGreater,
    .maxDepthBounds = 1.f,
  };
  // After the prepass only the visible fragment of every pixel passes, so the expensive
  // terrain material is evaluated exactly once per pixel (no overdraw)
  const vk::PipelineDepthStencilStateCreateInfo depthEqualNoWrite{
    .depthTestEnable = vk::True,
    .depthWriteEnable = vk::False,
    .depthCompareOp = vk::CompareOp::eEqual,
    .maxDepthBounds = 1.f,
  };
  const vk::PipelineDepthStencilStateCreateInfo noDepth{
    .depthTestEnable = vk::False,
    .depthWriteEnable = vk::False,
    .depthCompareOp = vk::CompareOp::eAlways,
    .maxDepthBounds = 1.f,
  };

  const auto noBlend = [](std::size_t count) {
    etna::GraphicsPipeline::CreateInfo::Blending b{};
    b.logicOp = vk::LogicOp::eCopy;
    b.attachments.assign(
      count,
      vk::PipelineColorBlendAttachmentState{
        .blendEnable = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
      });
    return b;
  };

  // --- Depth prepass ------------------------------------------------------------------------
  terrainDepthPipeline = {};
  terrainDepthPipeline = pipelineManager.createGraphicsPipeline(
    "terrain_depth",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = gridInput,
      .rasterizationConfig = noCull,
      .blendingConfig = noBlend(0),
      .depthConfig = depthCloserWrite,
      .fragmentShaderOutput = {.depthAttachmentFormat = GBuffer::DEPTH_FORMAT},
    });
  rockDepthPipeline = {};
  rockDepthPipeline = pipelineManager.createGraphicsPipeline(
    "rock_depth",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = rockInput,
      .rasterizationConfig = noCull,
      .blendingConfig = noBlend(0),
      .depthConfig = depthCloserWrite,
      .fragmentShaderOutput = {.depthAttachmentFormat = GBuffer::DEPTH_FORMAT},
    });

  // --- G-buffer + decals, one set per layout --------------------------------------------------
  for (std::size_t l = 0; l < 2; ++l)
  {
    const auto layout = static_cast<GBufferLayout>(l);
    const auto formats = GBuffer::colorFormats(layout);
    const std::vector<vk::Format> colorFormats(formats.begin(), formats.end());

    for (std::size_t prepass = 0; prepass < 2; ++prepass)
    {
      const auto& depthCfg = prepass != 0 ? depthEqualNoWrite : depthCloserWrite;
      terrainGBufferPipelines[l][prepass] = {};
      terrainGBufferPipelines[l][prepass] = pipelineManager.createGraphicsPipeline(
        "terrain_gbuffer",
        etna::GraphicsPipeline::CreateInfo{
          .vertexShaderInput = gridInput,
          .rasterizationConfig = noCull,
          .blendingConfig = noBlend(GBuffer::COLOR_TARGETS),
          .depthConfig = depthCfg,
          .fragmentShaderOutput =
            {.colorAttachmentFormats = colorFormats,
             .depthAttachmentFormat = GBuffer::DEPTH_FORMAT},
        });
      rockGBufferPipelines[l][prepass] = {};
      rockGBufferPipelines[l][prepass] = pipelineManager.createGraphicsPipeline(
        "rock_gbuffer",
        etna::GraphicsPipeline::CreateInfo{
          .vertexShaderInput = rockInput,
          .rasterizationConfig = noCull,
          .blendingConfig = noBlend(GBuffer::COLOR_TARGETS),
          .depthConfig = depthCfg,
          .fragmentShaderOutput =
            {.colorAttachmentFormats = colorFormats,
             .depthAttachmentFormat = GBuffer::DEPTH_FORMAT},
        });
    }

    // Decals: classic alpha blending into the G-buffer. The write masks restrict which
    // channels a decal may change: albedo.rgb (not AO), the normal, roughness (not metallic).
    const auto blend = [](vk::ColorComponentFlags mask) {
      return vk::PipelineColorBlendAttachmentState{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eZero,
        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = mask,
      };
    };
    using C = vk::ColorComponentFlagBits;
    etna::GraphicsPipeline::CreateInfo::Blending decalBlend{};
    decalBlend.logicOp = vk::LogicOp::eCopy;
    decalBlend.attachments = {
      blend(C::eR | C::eG | C::eB),
      blend(layout == GBufferLayout::Compact ? (C::eR | C::eG) : (C::eR | C::eG | C::eB)),
      blend(C::eR),
    };

    decalPipelines[l] = {};
    decalPipelines[l] = pipelineManager.createGraphicsPipeline(
      "decal",
      etna::GraphicsPipeline::CreateInfo{
        // Back faces only: works the same when the camera is inside the box,
        // and every covered pixel is shaded exactly once
        .rasterizationConfig =
          vk::PipelineRasterizationStateCreateInfo{
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eFront,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .lineWidth = 1.f,
          },
        .blendingConfig = decalBlend,
        .depthConfig = noDepth,
        .fragmentShaderOutput = {.colorAttachmentFormats = colorFormats},
      });
  }

  // --- Lighting -------------------------------------------------------------------------------
  lightingPipeline = {};
  lightingPipeline = pipelineManager.createGraphicsPipeline(
    "deferred_lighting",
    etna::GraphicsPipeline::CreateInfo{
      .rasterizationConfig = noCull,
      .depthConfig = noDepth,
      .fragmentShaderOutput = {.colorAttachmentFormats = {swapchain_format}},
    });

  // --- Forward reference ----------------------------------------------------------------------
  terrainForwardPipeline = {};
  terrainForwardPipeline = pipelineManager.createGraphicsPipeline(
    "terrain_forward",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = gridInput,
      .rasterizationConfig = noCull,
      .depthConfig = depthCloserWrite,
      .fragmentShaderOutput =
        {.colorAttachmentFormats = {swapchain_format},
         .depthAttachmentFormat = GBuffer::DEPTH_FORMAT},
    });
  rockForwardPipeline = {};
  rockForwardPipeline = pipelineManager.createGraphicsPipeline(
    "rock_forward",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = rockInput,
      .rasterizationConfig = noCull,
      .depthConfig = depthCloserWrite,
      .fragmentShaderOutput =
        {.colorAttachmentFormats = {swapchain_format},
         .depthAttachmentFormat = GBuffer::DEPTH_FORMAT},
    });
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
  const std::filesystem::path root = DEFERRED_ASSETS_ROOT;

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

    albedoLayers.push_back(
      albedo ? std::move(*albedo) : tex::make_solid(SIZE, SIZE, desc.fallbackAlbedo));

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

  albedoArray =
    tex::create_texture(*oneShotCmdMgr, "terrain_albedo", vk::Format::eR8G8B8A8Srgb, albedoLayers);
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


void WorldRenderer::loadRocks()
{
  const std::filesystem::path root = DEFERRED_ASSETS_ROOT;
  rockModels[0].load(*oneShotCmdMgr, root / "boulder_01" / "boulder_01.gltf", 1);
  rockModels[1].load(*oneShotCmdMgr, root / "rock_09" / "rock_09.gltf", 2);

  auto& ctx = etna::get_context();
  rockInstanceBuffer.emplace(ctx.getMainWorkCount(), [&ctx](std::size_t i) {
    return ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = sizeof(RockInstance) * MAX_ROCKS_PER_MODEL * ROCK_MODELS,
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .allocationCreate =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .name = fmt::format("rock_instances{}", i),
    });
  });
}

void WorldRenderer::gatherRockInstances(const Camera& cam)
{
  ZoneScoped;
  const auto start = std::chrono::steady_clock::now();

  for (auto& v : rockScratch)
    v.clear();
  rocksConsidered = 0;

  // Frustum planes from the view-projection matrix (Gribb-Hartmann)
  const glm::mat4 m = glm::transpose(worldViewProj);
  std::array<glm::vec4, 6> planes{
    m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[2], m[3] - m[2]};
  for (auto& p : planes)
    p /= glm::length(glm::vec3(p));

  const float cell = std::max(rockCellSize, 0.5f);
  const glm::ivec2 c0 =
    glm::ivec2(glm::floor((glm::vec2(cam.position.x, cam.position.z) - rockDrawDistance) / cell));
  const glm::ivec2 c1 =
    glm::ivec2(glm::floor((glm::vec2(cam.position.x, cam.position.z) + rockDrawDistance) / cell));
  const auto seed = static_cast<std::uint32_t>(heightmapParams.seed);

  for (int cz = c0.y; cz <= c1.y; ++cz)
    for (int cx = c0.x; cx <= c1.x; ++cx)
    {
      // Everything about a rock depends only on its cell, so rocks never move with the camera
      const auto ux = static_cast<std::uint32_t>(cx);
      const auto uz = static_cast<std::uint32_t>(cz);
      auto rnd = [&](std::uint32_t k) { return prng::to01(prng::hash(ux, uz, seed * 31u + k)); };

      const glm::vec2 xz =
        (glm::vec2(static_cast<float>(cx), static_cast<float>(cz)) + glm::vec2(rnd(1), rnd(2))) *
        cell;
      const glm::vec2 uv = xz / worldSize + 0.5f;
      if (uv.x <= 0.0f || uv.y <= 0.0f || uv.x >= 1.0f || uv.y >= 1.0f)
        continue;

      const float h = HeightmapGenerator::sample(heights, uv);
      // Density depends on altitude: no rocks under water, few on beaches and snowy peaks
      float density = rockDensity;
      density *= glm::smoothstep(seaLevel, layerHeights.x + 0.03f, h);
      density *= 1.0f - 0.8f * glm::smoothstep(layerHeights.z - 0.03f, layerHeights.z + 0.05f, h);
      density *= 0.6f + 0.6f * glm::smoothstep(layerHeights.x, layerHeights.y, h);
      if (rnd(3) >= density)
        continue;
      ++rocksConsidered;

      const glm::vec3 pos{xz.x, h * heightScale, xz.y};
      const float dist = glm::length(pos - cam.position);
      if (dist > rockDrawDistance)
        continue;

      const std::uint32_t modelIdx = prng::hash(ux, uz, seed + 7u) % ROCK_MODELS;
      const float t = rnd(4);
      const float scale = glm::mix(rockScaleRange.x, rockScaleRange.y, t * t);
      const float radius = rockModels[modelIdx].boundingRadius * scale;

      if (frustumCulling)
      {
        bool visible = true;
        for (const auto& p : planes)
          if (glm::dot(glm::vec3(p), pos) + p.w < -radius)
          {
            visible = false;
            break;
          }
        if (!visible)
          continue;
      }

      auto& list = rockScratch[modelIdx];
      if (list.size() >= MAX_ROCKS_PER_MODEL)
        continue;

      glm::mat4 tm = glm::translate(glm::mat4(1.0f), pos);
      tm = glm::rotate(tm, rnd(5) * glm::two_pi<float>(), glm::vec3(0, 1, 0));
      tm = glm::rotate(tm, (rnd(6) - 0.5f) * 0.4f, glm::vec3(1, 0, 0));
      tm = glm::scale(
        tm,
        glm::vec3(scale) *
          glm::vec3(0.85f + 0.3f * rnd(7), 0.8f + 0.4f * rnd(8), 0.85f + 0.3f * rnd(9)));

      const glm::vec3 tint = rockModels[modelIdx].baseColor *
        (glm::vec3(0.85f) +
         0.3f * glm::vec3(rnd(10), rnd(11), rnd(12)) * glm::vec3(1.0f, 0.9f, 0.8f));
      list.push_back(RockInstance{tm, glm::vec4(tint, 1.0f)});
    }

  // Upload grouped instances: model i occupies [i * MAX_ROCKS_PER_MODEL, ...)
  auto* dst = reinterpret_cast<RockInstance*>(rockInstanceBuffer->get().data());
  for (std::uint32_t i = 0; i < ROCK_MODELS; ++i)
  {
    visibleRocks[i] = static_cast<std::uint32_t>(rockScratch[i].size());
    std::memcpy(
      dst + static_cast<std::size_t>(i) * MAX_ROCKS_PER_MODEL,
      rockScratch[i].data(),
      rockScratch[i].size() * sizeof(RockInstance));
  }

  rockCpuMs =
    std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
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


TerrainInfo WorldRenderer::terrainInfo() const
{
  return TerrainInfo{
    .heights = &heights,
    .worldSize = worldSize,
    .heightScale = heightScale,
    .seaLevel = seaLevel,
  };
}

void WorldRenderer::applyOverrides(const Overrides& o)
{
  if (o.deferredMode)
    deferredMode = *o.deferredMode;
  if (o.layout)
    gbufferLayout = *o.layout;
  if (o.decals)
    decals.enabled = *o.decals;
  if (o.debugView)
    debugView = *o.debugView;
}

void WorldRenderer::printTimings() const
{
  spdlog::info(
    "Mode: {}, G-buffer layout: {} ({} B/px), prepass: {}, decals: {}, resolution {}x{}",
    deferredMode ? "deferred" : "forward",
    gbufferLayout == GBufferLayout::Compact ? "compact" : "naive",
    GBuffer::bytesPerPixel(gbufferLayout),
    useDepthPrepass,
    decals.gpuCount(),
    resolution.x,
    resolution.y);
  for (const auto& s : gpuTimer.sections())
    spdlog::info("  {:<16} {:8.3f} ms", s.name, s.smoothedMs);
}

void WorldRenderer::update(const FramePacket& packet)
{
  ZoneScoped;

  if (heightmapDirty)
    regenerateHeightmap();

  const float aspect = float(resolution.x) / float(resolution.y);
  // Reverse-Z: near plane -> depth 1, far plane -> depth 0 (near and far are simply swapped).
  // Together with a float depth buffer this gives almost uniform relative precision over the
  // whole depth range, which matters a lot once positions (and normals for decals) are restored
  // from depth instead of being interpolated.
  glm::mat4 proj = glm::perspectiveRH_ZO(
    glm::radians(packet.mainCam.fov), aspect, packet.mainCam.zFar, packet.mainCam.zNear);
  proj[1][1] *= -1; // flip Y for Vulkan
  const glm::mat4 view = packet.mainCam.viewTm();
  worldViewProj = proj * view;
  cameraPos = packet.mainCam.position;
  cameraForward = packet.mainCam.forward();
  currentTime = packet.currentTime;

  // View matrix without the translation: maps camera-relative offsets, so the inverse of
  // proj * viewRot turns NDC into a small vector that is added to the camera position.
  const glm::mat4 viewRot = glm::mat4(glm::mat3(view));

  sceneParams.viewProj = worldViewProj;
  sceneParams.invViewProjRel = glm::inverse(proj * viewRot);
  sceneParams.cameraPos = glm::vec4(cameraPos, currentTime);
  sceneParams.screen = glm::vec4(
    static_cast<float>(resolution.x),
    static_cast<float>(resolution.y),
    1.0f / static_cast<float>(resolution.x),
    1.0f / static_cast<float>(resolution.y));
  sceneParams.sunDir = glm::vec4(sunDirection(), sunIntensity);
  sceneParams.sunColor = glm::vec4(sunColor, ambientIntensity);
  sceneParams.skyColor = glm::vec4(skyColor, exposure);
  sceneParams.groundColor = glm::vec4(groundAmbient, fogDensity);
  sceneParams.terrainSize =
    glm::vec4(worldSize, heightScale, seaLevel, static_cast<float>(HeightmapGenerator::RESOLUTION));
  sceneParams.layerHeights = layerHeights;
  sceneParams.detail = glm::vec4(textureTiling, detailStrength, slopeRock, heightBlendSharpness);
  sceneParams.debug =
    glm::vec4(static_cast<float>(debugView), detailNormals ? 1.0f : 0.0f, roughnessScale, cavityAo);
  sceneParams.gbuffer = glm::vec4(gbufferLayout == GBufferLayout::Compact ? 1.0f : 0.0f, 0, 0, 0);
  sceneParams.lighting = glm::vec4(ambientSpecular, 0, 0, 0);

  if (drawRocksEnabled)
    gatherRockInstances(packet.mainCam);
  else
    visibleRocks = {};

  decals.update(terrainInfo());

  // This frame's buffer is guaranteed to not be in use by the GPU anymore
  std::memcpy(sceneUbo->get().data(), &sceneParams, sizeof(sceneParams));
}

// ---------------------------------------------------------------------------------------------
// Scene drawing
// ---------------------------------------------------------------------------------------------

void WorldRenderer::drawTerrain(
  vk::CommandBuffer cmd_buf, const etna::GraphicsPipeline& pipeline, TerrainProgram program)
{
  const char* programName = program == TerrainProgram::DepthOnly ? "terrain_depth"
    : program == TerrainProgram::GBuffer                         ? "terrain_gbuffer"
                                                                 : "terrain_forward";
  auto programInfo = etna::get_shader_program(programName);

  std::vector<etna::Binding> bindings{
    etna::Binding{0, sceneUbo->get().genBinding()},
    etna::Binding{
      1, heightMap.genBinding(heightSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
  };
  if (program != TerrainProgram::DepthOnly)
  {
    bindings.push_back(etna::Binding{
      2,
      albedoArray.genBinding(
        materialSampler.get(),
        vk::ImageLayout::eShaderReadOnlyOptimal,
        {.type = vk::ImageViewType::e2DArray})});
    bindings.push_back(etna::Binding{
      3,
      roughHeightArray.genBinding(
        materialSampler.get(),
        vk::ImageLayout::eShaderReadOnlyOptimal,
        {.type = vk::ImageViewType::e2DArray})});
  }

  auto set = etna::create_descriptor_set(programInfo.getDescriptorLayoutId(0), cmd_buf, bindings);

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, pipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});

  cmd_buf.bindVertexBuffers(0, {gridVertices.get()}, {0});
  cmd_buf.bindIndexBuffer(gridIndices.get(), 0, vk::IndexType::eUint32);
  cmd_buf.drawIndexed(gridIndexCount, 1, 0, 0, 0);
}

void WorldRenderer::drawRocks(
  vk::CommandBuffer cmd_buf, const etna::GraphicsPipeline& pipeline, TerrainProgram program)
{
  if (visibleRocks[0] + visibleRocks[1] == 0)
    return;

  const char* programName = program == TerrainProgram::DepthOnly ? "rock_depth"
    : program == TerrainProgram::GBuffer                         ? "rock_gbuffer"
                                                                 : "rock_forward";
  auto programInfo = etna::get_shader_program(programName);
  auto set = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmd_buf,
    {
      etna::Binding{0, rockInstanceBuffer->get().genBinding()},
      etna::Binding{1, sceneUbo->get().genBinding()},
    });

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics, pipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});

  // The depth-only program has no fragment shader, so the push constants live in fewer stages
  const vk::ShaderStageFlags stages = program == TerrainProgram::DepthOnly
    ? vk::ShaderStageFlagBits::eVertex
    : vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;

  for (std::uint32_t i = 0; i < ROCK_MODELS; ++i)
  {
    if (visibleRocks[i] == 0)
      continue;
    const RockPushConstants pc{
      .rock =
        glm::vec4(static_cast<float>(i * MAX_ROCKS_PER_MODEL), rockRoughness, rockMetallic, 0.0f),
    };
    cmd_buf.pushConstants<RockPushConstants>(pipeline.getVkPipelineLayout(), stages, 0, {pc});
    cmd_buf.bindVertexBuffers(0, {rockModels[i].vertexBuffer.get()}, {0});
    cmd_buf.bindIndexBuffer(rockModels[i].indexBuffer.get(), 0, vk::IndexType::eUint32);
    // All instances of one model in a single draw call
    cmd_buf.drawIndexed(rockModels[i].indexCount, visibleRocks[i], 0, 0, 0);
  }
}

// ---------------------------------------------------------------------------------------------
// Deferred passes
// ---------------------------------------------------------------------------------------------

void WorldRenderer::depthPrepass(vk::CommandBuffer cmd_buf, GBuffer& gb)
{
  ETNA_PROFILE_GPU(cmd_buf, depthPrepass);
  gpuTimer.begin(cmd_buf, "Depth prepass");

  etna::RenderTargetState renderTargets(
    cmd_buf,
    {{0, 0}, {resolution.x, resolution.y}},
    {},
    {.image = gb.depth.get(), .view = gb.depth.getView({}), .clearDepthStencilValue = {0.0f, 0}});

  drawTerrain(cmd_buf, terrainDepthPipeline, TerrainProgram::DepthOnly);
  drawRocks(cmd_buf, rockDepthPipeline, TerrainProgram::DepthOnly);

  gpuTimer.end(cmd_buf);
}

void WorldRenderer::geometryPass(vk::CommandBuffer cmd_buf, GBuffer& gb)
{
  ETNA_PROFILE_GPU(cmd_buf, gbufferPass);
  gpuTimer.begin(cmd_buf, "G-buffer pass");

  if (useDepthPrepass)
  {
    // The depth attachment keeps the same state between the two passes, so etna would not emit
    // a barrier by itself. The EQUAL test reads what the prepass wrote: force a dependency.
    etna::set_state(
      cmd_buf,
      gb.depth.get(),
      vk::PipelineStageFlagBits2::eEarlyFragmentTests |
        vk::PipelineStageFlagBits2::eLateFragmentTests,
      vk::AccessFlagBits2::eDepthStencilAttachmentRead |
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
      vk::ImageLayout::eDepthStencilAttachmentOptimal,
      vk::ImageAspectFlagBits::eDepth,
      ForceSetState::eTrue);
  }

  std::vector<etna::RenderTargetState::AttachmentParams> colors;
  for (const etna::Image* img : gb.colorTargets())
    colors.push_back({
      .image = img->get(),
      .view = img->getView({}),
      // Every pixel is either covered by geometry or is sky (depth = 1, never read)
      .loadOp = vk::AttachmentLoadOp::eDontCare,
    });

  const std::size_t l = static_cast<std::size_t>(gb.layout());
  const std::size_t p = useDepthPrepass ? 1 : 0;
  {
    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      colors,
      {
        .image = gb.depth.get(),
        .view = gb.depth.getView({}),
        .loadOp = useDepthPrepass ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear,
        .clearDepthStencilValue = {0.0f, 0}, // reverse-Z: 0 = infinitely far
      });

    drawTerrain(cmd_buf, terrainGBufferPipelines[l][p], TerrainProgram::GBuffer);
    drawRocks(cmd_buf, rockGBufferPipelines[l][p], TerrainProgram::GBuffer);
  }

  gpuTimer.end(cmd_buf);
}

void WorldRenderer::decalPass(vk::CommandBuffer cmd_buf, GBuffer& gb)
{
  if (decals.gpuCount() == 0)
    return;

  ETNA_PROFILE_GPU(cmd_buf, decalPass);
  gpuTimer.begin(cmd_buf, "Decals");

  auto programInfo = etna::get_shader_program("decal");
  // Created before the render pass begins, so the depth -> sampled transition is recorded
  // in the barrier batch flushed by RenderTargetState
  auto set = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmd_buf,
    {
      etna::Binding{0, sceneUbo->get().genBinding()},
      etna::Binding{1, decals.currentBuffer().genBinding()},
      etna::Binding{
        2, gb.depth.genBinding(pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{
        3,
        decals.albedoArray.genBinding(
          decalSampler.get(),
          vk::ImageLayout::eShaderReadOnlyOptimal,
          {.type = vk::ImageViewType::e2DArray})},
      etna::Binding{
        4,
        decals.normalArray.genBinding(
          decalSampler.get(),
          vk::ImageLayout::eShaderReadOnlyOptimal,
          {.type = vk::ImageViewType::e2DArray})},
    });

  // Blending reads the G-buffer written by the previous pass: same state, so force a barrier
  std::vector<etna::RenderTargetState::AttachmentParams> colors;
  for (const etna::Image* img : gb.colorTargets())
  {
    etna::set_state(
      cmd_buf,
      img->get(),
      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
      vk::ImageLayout::eColorAttachmentOptimal,
      vk::ImageAspectFlagBits::eColor,
      ForceSetState::eTrue);
    colors.push_back({
      .image = img->get(),
      .view = img->getView({}),
      .loadOp = vk::AttachmentLoadOp::eLoad,
    });
  }

  {
    etna::RenderTargetState renderTargets(
      cmd_buf, {{0, 0}, {resolution.x, resolution.y}}, colors, {});

    const auto& pipeline = decalPipelines[static_cast<std::size_t>(gb.layout())];
    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics, pipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});

    const DecalPushConstants pc{
      .global = glm::vec4(
        decals.globalOpacity,
        decals.showBoxes ? 1.0f : 0.0f,
        gb.layout() == GBufferLayout::Compact ? 1.0f : 0.0f,
        0.0f),
    };
    cmd_buf.pushConstants<DecalPushConstants>(
      pipeline.getVkPipelineLayout(),
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
      0,
      {pc});
    // 36 vertices of a box per decal, one instance per decal
    cmd_buf.draw(36, decals.gpuCount(), 0, 0);
  }

  gpuTimer.end(cmd_buf);
}

void WorldRenderer::lightingPass(
  vk::CommandBuffer cmd_buf, GBuffer& gb, vk::Image target_image, vk::ImageView target_view)
{
  ETNA_PROFILE_GPU(cmd_buf, lightingPass);
  gpuTimer.begin(cmd_buf, "Lighting");

  auto programInfo = etna::get_shader_program("deferred_lighting");
  // G-buffer -> sampled transitions are recorded here and flushed by RenderTargetState below
  auto set = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmd_buf,
    {
      etna::Binding{0, sceneUbo->get().genBinding()},
      etna::Binding{
        1, gb.albedo.genBinding(pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{
        2, gb.normal.genBinding(pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{
        3, gb.material.genBinding(pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{
        4, gb.depth.genBinding(pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    });

  {
    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {{
        .image = target_image,
        .view = target_view,
        // The fullscreen triangle overwrites every pixel
        .loadOp = vk::AttachmentLoadOp::eDontCare,
      }},
      {});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, lightingPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      lightingPipeline.getVkPipelineLayout(),
      0,
      {set.getVkSet()},
      {});
    cmd_buf.draw(3, 1, 0, 0);
  }

  gpuTimer.end(cmd_buf);
}

void WorldRenderer::renderDeferred(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_view)
{
  GBuffer& gb = activeGBuffer();
  if (useDepthPrepass)
    depthPrepass(cmd_buf, gb);
  geometryPass(cmd_buf, gb);
  decalPass(cmd_buf, gb);
  lightingPass(cmd_buf, gb, target_image, target_view);
}

void WorldRenderer::renderForward(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_view)
{
  ETNA_PROFILE_GPU(cmd_buf, forwardPass);
  gpuTimer.begin(cmd_buf, "Forward pass");

  // Sky color for the clear, run through the same exposure + tonemapping as the surfaces
  const glm::vec3 x = skyColor * exposure;
  glm::vec3 sky = glm::clamp(
    (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f),
    glm::vec3(0.0f),
    glm::vec3(1.0f));
  if (debugView != 0)
    sky = glm::vec3(0.0f);

  GBuffer& gb = activeGBuffer();
  {
    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {{
        .image = target_image,
        .view = target_view,
        .clearColorValue = std::array<float, 4>{sky.r, sky.g, sky.b, 1.0f},
      }},
      {.image = gb.depth.get(), .view = gb.depth.getView({}), .clearDepthStencilValue = {0.0f, 0}});

    drawTerrain(cmd_buf, terrainForwardPipeline, TerrainProgram::Forward);
    drawRocks(cmd_buf, rockForwardPipeline, TerrainProgram::Forward);
  }

  gpuTimer.end(cmd_buf);
}

void WorldRenderer::renderWorld(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd_buf, renderWorld);

  gpuTimer.beginFrame(cmd_buf);
  gpuTimer.begin(cmd_buf, "Total (world)");

  if (deferredMode)
    renderDeferred(cmd_buf, target_image, target_image_view);
  else
    renderForward(cmd_buf, target_image, target_image_view);

  gpuTimer.end(cmd_buf);
}

void WorldRenderer::drawGui()
{
  ImGui::Begin("Deferred renderer");

  ImGui::Text(
    "%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

  if (ImGui::CollapsingHeader("Deferred shading", ImGuiTreeNodeFlags_DefaultOpen))
  {
    int mode = deferredMode ? 0 : 1;
    ImGui::RadioButton("Deferred", &mode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Forward (reference)", &mode, 1);
    if ((mode == 0) != deferredMode)
    {
      deferredMode = mode == 0;
      gpuTimer.resetAverages();
    }

    int layout = static_cast<int>(gbufferLayout);
    const char* layouts[] = {"Compact (14 B/px)", "Naive RGBA16F (28 B/px)"};
    if (ImGui::Combo("G-buffer layout", &layout, layouts, IM_ARRAYSIZE(layouts)))
    {
      gbufferLayout = static_cast<GBufferLayout>(layout);
      gpuTimer.resetAverages();
    }
    if (ImGui::Checkbox("Depth prepass (G-buffer with EQUAL test)", &useDepthPrepass))
      gpuTimer.resetAverages();

    const std::uint32_t bpp = GBuffer::bytesPerPixel(gbufferLayout);
    ImGui::Text(
      "G-buffer: %u B/px x %ux%u = %.1f MiB",
      bpp,
      resolution.x,
      resolution.y,
      static_cast<double>(bpp) * resolution.x * resolution.y / (1024.0 * 1024.0));

    if (gpuTimer.supported())
    {
      if (ImGui::BeginTable("gpu_timings", 2, ImGuiTableFlags_BordersInnerH))
      {
        ImGui::TableSetupColumn("GPU pass");
        ImGui::TableSetupColumn("ms (avg)");
        ImGui::TableHeadersRow();
        for (const auto& s : gpuTimer.sections())
        {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted(s.name.data(), s.name.data() + s.name.size());
          ImGui::TableNextColumn();
          ImGui::Text("%.3f", s.smoothedMs);
        }
        ImGui::EndTable();
      }
    }
    else
      ImGui::TextDisabled("Timestamp queries are not supported, use Tracy");

    const char* views[] = {
      "Shaded",
      "Albedo",
      "Normals",
      "Roughness",
      "Metallic",
      "AO",
      "Linear depth",
      "Reconstructed position (10 m grid)",
      "Terrain layer weights",
      "Terrain height"};
    ImGui::Combo("Debug view", &debugView, views, IM_ARRAYSIZE(views));
  }

  if (ImGui::CollapsingHeader("Decals", ImGuiTreeNodeFlags_DefaultOpen))
  {
    if (!deferredMode)
      ImGui::TextDisabled("Decals are applied to the G-buffer: deferred mode only");
    decals.drawGui(terrainInfo(), cameraPos, cameraForward);
  }

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
      heightmapParams.seed =
        static_cast<int>(prng::pcg(static_cast<std::uint32_t>(heightmapParams.seed)) & 0xFFFFF);
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
    ImGui::SliderFloat("Cavity AO from height maps", &cavityAo, 0.0f, 1.0f);
  }

  if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::SliderFloat("Sun azimuth", &sunAzimuth, 0.0f, 360.0f);
    ImGui::SliderFloat("Sun elevation", &sunElevation, 1.0f, 90.0f);
    ImGui::SliderFloat("Sun intensity", &sunIntensity, 0.0f, 10.0f);
    ImGui::ColorEdit3("Sun color", &sunColor.x);
    ImGui::SliderFloat("Ambient", &ambientIntensity, 0.0f, 2.0f);
    ImGui::SliderFloat("Ambient specular (sky reflections)", &ambientSpecular, 0.0f, 2.0f);
    ImGui::ColorEdit3("Sky color", &skyColor.x);
    ImGui::ColorEdit3("Ground bounce", &groundAmbient.x);
    ImGui::SliderFloat("Exposure", &exposure, 0.1f, 4.0f);
    ImGui::SliderFloat("Fog density", &fogDensity, 0.0f, 0.005f, "%.5f");
  }

  if (ImGui::CollapsingHeader("Rocks (instancing)", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::Checkbox("Draw rocks", &drawRocksEnabled);
    ImGui::Checkbox("Frustum culling", &frustumCulling);
    ImGui::SliderFloat("Density", &rockDensity, 0.0f, 1.0f);
    ImGui::SliderFloat("Cell size (m)", &rockCellSize, 1.0f, 30.0f);
    ImGui::SliderFloat("Draw distance (m)", &rockDrawDistance, 20.0f, 1000.0f);
    ImGui::SliderFloat("Rock roughness", &rockRoughness, 0.04f, 1.0f);
    ImGui::SliderFloat("Rock metallic", &rockMetallic, 0.0f, 1.0f);
    ImGui::DragFloatRange2("Size range", &rockScaleRange.x, &rockScaleRange.y, 0.01f, 0.05f, 10.0f);
    ImGui::Text(
      "In range: %u, drawn: boulder %u + rock %u",
      rocksConsidered,
      visibleRocks[0],
      visibleRocks[1]);
    ImGui::Text("CPU placement + culling: %.2f ms, draw calls: 2", rockCpuMs);
  }

  ImGui::Separator();
  ImGui::TextWrapped(
    "WASD + R/F (E/Q): move, Shift: faster, RMB: toggle mouse look, G: regenerate, "
    "B: reload shaders");

  ImGui::End();
}
