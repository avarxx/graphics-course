#include "Renderer.hpp"

#include <cstdint>
#include <vector>

#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>
#include <imgui.h>

#include <gui/ImGuiRenderer.hpp>


// stb_image_write is compiled into tinygltf, which this target already links
extern "C" int stbi_write_png(
  char const* filename, int w, int h, int comp, const void* data, int stride_in_bytes);


Renderer::Renderer(glm::uvec2 res)
  : resolution{res}
{
}

void Renderer::initVulkan(std::span<const char*> instance_extensions)
{
  std::vector<const char*> instanceExtensions;

  for (auto ext : instance_extensions)
    instanceExtensions.push_back(ext);

  std::vector<const char*> deviceExtensions;

  deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  etna::initialize(etna::InitParams{
    .applicationName = "Grass",
    .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
    .instanceExtensions = instanceExtensions,
    .deviceExtensions = deviceExtensions,
    // independentBlend: decals use different color write masks per G-buffer attachment
    .features = vk::PhysicalDeviceFeatures2{.features = {.independentBlend = vk::True}},
    // Replace with an index if etna detects your preferred GPU incorrectly
    .physicalDeviceIndexOverride = {},
    // How much frames we buffer on the GPU without waiting for their completion on the CPU
    .numFramesInFlight = 2,
  });
}

void Renderer::initFrameDelivery(vk::UniqueSurfaceKHR a_surface, ResolutionProvider res_provider)
{
  auto& ctx = etna::get_context();

  resolutionProvider = std::move(res_provider);
  commandManager = ctx.createPerFrameCmdMgr();

  window = ctx.createWindow(etna::Window::CreateInfo{
    .surface = std::move(a_surface),
  });

  auto [w, h] = window->recreateSwapchain(etna::Window::DesiredProperties{
    .resolution = {resolution.x, resolution.y},
    .vsync = true,
    .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
  });
  resolution = {w, h};

  worldRenderer = std::make_unique<WorldRenderer>(skyFaceSize);

  worldRenderer->allocateResources(resolution);
  worldRenderer->loadResources();
  worldRenderer->loadShaders();
  worldRenderer->setupPipelines(window->getCurrentFormat());

  guiRenderer = std::make_unique<ImGuiRenderer>(window->getCurrentFormat());
}

void Renderer::recreateSwapchain(glm::uvec2 res)
{
  auto& ctx = etna::get_context();

  ETNA_CHECK_VK_RESULT(ctx.getDevice().waitIdle());

  auto [w, h] = window->recreateSwapchain(etna::Window::DesiredProperties{
    .resolution = {res.x, res.y},
    .vsync = true,
    .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
  });
  resolution = {w, h};

  // Most resources depend on the current resolution, so we recreate them.
  worldRenderer->allocateResources(resolution);

  // Format of the swapchain CAN change on android
  worldRenderer->setupPipelines(window->getCurrentFormat());
}

void Renderer::debugInput(const Keyboard& kb)
{
  worldRenderer->debugInput(kb);

  if (kb[KeyboardKey::kB] == ButtonState::Falling)
  {
    const int retval = std::system("cd " GRAPHICS_COURSE_ROOT "/build"
                                   " && cmake --build . --target grass_task_shaders");
    if (retval != 0)
      spdlog::warn("Shader recompilation returned a non-zero return code!");
    else
    {
      ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
      etna::reload_shaders();
      spdlog::info("Successfully reloaded shaders!");
    }
  }
}

void Renderer::applyOptions(const WorldRenderer::Overrides& overrides, bool hide_gui)
{
  worldRenderer->applyOverrides(overrides);
  hideGui = hide_gui;
}

void Renderer::update(const FramePacket& packet)
{
  worldRenderer->update(packet);
}

void Renderer::drawFrame()
{
  ZoneScoped;

  {
    ZoneScopedN("drawGui");
    guiRenderer->nextFrame();
    ImGui::NewFrame();
    worldRenderer->drawGui();
    ImGui::Render();
  }

  auto currentCmdBuf = commandManager->acquireNext();

  // TODO: this makes literally 0 sense here, rename/refactor,
  // it doesn't actually begin anything, just resets descriptor pools
  etna::begin_frame();

  auto nextSwapchainImage = window->acquireNext();

  std::optional<etna::Buffer> screenshotBuffer;
  if (nextSwapchainImage && screenshotPath)
    screenshotBuffer.emplace(etna::get_context().createBuffer(etna::Buffer::CreateInfo{
      .size = static_cast<vk::DeviceSize>(resolution.x) * resolution.y * 4,
      .bufferUsage = vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .allocationCreate =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .name = "screenshot_buffer",
    }));

  // NOTE: here, we skip frames when the window is in the process of being
  // re-sized. This is not mandatory, it is possible to submit frames to a
  // "sub-optimal" swap chain and still get something drawn while resizing,
  // but only on some platforms (not windows+nvidia, sadly).
  if (nextSwapchainImage)
  {
    auto [image, view, availableSem, readyForPresentSem] = *nextSwapchainImage;

    ETNA_CHECK_VK_RESULT(currentCmdBuf.begin(vk::CommandBufferBeginInfo{}));
    {
      ETNA_PROFILE_GPU(currentCmdBuf, renderFrame);

      worldRenderer->renderWorld(currentCmdBuf, image, view);

      if (!hideGui)
      {
        ImDrawData* pDrawData = ImGui::GetDrawData();
        guiRenderer->render(
          currentCmdBuf, {{0, 0}, {resolution.x, resolution.y}}, image, view, pDrawData);
      }

      if (screenshotBuffer)
      {
        etna::set_state(
          currentCmdBuf,
          image,
          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
          vk::AccessFlagBits2::eColorAttachmentWrite,
          vk::ImageLayout::eTransferSrcOptimal,
          vk::ImageAspectFlagBits::eColor,
          ForceSetState::eTrue);
        etna::flush_barriers(currentCmdBuf);

        const vk::BufferImageCopy region{
          .bufferOffset = 0,
          .bufferRowLength = 0,
          .bufferImageHeight = 0,
          .imageSubresource =
            {.aspectMask = vk::ImageAspectFlagBits::eColor,
             .mipLevel = 0,
             .baseArrayLayer = 0,
             .layerCount = 1},
          .imageOffset = {0, 0, 0},
          .imageExtent = {resolution.x, resolution.y, 1},
        };
        currentCmdBuf.copyImageToBuffer(
          image, vk::ImageLayout::eTransferSrcOptimal, screenshotBuffer->get(), 1, &region);
      }

      etna::set_state(
        currentCmdBuf,
        image,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        {},
        vk::ImageLayout::ePresentSrcKHR,
        vk::ImageAspectFlagBits::eColor);

      etna::flush_barriers(currentCmdBuf);

      ETNA_READ_BACK_GPU_PROFILING(currentCmdBuf);
    }
    ETNA_CHECK_VK_RESULT(currentCmdBuf.end());

    auto renderingDone = commandManager->submit(
      std::move(currentCmdBuf), std::move(availableSem), std::move(readyForPresentSem));

    const bool presented = window->present(std::move(renderingDone), view);

    if (!presented)
      nextSwapchainImage = std::nullopt;
  }

  etna::end_frame();

  if (screenshotBuffer)
  {
    // Only used by the offscreen capture mode, where a short CPU stall does not matter
    ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());

    const vk::Format format = window->getCurrentFormat();
    const bool bgra = format == vk::Format::eB8G8R8A8Unorm || format == vk::Format::eB8G8R8A8Srgb;
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(resolution.x) * resolution.y * 4, 255);
    const auto* src = reinterpret_cast<const std::uint8_t*>(screenshotBuffer->data());
    for (std::size_t i = 0; i < static_cast<std::size_t>(resolution.x) * resolution.y; ++i)
    {
      rgba[i * 4 + 0] = bgra ? src[i * 4 + 2] : src[i * 4 + 0];
      rgba[i * 4 + 1] = src[i * 4 + 1];
      rgba[i * 4 + 2] = bgra ? src[i * 4 + 0] : src[i * 4 + 2];
      rgba[i * 4 + 3] = 255;
    }
    const int written = stbi_write_png(
      screenshotPath->c_str(),
      static_cast<int>(resolution.x),
      static_cast<int>(resolution.y),
      4,
      rgba.data(),
      static_cast<int>(resolution.x) * 4);
    if (written == 0)
      spdlog::error("Failed to write screenshot to {}", *screenshotPath);
    else
      spdlog::info("Screenshot written to {}", *screenshotPath);
    screenshotPath.reset();
  }

  if (!nextSwapchainImage)
  {
    auto res = resolutionProvider();
    // On windows, we get 0,0 while the window is minimized and
    // must skip frames until the window is un-minimized again
    if (res.x != 0 && res.y != 0)
      recreateSwapchain(res);
  }
}

Renderer::~Renderer()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}
