#include "App.hpp"

#include <cstdlib>
#include <array>

#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/DescriptorSet.hpp>


struct ToyParams
{
  glm::vec4 mouse;
  glm::vec2 resolution;
  float time;
};

App::App()
  : resolution{1280, 720}
  , useVsync{true}
{
  {
    auto glfwInstExts = windowing.getRequiredVulkanInstanceExtensions();

    std::vector<const char*> instanceExtensions{glfwInstExts.begin(), glfwInstExts.end()};

    std::vector<const char*> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    etna::initialize(etna::InitParams{
      .applicationName = "Local Shadertoy",
      .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
      .instanceExtensions = instanceExtensions,
      .deviceExtensions = deviceExtensions,
      .physicalDeviceIndexOverride = {},
      .numFramesInFlight = 1,
    });
  }

  commandManager = etna::get_context().createPerFrameCmdMgr();

  osWindow = windowing.createWindow(OsWindow::CreateInfo{
    .resolution = resolution,
  });

  {
    auto surface = osWindow->createVkSurface(etna::get_context().getInstance());

    vkWindow = etna::get_context().createWindow(etna::Window::CreateInfo{
      .surface = std::move(surface),
    });

    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
      .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
    });

    resolution = {w, h};
  }

  etna::create_program("local_shadertoy", {LOCAL_SHADERTOY1_SHADERS_ROOT "toy.comp.spv"});
  toyPipeline =
    etna::get_context().getPipelineManager().createComputePipeline("local_shadertoy", {});

  resultImage = etna::get_context().createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "toy_result",
    .format = vk::Format::eR16G16B16A16Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
  });

  startTime = std::chrono::steady_clock::now();
}

App::~App()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::run()
{
  while (!osWindow->isBeingClosed())
  {
    windowing.poll();

    processInput();

    drawFrame();
  }

  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::processInput()
{
  if (osWindow->keyboard[KeyboardKey::kR] == ButtonState::Falling)
  {
    const int retval = std::system(
      "cd " LOCAL_SHADERTOY_BUILD_DIR " && cmake --build . --target local_shadertoy1_shaders");
    if (retval == 0)
    {
      ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
      etna::reload_shaders();
    }
  }

  const glm::vec2 pos = osWindow->mouse.freePos;
  const glm::vec2 flipped{pos.x, static_cast<float>(resolution.y) - pos.y};
  const ButtonState lmb = osWindow->mouse[MouseButton::mbLeft];

  if (lmb == ButtonState::Rising || lmb == ButtonState::High)
  {
    if (mouseState.z <= 0.0f)
    {
      mouseState.z = flipped.x;
      mouseState.w = flipped.y;
    }
    mouseState.x = flipped.x;
    mouseState.y = flipped.y;
  }
  else
  {
    mouseState.z = -glm::abs(mouseState.z);
    mouseState.w = -glm::abs(mouseState.w);
  }
}

void App::drawFrame()
{
  auto currentCmdBuf = commandManager->acquireNext();

  etna::begin_frame();

  auto nextSwapchainImage = vkWindow->acquireNext();

  if (nextSwapchainImage)
  {
    auto [backbuffer, backbufferView, backbufferAvailableSem, backbufferReadyForPresentSem] =
      *nextSwapchainImage;

    ETNA_CHECK_VK_RESULT(currentCmdBuf.begin(vk::CommandBufferBeginInfo{}));
    {
      etna::set_state(
        currentCmdBuf,
        resultImage.get(),
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlags2(vk::AccessFlagBits2::eShaderStorageWrite),
        vk::ImageLayout::eGeneral,
        vk::ImageAspectFlagBits::eColor);

      auto programInfo = etna::get_shader_program("local_shadertoy");
      auto set = etna::create_descriptor_set(
        programInfo.getDescriptorLayoutId(0),
        currentCmdBuf,
        {etna::Binding{0, resultImage.genBinding({}, vk::ImageLayout::eGeneral)}});

      etna::flush_barriers(currentCmdBuf);

      currentCmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, toyPipeline.getVkPipeline());
      currentCmdBuf.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        toyPipeline.getVkPipelineLayout(),
        0,
        {set.getVkSet()},
        {});

      const float time =
        std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();

      ToyParams params{
        .mouse = mouseState,
        .resolution = glm::vec2(resolution),
        .time = time,
      };

      currentCmdBuf.pushConstants(
        toyPipeline.getVkPipelineLayout(),
        vk::ShaderStageFlagBits::eCompute,
        0,
        sizeof(params),
        &params);

      currentCmdBuf.dispatch((resolution.x + 15) / 16, (resolution.y + 15) / 16, 1);

      etna::set_state(
        currentCmdBuf,
        resultImage.get(),
        vk::PipelineStageFlagBits2::eBlit,
        vk::AccessFlags2(vk::AccessFlagBits2::eTransferRead),
        vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageAspectFlagBits::eColor);

      etna::set_state(
        currentCmdBuf,
        backbuffer,
        vk::PipelineStageFlagBits2::eBlit,
        vk::AccessFlags2(vk::AccessFlagBits2::eTransferWrite),
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageAspectFlagBits::eColor);

      etna::flush_barriers(currentCmdBuf);

      vk::ImageBlit region{};
      region.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
      region.srcSubresource.mipLevel = 0;
      region.srcSubresource.baseArrayLayer = 0;
      region.srcSubresource.layerCount = 1;
      region.srcOffsets[0] = vk::Offset3D{0, 0, 0};
      region.srcOffsets[1] =
        vk::Offset3D{static_cast<int32_t>(resolution.x), static_cast<int32_t>(resolution.y), 1};
      region.dstSubresource = region.srcSubresource;
      region.dstOffsets[0] = vk::Offset3D{0, 0, 0};
      region.dstOffsets[1] = region.srcOffsets[1];

      currentCmdBuf.blitImage(
        resultImage.get(),
        vk::ImageLayout::eTransferSrcOptimal,
        backbuffer,
        vk::ImageLayout::eTransferDstOptimal,
        region,
        vk::Filter::eLinear);

      etna::set_state(
        currentCmdBuf,
        backbuffer,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        {},
        vk::ImageLayout::ePresentSrcKHR,
        vk::ImageAspectFlagBits::eColor);

      etna::flush_barriers(currentCmdBuf);
    }
    ETNA_CHECK_VK_RESULT(currentCmdBuf.end());

    auto renderingDone = commandManager->submit(
      std::move(currentCmdBuf),
      std::move(backbufferAvailableSem),
      std::move(backbufferReadyForPresentSem));

    const bool presented = vkWindow->present(std::move(renderingDone), backbufferView);

    if (!presented)
      nextSwapchainImage = std::nullopt;
  }

  etna::end_frame();

  if (!nextSwapchainImage && osWindow->getResolution() != glm::uvec2{0, 0})
  {
    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
      .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
    });
    ETNA_VERIFY((resolution == glm::uvec2{w, h}));
  }
}
