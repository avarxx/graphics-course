#include "App.hpp"

#include <tracy/Tracy.hpp>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include "gui/ImGuiRenderer.hpp"


App::App(AppOptions opts)
  : options{std::move(opts)}
{
  glm::uvec2 initialRes = {1280, 720};
  mainWindow = windowing.createWindow(OsWindow::CreateInfo{
    .resolution = initialRes,
    .resizeable = true,
    .refreshCb =
      [this]() {
        // NOTE: this is only called when the window is being resized.
        drawFrame(0.0f);
        FrameMark;
      },
    .resizeCb =
      [this](glm::uvec2 res) {
        if (res.x == 0 || res.y == 0)
          return;

        renderer->recreateSwapchain(res);
      },
  });

  renderer.reset(new Renderer(initialRes));

  auto instExts = windowing.getRequiredVulkanInstanceExtensions();
  renderer->initVulkan(instExts);

  auto surface = mainWindow->createVkSurface(etna::get_context().getInstance());

  renderer->initFrameDelivery(
    std::move(surface), [window = mainWindow.get()]() { return window->getResolution(); });

  ImGuiRenderer::enableImGuiForWindow(mainWindow->native());

  mainCam.zNear = 0.1f;
  mainCam.zFar = 3000.0f;
  mainCam.lookAt({0, 160, 420}, {0, 20, 0}, {0, 1, 0});
  if (options.camera)
    mainCam.lookAt(options.camera->first, options.camera->second, {0, 1, 0});

  renderer->applyOptions(options.overrides, options.hideGui);
}

void App::run()
{
  double lastTime = windowing.getTime();
  while (!mainWindow->isBeingClosed())
  {
    const double currTime = windowing.getTime();
    const float diffTime = static_cast<float>(currTime - lastTime);
    lastTime = currTime;

    windowing.poll();

    processInput(diffTime);

    drawFrame(diffTime);

    FrameMark;

    if (options.exitAfterFrames > 0)
    {
      if (++frameCount % 10 == 0)
        spdlog::info("frame {}", frameCount);
      if (frameCount >= options.exitAfterFrames)
        break;
    }
  }

  if (options.printTimings)
    renderer->printTimings();
}

void App::processInput(float dt)
{
  ZoneScoped;

  if (mainWindow->keyboard[KeyboardKey::kEscape] == ButtonState::Falling)
    mainWindow->askToClose();

  if (is_held_down(mainWindow->keyboard[KeyboardKey::kLeftShift]))
    camMoveSpeed = 150;
  else
    camMoveSpeed = 30;

  if (mainWindow->mouse[MouseButton::mbRight] == ButtonState::Rising)
    mainWindow->captureMouse = !mainWindow->captureMouse;

  // Do not fly around while the user is typing into ImGui fields
  if (!ImGui::GetIO().WantCaptureKeyboard)
    moveCam(mainCam, mainWindow->keyboard, dt);
  if (mainWindow->captureMouse)
    rotateCam(mainCam, mainWindow->mouse, dt);

  renderer->debugInput(mainWindow->keyboard);
}

void App::drawFrame(float dt)
{
  ZoneScoped;

  renderer->update(FramePacket{
    .mainCam = mainCam,
    .currentTime = static_cast<float>(windowing.getTime()),
    .deltaTime = dt,
  });
  renderer->drawFrame();
}

void App::moveCam(Camera& cam, const Keyboard& kb, float dt)
{
  // Free camera: WASD to move, R/F (or E/Q) for up and down

  glm::vec3 dir = {0, 0, 0};

  if (is_held_down(kb[KeyboardKey::kS]))
    dir -= cam.forward();

  if (is_held_down(kb[KeyboardKey::kW]))
    dir += cam.forward();

  if (is_held_down(kb[KeyboardKey::kA]))
    dir -= cam.right();

  if (is_held_down(kb[KeyboardKey::kD]))
    dir += cam.right();

  if (is_held_down(kb[KeyboardKey::kF]) || is_held_down(kb[KeyboardKey::kQ]))
    dir -= cam.up();

  if (is_held_down(kb[KeyboardKey::kR]) || is_held_down(kb[KeyboardKey::kE]))
    dir += cam.up();

  cam.move(dt * camMoveSpeed * (length(dir) > 1e-9 ? normalize(dir) : dir));
}

void App::rotateCam(Camera& cam, const Mouse& ms, float /*dt*/)
{
  cam.rotate(-camRotateSpeed * ms.capturedPosDelta.y, -camRotateSpeed * ms.capturedPosDelta.x);

  cam.fov -= zoomSensitivity * ms.scrollDelta.y;
  if (cam.fov < 1.0f)
    cam.fov = 1.0f;
  if (cam.fov > 120.0f)
    cam.fov = 120.0f;
}
