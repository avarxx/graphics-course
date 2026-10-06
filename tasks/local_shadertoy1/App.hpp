#pragma once

#include <chrono>

#include <etna/Window.hpp>
#include <etna/PerFrameCmdMgr.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/Image.hpp>

#include "wsi/OsWindowingManager.hpp"


class App
{
public:
  App();
  ~App();

  void run();

private:
  void drawFrame();
  void processInput();

private:
  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> osWindow;

  glm::uvec2 resolution;
  bool useVsync;

  std::unique_ptr<etna::Window> vkWindow;
  std::unique_ptr<etna::PerFrameCmdMgr> commandManager;

  etna::ComputePipeline toyPipeline;
  etna::Image resultImage;

  std::chrono::steady_clock::time_point startTime;
  glm::vec4 mouseState{0.0f, 0.0f, -1.0f, -1.0f};
};
