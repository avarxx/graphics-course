#pragma once

#include <optional>
#include <string>

#include <glm/glm.hpp>

#include "WorldRenderer.hpp"


// Command line options. Handy for automated comparison screenshots and benchmarks:
//   deferred --forward | --naive | --no-decals | --debug-view N | --no-gui
//            --camera fx fy fz tx ty tz | --frames N (quit after N frames) | --print-timings
struct AppOptions
{
  WorldRenderer::Overrides overrides;
  bool hideGui = false;
  bool printTimings = false;
  int exitAfterFrames = -1;
  std::optional<std::pair<glm::vec3, glm::vec3>> camera;

  static AppOptions parse(int argc, char** argv);
};
