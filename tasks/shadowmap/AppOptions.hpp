#pragma once

#include <optional>
#include <string>

#include <glm/glm.hpp>

#include "WorldRenderer.hpp"


// Command line options. Handy for automated comparison screenshots and benchmarks:
//   tonemapping --forward | --naive | --no-decals | --debug-view N | --no-gui
//            --camera fx fy fz tx ty tz | --frames N (quit after N frames) | --print-timings
//            --sun azimuth elevation | --coverage C | --no-clouds | --sky-size N (default 1024)
//            --no-tonemap | --no-grading | --lut NAME | --tonemap-op N | --no-dither | --no-srgb
//            --global-atomics (min/max + histogram without shared memory)
//            --no-shadows | --pcf N (PCF kernel radius in texels, 2 = 5x5)
//            --screenshot frame.png (write the first rendered frame, use with --no-gui)
struct AppOptions
{
  WorldRenderer::Overrides overrides;
  bool hideGui = false;
  bool printTimings = false;
  int exitAfterFrames = -1;
  std::string screenshotPath;
  std::uint32_t skyFaceSize = 1024;
  std::optional<std::pair<glm::vec3, glm::vec3>> camera;

  static AppOptions parse(int argc, char** argv);
};
