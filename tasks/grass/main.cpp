#include <cstdlib>
#include <cstring>

#include <spdlog/spdlog.h>

#include "App.hpp"


AppOptions AppOptions::parse(int argc, char** argv)
{
  AppOptions o;
  for (int i = 1; i < argc; ++i)
  {
    const char* a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc)
      {
        spdlog::error("Missing value after {}", a);
        std::exit(1);
      }
      return argv[++i];
    };
    if (std::strcmp(a, "--forward") == 0)
      o.overrides.deferredMode = false;
    else if (std::strcmp(a, "--naive") == 0)
      o.overrides.layout = GBufferLayout::Naive;
    else if (std::strcmp(a, "--no-decals") == 0)
      o.overrides.decals = false;
    else if (std::strcmp(a, "--debug-view") == 0)
      o.overrides.debugView = std::atoi(next());
    else if (std::strcmp(a, "--no-gui") == 0)
      o.hideGui = true;
    else if (std::strcmp(a, "--print-timings") == 0)
      o.printTimings = true;
    else if (std::strcmp(a, "--frames") == 0)
      o.exitAfterFrames = std::atoi(next());
    else if (std::strcmp(a, "--screenshot") == 0)
      o.screenshotPath = next();
    else if (std::strcmp(a, "--sun") == 0)
    {
      const float az = static_cast<float>(std::atof(next()));
      const float el = static_cast<float>(std::atof(next()));
      o.overrides.sun = glm::vec2(az, el);
    }
    else if (std::strcmp(a, "--coverage") == 0)
      o.overrides.cloudCoverage = static_cast<float>(std::atof(next()));
    else if (std::strcmp(a, "--no-clouds") == 0)
      o.overrides.clouds = false;
    else if (std::strcmp(a, "--sky-size") == 0)
      o.skyFaceSize = static_cast<std::uint32_t>(std::atoi(next()));
    else if (std::strcmp(a, "--no-tonemap") == 0)
      o.overrides.tonemapEnabled = false;
    else if (std::strcmp(a, "--no-grading") == 0)
      o.overrides.colorGrading = false;
    else if (std::strcmp(a, "--lut") == 0)
      o.overrides.lut = std::string(next());
    else if (std::strcmp(a, "--tonemap-op") == 0)
      o.overrides.tonemapOperator = std::atoi(next());
    else if (std::strcmp(a, "--no-dither") == 0)
      o.overrides.dithering = false;
    else if (std::strcmp(a, "--global-atomics") == 0)
      o.overrides.sharedMemory = false;
    else if (std::strcmp(a, "--no-shadows") == 0)
      o.overrides.shadows = false;
    else if (std::strcmp(a, "--pcf") == 0)
      o.overrides.pcfRadius = std::atoi(next());
    else if (std::strcmp(a, "--shadow-bias") == 0)
      o.overrides.shadowDepthBias = static_cast<float>(std::atof(next()));
    else if (std::strcmp(a, "--normal-bias") == 0)
      o.overrides.shadowNormalBias = static_cast<float>(std::atof(next()));
    else if (std::strcmp(a, "--no-grass") == 0)
      o.overrides.grass = false;
    else if (std::strcmp(a, "--no-grass-prepass") == 0)
      o.overrides.grassPrepass = false;
    else if (std::strcmp(a, "--grass-distance") == 0)
      o.overrides.grassDistance = static_cast<float>(std::atof(next()));
    else if (std::strcmp(a, "--grass-density") == 0)
      o.overrides.grassDensity = static_cast<float>(std::atof(next()));
    else if (std::strcmp(a, "--no-srgb") == 0)
      o.overrides.srgbEncoding = false;
    else if (std::strcmp(a, "--camera") == 0)
    {
      float v[6];
      for (float& x : v)
        x = static_cast<float>(std::atof(next()));
      o.camera = std::pair{glm::vec3(v[0], v[1], v[2]), glm::vec3(v[3], v[4], v[5])};
    }
    else
      spdlog::warn("Unknown argument {}", a);
  }
  return o;
}

int main(int argc, char** argv)
{
  {
    App app(AppOptions::parse(argc, argv));
    app.run();
  }

  // Etna needs to be de-initialized after all resources allocated by app
  // and it's sub-fields are already freed.
  if (etna::is_initilized())
    etna::shutdown();

  return 0;
}
