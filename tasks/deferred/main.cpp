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
