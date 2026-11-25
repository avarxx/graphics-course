#pragma once

#include "wsi/OsWindowingManager.hpp"
#include "scene/Camera.hpp"

#include "AppOptions.hpp"
#include "Renderer.hpp"


class App
{
public:
  explicit App(AppOptions options = {});

  void run();

private:
  void processInput(float dt);
  void drawFrame(float dt);

  void moveCam(Camera& cam, const Keyboard& kb, float dt);
  void rotateCam(Camera& cam, const Mouse& ms, float dt);

private:
  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> mainWindow;

  float camMoveSpeed = 30;
  float camRotateSpeed = 0.1f;
  float zoomSensitivity = 2.0f;
  Camera mainCam;

  std::unique_ptr<Renderer> renderer;

  AppOptions options;
  int frameCount = 0;
};
