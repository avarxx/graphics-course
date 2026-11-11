include(${CMAKE_CURRENT_LIST_DIR}/cmake/PolyHaven.cmake)

set(DEFERRED_ASSETS_DIR "${CMAKE_CURRENT_BINARY_DIR}/assets")

foreach(asset sand_03 sparse_grass rock_06 snow_01)
  polyhaven_fetch_texture(${asset} "${DEFERRED_ASSETS_DIR}" Diffuse Rough Displacement)
endforeach()

foreach(asset boulder_01 rock_09)
  polyhaven_fetch_model(${asset} "${DEFERRED_ASSETS_DIR}")
endforeach()

add_executable(deferred
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  HeightmapGenerator.cpp
  TextureLoader.cpp
  RockModel.cpp
  GBuffer.cpp
  Decals.cpp
  GpuTimer.cpp
)

target_link_libraries(deferred
  PRIVATE glfw etna glm::glm wsi gui scene render_utils tinygltf)

target_include_directories(deferred PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(deferred PRIVATE DEFERRED_ASSETS_ROOT="${DEFERRED_ASSETS_DIR}/")

target_add_shaders(deferred
  shaders/terrain.vert
  shaders/terrain_gbuffer.frag
  shaders/terrain_forward.frag
  shaders/rock.vert
  shaders/rock_gbuffer.frag
  shaders/rock_forward.frag
  shaders/fullscreen.vert
  shaders/deferred_lighting.frag
  shaders/decal.vert
  shaders/decal.frag
)
