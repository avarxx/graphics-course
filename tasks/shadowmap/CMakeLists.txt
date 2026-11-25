include(${CMAKE_CURRENT_LIST_DIR}/cmake/PolyHaven.cmake)

set(SHADOWMAP_ASSETS_DIR "${CMAKE_CURRENT_BINARY_DIR}/assets")

foreach(asset sand_03 sparse_grass rock_06 snow_01)
  polyhaven_fetch_texture(${asset} "${SHADOWMAP_ASSETS_DIR}" Diffuse Rough Displacement)
endforeach()

foreach(asset boulder_01 rock_09)
  polyhaven_fetch_model(${asset} "${SHADOWMAP_ASSETS_DIR}")
endforeach()

add_executable(shadowmap_task
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
  SkyBaker.cpp
  Tonemapper.cpp
  ColorLut.cpp
)

target_link_libraries(shadowmap_task
  PRIVATE glfw etna glm::glm wsi gui scene render_utils tinygltf)

target_include_directories(shadowmap_task PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# The sample in samples/shadowmap already owns the plain "shadowmap" target name
set_target_properties(shadowmap_task PROPERTIES OUTPUT_NAME shadowmap)

target_compile_definitions(shadowmap_task PRIVATE
  SHADOWMAP_ASSETS_ROOT="${SHADOWMAP_ASSETS_DIR}/"
  SHADOWMAP_LUTS_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/luts/")

target_add_shaders(shadowmap_task
  shaders/terrain.vert
  shaders/terrain_shadow.vert
  shaders/terrain_gbuffer.frag
  shaders/terrain_forward.frag
  shaders/rock.vert
  shaders/rock_shadow.vert
  shaders/rock_gbuffer.frag
  shaders/rock_forward.frag
  shaders/fullscreen.vert
  shaders/deferred_lighting.frag
  shaders/decal.vert
  shaders/decal.frag
  shaders/sky_forward.frag
  shaders/sky_bake.comp
  shaders/minmax_global.comp
  shaders/minmax_shared.comp
  shaders/histogram_global.comp
  shaders/histogram_shared.comp
  shaders/exposure.comp
  shaders/postprocess.frag
)
