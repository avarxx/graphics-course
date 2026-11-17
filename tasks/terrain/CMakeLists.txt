include(${CMAKE_CURRENT_LIST_DIR}/cmake/PolyHaven.cmake)

set(TERRAIN_ASSETS_DIR "${CMAKE_CURRENT_BINARY_DIR}/assets")

foreach(asset sand_03 sparse_grass rock_06 snow_01)
  polyhaven_fetch_texture(${asset} "${TERRAIN_ASSETS_DIR}" Diffuse Rough Displacement)
endforeach()

add_executable(terrain
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  HeightmapGenerator.cpp
  TextureLoader.cpp
)

target_link_libraries(terrain
  PRIVATE glfw etna glm::glm wsi gui scene render_utils tinygltf)

target_include_directories(terrain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(terrain PRIVATE TERRAIN_ASSETS_ROOT="${TERRAIN_ASSETS_DIR}/")

target_add_shaders(terrain
  shaders/terrain.vert
  shaders/terrain.frag
)
