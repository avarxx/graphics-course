include(${CMAKE_CURRENT_LIST_DIR}/cmake/PolyHaven.cmake)

set(INSTANCING_ASSETS_DIR "${CMAKE_CURRENT_BINARY_DIR}/assets")

foreach(asset sand_03 sparse_grass rock_06 snow_01)
  polyhaven_fetch_texture(${asset} "${INSTANCING_ASSETS_DIR}" Diffuse Rough Displacement)
endforeach()

foreach(asset boulder_01 rock_09)
  polyhaven_fetch_model(${asset} "${INSTANCING_ASSETS_DIR}")
endforeach()

add_executable(instancing
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  HeightmapGenerator.cpp
  TextureLoader.cpp
  RockModel.cpp
)

target_link_libraries(instancing
  PRIVATE glfw etna glm::glm wsi gui scene render_utils tinygltf)

target_include_directories(instancing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(instancing PRIVATE INSTANCING_ASSETS_ROOT="${INSTANCING_ASSETS_DIR}/")

target_add_shaders(instancing
  shaders/terrain.vert
  shaders/terrain.frag
  shaders/rock.vert
  shaders/rock.frag
)
