#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include <etna/Vulkan.hpp>


/**
 * Minimal GPU pass timer on top of Vulkan timestamp queries, so that the cost of the G-buffer
 * passes can be compared right in ImGui (Tracy shows the same zones via ETNA_PROFILE_GPU).
 * One query pool per frame in flight: results of a frame are read when its slot is reused,
 * by which point etna has already waited for that frame to finish on the GPU.
 */
class GpuTimer
{
public:
  static constexpr std::uint32_t MAX_SECTIONS = 16;

  GpuTimer() = default;
  ~GpuTimer();

  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void init(std::size_t frames_in_flight);

  // Reads back the results of the frame that previously used this slot and resets the pool.
  void beginFrame(vk::CommandBuffer cmd_buf);
  void begin(vk::CommandBuffer cmd_buf, std::string_view name);
  void end(vk::CommandBuffer cmd_buf);

  struct Section
  {
    std::string_view name;
    float smoothedMs = 0.0f;
    float lastMs = 0.0f;
  };
  const std::vector<Section>& sections() const { return results; }
  bool supported() const { return isSupported; }

  // Drops the accumulated averages (e.g. after switching the G-buffer layout)
  void resetAverages();

private:
  struct Slot
  {
    vk::QueryPool pool;
    std::vector<std::string_view> names;
    std::uint32_t used = 0;
    bool pending = false;
  };

  vk::Device device;
  std::vector<Slot> slots;
  std::size_t current = 0;
  std::vector<std::uint32_t> openStack;
  std::vector<Section> results;
  double periodNs = 1.0;
  std::uint64_t validMask = ~0ull;
  bool isSupported = false;
};
