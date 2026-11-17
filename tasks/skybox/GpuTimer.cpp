#include "GpuTimer.hpp"

#include <etna/GlobalContext.hpp>


GpuTimer::~GpuTimer()
{
  for (auto& s : slots)
    if (s.pool)
      device.destroyQueryPool(s.pool);
}

void GpuTimer::init(std::size_t frames_in_flight)
{
  auto& ctx = etna::get_context();
  device = ctx.getDevice();

  const auto props = ctx.getPhysicalDevice().getProperties();
  const auto queues = ctx.getPhysicalDevice().getQueueFamilyProperties();
  const std::uint32_t validBits = queues[ctx.getQueueFamilyIdx()].timestampValidBits;
  isSupported = props.limits.timestampComputeAndGraphics == vk::True && validBits > 0;
  periodNs = static_cast<double>(props.limits.timestampPeriod);
  validMask = validBits >= 64 ? ~0ull : ((1ull << validBits) - 1ull);

  if (!isSupported)
    return;

  slots.resize(frames_in_flight);
  for (auto& s : slots)
  {
    s.pool = etna::unwrap_vk_result(device.createQueryPool(vk::QueryPoolCreateInfo{
      .queryType = vk::QueryType::eTimestamp,
      .queryCount = MAX_SECTIONS * 2,
    }));
    s.names.resize(MAX_SECTIONS);
  }
}

void GpuTimer::beginFrame(vk::CommandBuffer cmd_buf)
{
  if (!isSupported)
    return;

  // Same multi-buffering index as etna::GpuSharedResource: the GPU is done with this slot
  current = etna::get_context().getMainWorkCount().currentResource() % slots.size();
  Slot& slot = slots[current];

  if (slot.pending && slot.used > 0)
  {
    std::array<std::uint64_t, MAX_SECTIONS * 2> stamps{};
    const auto res = device.getQueryPoolResults(
      slot.pool,
      0,
      slot.used * 2,
      sizeof(std::uint64_t) * slot.used * 2,
      stamps.data(),
      sizeof(std::uint64_t),
      vk::QueryResultFlagBits::e64);
    if (res == vk::Result::eSuccess)
    {
      for (std::uint32_t i = 0; i < slot.used; ++i)
      {
        const std::uint64_t ticks = (stamps[2 * i + 1] - stamps[2 * i]) & validMask;
        const float ms = static_cast<float>(static_cast<double>(ticks) * periodNs * 1e-6);

        Section* section = nullptr;
        for (auto& r : results)
          if (r.name == slot.names[i])
            section = &r;
        if (section == nullptr)
        {
          results.push_back(Section{.name = slot.names[i], .smoothedMs = ms});
          section = &results.back();
        }
        section->lastMs = ms;
        section->smoothedMs += (ms - section->smoothedMs) * 0.05f;
      }
    }
  }

  cmd_buf.resetQueryPool(slot.pool, 0, MAX_SECTIONS * 2);
  slot.used = 0;
  slot.pending = true;
  openStack.clear();
}

void GpuTimer::begin(vk::CommandBuffer cmd_buf, std::string_view name)
{
  if (!isSupported)
    return;
  Slot& slot = slots[current];
  if (slot.used >= MAX_SECTIONS)
    return;
  const std::uint32_t idx = slot.used++;
  slot.names[idx] = name;
  openStack.push_back(idx);
  cmd_buf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, slot.pool, idx * 2);
}

void GpuTimer::end(vk::CommandBuffer cmd_buf)
{
  if (!isSupported || openStack.empty())
    return;
  const std::uint32_t idx = openStack.back();
  openStack.pop_back();
  cmd_buf.writeTimestamp2(
    vk::PipelineStageFlagBits2::eAllCommands, slots[current].pool, idx * 2 + 1);
}

void GpuTimer::resetAverages()
{
  for (auto& r : results)
    r.smoothedMs = r.lastMs;
}
