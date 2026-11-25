// Pass 2: histogram of log2 luminance with HIST_BINS bins spread uniformly between the
// minimum and the maximum found by pass 1.
// USE_SHARED_MEMORY = 1: the group accumulates a local histogram in shared memory and then adds
// only its non-empty bins to the global one. USE_SHARED_MEMORY = 0: one global atomic per pixel,
// heavily contended because neighboring pixels mostly fall into the same few bins.

#include "luminance_common.glsl"

#if USE_SHARED_MEMORY
shared uint sHist[HIST_BINS];
#endif

void main()
{
  float logLum;
  const bool inside = loadLogLuminance(logLum);

  const float lo = orderedUintToFloat(state.minLogBits);
  const float hi = orderedUintToFloat(state.maxLogBits);
  const float range = max(hi - lo, 1e-4);
  const int bin = clamp(int((logLum - lo) / range * float(HIST_BINS)), 0, HIST_BINS - 1);

#if USE_SHARED_MEMORY
  const uint i = gl_LocalInvocationIndex;
  if (i < HIST_BINS)
    sHist[i] = 0u;
  barrier();

  if (inside)
    atomicAdd(sHist[bin], 1u);
  barrier();

  if (i < HIST_BINS && sHist[i] != 0u)
    atomicAdd(state.histogram[i], sHist[i]);
#else
  if (inside)
    atomicAdd(state.histogram[bin], 1u);
#endif
}
