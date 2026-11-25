// Pass 1: minimum and maximum log2 luminance of the HDR image.
// USE_SHARED_MEMORY = 1: tree reduction in shared memory, one pair of global atomics per group.
// USE_SHARED_MEMORY = 0: every pixel does its own global atomics (the naive variant).

#include "luminance_common.glsl"

#if USE_SHARED_MEMORY
const uint GROUP_THREADS = LUM_GROUP_SIZE * LUM_GROUP_SIZE;
shared float sMin[GROUP_THREADS];
shared float sMax[GROUP_THREADS];
#endif

void main()
{
  float logLum;
  const bool inside = loadLogLuminance(logLum);

#if USE_SHARED_MEMORY
  const uint i = gl_LocalInvocationIndex;
  sMin[i] = inside ? logLum : MAX_LOG_LUM;
  sMax[i] = inside ? logLum : MIN_LOG_LUM;
  barrier();

  for (uint stride = GROUP_THREADS / 2; stride > 0; stride >>= 1)
  {
    if (i < stride)
    {
      sMin[i] = min(sMin[i], sMin[i + stride]);
      sMax[i] = max(sMax[i], sMax[i + stride]);
    }
    barrier();
  }

  if (i == 0)
  {
    atomicMin(state.minLogBits, floatToOrderedUint(sMin[0]));
    atomicMax(state.maxLogBits, floatToOrderedUint(sMax[0]));
  }
#else
  if (inside)
  {
    atomicMin(state.minLogBits, floatToOrderedUint(logLum));
    atomicMax(state.maxLogBits, floatToOrderedUint(logLum));
  }
#endif
}
