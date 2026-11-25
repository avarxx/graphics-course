#ifndef LUMINANCE_GLSL_INCLUDED
#define LUMINANCE_GLSL_INCLUDED

// "Brightness" of a linear RGB color = relative luminance Y of CIE 1931 XYZ.
// RGB channels are not perceived equally bright: for the sRGB / Rec.709 primaries
// Y = 0.2126 R + 0.7152 G + 0.0722 B (only valid for LINEAR values, not gamma-encoded ones).
float luminance(vec3 linearRgb)
{
  return dot(linearRgb, vec3(0.2126, 0.7152, 0.0722));
}

// The histogram is built over log2(Y): an HDR frame spans many orders of magnitude, and the eye
// perceives ratios, not differences (Weber-Fechner), so equal bins in log space are equal
// perceptual steps. Linear bins would squeeze all the dark pixels into the first bin.
const float MIN_LOG_LUM = -16.0;
const float MAX_LOG_LUM = 16.0;

float logLuminance(vec3 linearRgb)
{
  return clamp(log2(max(luminance(linearRgb), 1e-6)), MIN_LOG_LUM, MAX_LOG_LUM);
}

// Atomic min/max exist only for integers. This mapping keeps the order of floats:
// positive floats get the sign bit set, negative ones are inverted.
uint floatToOrderedUint(float f)
{
  const uint u = floatBitsToUint(f);
  return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

float orderedUintToFloat(uint u)
{
  return uintBitsToFloat((u & 0x80000000u) != 0u ? (u & 0x7FFFFFFFu) : ~u);
}

#endif // LUMINANCE_GLSL_INCLUDED
