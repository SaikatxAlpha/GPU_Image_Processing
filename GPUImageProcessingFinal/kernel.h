#pragma once
#include <cuda_runtime.h>

// ============================================================
//  GPU Image Processing  –  CUDA Kernel Declarations  v2.0
//  All d_* pointers must reside in device (GPU) memory.
//  Image layout: row-major HWC  (height × width × channels).
// ============================================================


// ── ORIGINAL KERNELS ────────────────────────────────────────

// BGR → Grayscale  (3-ch in, 1-ch out)
void launchGrayscaleKernel(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height);

// 5×5 Gaussian Blur – naive global-memory version (any channel count)
void launchGaussianBlurKernel(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height, int channels);

// Sobel Edge Detection  (grayscale in/out)
void launchSobelKernel(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height);

// Brightness & Contrast:  out = clamp(alpha·in + beta, 0, 255)
void launchBrightnessContrastKernel(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height, int channels,
    float alpha, int beta);

// Histogram Equalization  (grayscale in/out)
void launchHistogramEqualizationKernel(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height);


// ── NEW KERNELS ──────────────────────────────────────────────

// 5×5 Gaussian Blur – tiled SHARED-MEMORY version (significantly faster)
// Reduces global-memory bandwidth ~25× vs naive for a 5×5 kernel.
void launchGaussianBlurSharedKernel(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height, int channels);

// Async (stream-aware) Gaussian Blur for CUDA Streams pipelining demo.
// Does NOT synchronize – caller is responsible for stream/device sync.
void launchGaussianBlurSharedAsync(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height, int channels,
    cudaStream_t stream);

// Bilateral Filter – edge-preserving smoothing (any channel count)
//   sigma_s : spatial Gaussian std-dev   (e.g. 10.f  → larger = more blur radius)
//   sigma_r : intensity range  std-dev   (e.g. 75.f  → larger = less edge preservation)
void launchBilateralFilterKernel(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height, int channels,
    float sigma_s, float sigma_r);

// Full Canny Edge Detection – complete GPU pipeline, no CPU involvement between passes
//   Pass 1: Gaussian pre-blur
//   Pass 2: Sobel gradient magnitude + quantised direction
//   Pass 3: Non-Maximum Suppression (edge thinning)
//   Pass 4: Double threshold   → strong / weak / suppressed
//   Pass 5+: Hysteresis        → iterative until convergence
//   Pass 6: Cleanup remaining weak pixels
//   Expects grayscale input; outputs grayscale edge map.
void launchCannyKernel(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height,
    float low_thresh, float high_thresh);

// Morphological Erosion  (square structuring element, any channel count)
//   radius 1 → 3×3 SE,  radius 2 → 5×5 SE,  etc.
void launchErosionKernel(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height, int channels, int radius);

// Morphological Dilation  (square structuring element, any channel count)
void launchDilationKernel(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height, int channels, int radius);

// Unsharp Masking (sharpening):
//   out = clamp( in + strength × (in – GaussianBlur(in)), 0, 255 )
//   strength 0.5–1.0 = subtle,  1.5–2.5 = strong sharpening
void launchUnsharpMaskKernel(const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height, int channels,
    float strength);