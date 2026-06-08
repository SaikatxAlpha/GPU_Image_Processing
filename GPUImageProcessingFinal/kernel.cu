// kernel.cu  –  GPU Image Processing Kernels  v2.0
// Compile with NVCC as part of a Visual Studio CUDA project.
// Requires compute capability 3.0+  (atomics, constant memory).

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include <math.h>
#include <stdio.h>

// ============================================================
//  CUDA error-check macro
// ============================================================
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess)                                              \
            printf("CUDA error  %s:%d  %s\n",                              \
                   __FILE__, __LINE__, cudaGetErrorString(_e));             \
    } while (0)

// ============================================================
//  Shared grid/block helpers
// ============================================================
static inline dim3 makeGrid(int w, int h, int bx = 16, int by = 16)
{
    return dim3((w + bx - 1) / bx, (h + by - 1) / by);
}
static const dim3 BLOCK16(16, 16);


// ============================================================
//  1.  GRAYSCALE CONVERSION
//      One thread per pixel.
//      Input : BGR interleaved  (3 bytes/pixel)
//      Output: luminance        (1 byte/pixel)
// ============================================================
__global__ void grayscaleKernel(const unsigned char* input,
    unsigned char* output,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    // ITU-R BT.601 luminance weights
    output[y * width + x] =
        (unsigned char)(0.114f * input[idx + 0]     // B
            + 0.587f * input[idx + 1]     // G
            + 0.299f * input[idx + 2]);   // R
}

void launchGrayscaleKernel(const unsigned char* d_in, unsigned char* d_out,
    int w, int h)
{
    grayscaleKernel << <makeGrid(w, h), BLOCK16 >> > (d_in, d_out, w, h);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ============================================================
//  2.  5×5 GAUSSIAN BLUR  (naive – constant memory kernel)
//      Works for any number of channels.
// ============================================================
__constant__ float c_gaussian[25] = {
     1 / 273.f,  4 / 273.f,  7 / 273.f,  4 / 273.f, 1 / 273.f,
     4 / 273.f, 16 / 273.f, 26 / 273.f, 16 / 273.f, 4 / 273.f,
     7 / 273.f, 26 / 273.f, 41 / 273.f, 26 / 273.f, 7 / 273.f,
     4 / 273.f, 16 / 273.f, 26 / 273.f, 16 / 273.f, 4 / 273.f,
     1 / 273.f,  4 / 273.f,  7 / 273.f,  4 / 273.f, 1 / 273.f
};

__global__ void gaussianBlurKernel(const unsigned char* input,
    unsigned char* output,
    int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        float acc = 0.f;
        for (int ky = -2; ky <= 2; ky++)
            for (int kx = -2; kx <= 2; kx++) {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);
                acc += (float)input[(ny * width + nx) * channels + c]
                    * c_gaussian[(ky + 2) * 5 + (kx + 2)];
            }
        output[(y * width + x) * channels + c] =
            (unsigned char)fminf(fmaxf(acc, 0.f), 255.f);
    }
}

void launchGaussianBlurKernel(const unsigned char* d_in, unsigned char* d_out,
    int w, int h, int ch)
{
    gaussianBlurKernel << <makeGrid(w, h), BLOCK16 >> > (d_in, d_out, w, h, ch);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ============================================================
//  3.  SOBEL EDGE DETECTION  (grayscale)
// ============================================================
__global__ void sobelKernel(const unsigned char* input,
    unsigned char* output,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    int gx = -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)]
        - 2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)]
        - input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

    int gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
        + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

    output[y * width + x] =
        (unsigned char)fminf(sqrtf((float)(gx * gx + gy * gy)), 255.f);
}

void launchSobelKernel(const unsigned char* d_in, unsigned char* d_out,
    int w, int h)
{
    sobelKernel << <makeGrid(w, h), BLOCK16 >> > (d_in, d_out, w, h);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ============================================================
//  4.  BRIGHTNESS & CONTRAST
// ============================================================
__global__ void brightnessContrastKernel(const unsigned char* input,
    unsigned char* output,
    int width, int height, int channels,
    float alpha, int beta)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        int   idx = (y * width + x) * channels + c;
        float v = alpha * (float)input[idx] + (float)beta;
        output[idx] = (unsigned char)fminf(fmaxf(v, 0.f), 255.f);
    }
}

void launchBrightnessContrastKernel(const unsigned char* d_in, unsigned char* d_out,
    int w, int h, int ch, float alpha, int beta)
{
    brightnessContrastKernel << <makeGrid(w, h), BLOCK16 >> > (
        d_in, d_out, w, h, ch, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ============================================================
//  5.  HISTOGRAM EQUALIZATION  (grayscale)
//      Step 1 – histogram on GPU (atomic adds)
//      Step 2 – CDF + LUT on CPU (256 values, trivial cost)
//      Step 3 – LUT remap on GPU
// ============================================================
__global__ void histogramKernel(const unsigned char* input, int* hist,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        atomicAdd(&hist[input[y * width + x]], 1);
}

__global__ void lutApplyKernel(const unsigned char* input,
    unsigned char* output,
    const int* lut,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        output[y * width + x] = (unsigned char)lut[input[y * width + x]];
}

void launchHistogramEqualizationKernel(const unsigned char* d_in, unsigned char* d_out,
    int w, int h)
{
    dim3 grid = makeGrid(w, h);

    int* d_hist;
    CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, 256 * sizeof(int)));
    histogramKernel << <grid, BLOCK16 >> > (d_in, d_hist, w, h);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_hist[256];
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    long long cdf[256] = {};
    cdf[0] = h_hist[0];
    for (int i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + h_hist[i];

    long long cdf_min = 0;
    for (int i = 0; i < 256; i++)
        if (cdf[i] > 0) { cdf_min = cdf[i]; break; }

    int total = w * h;
    int h_lut[256];
    for (int i = 0; i < 256; i++) {
        if (total == (int)cdf_min) { h_lut[i] = i; continue; }
        int v = (int)(((float)(cdf[i] - cdf_min) / (float)(total - cdf_min)) * 255.f + 0.5f);
        h_lut[i] = min(max(v, 0), 255);
    }

    int* d_lut;
    CUDA_CHECK(cudaMalloc(&d_lut, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_lut, h_lut, 256 * sizeof(int), cudaMemcpyHostToDevice));
    lutApplyKernel << <grid, BLOCK16 >> > (d_in, d_out, d_lut, w, h);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_hist);
    cudaFree(d_lut);
}


// ============================================================
//  6.  SHARED-MEMORY GAUSSIAN BLUR  ★ NEW ★
//
//  Each 16×16 block loads a (16+4)×(16+4) = 20×20 halo region
//  into shared memory before applying the same 5×5 kernel.
//
//  Why faster?
//    Naive: each global-memory pixel is read by up to 25 threads.
//    SMEM : each pixel is read from global memory exactly once
//           per block; all 25 accesses come from L1 shared mem.
//  Expected speedup: 1.5–3× on large images (bandwidth-bound).
// ============================================================
#define SMEM_TILE_W   16
#define SMEM_TILE_H   16
#define SMEM_HALO      2
#define SMEM_BLK_W    (SMEM_TILE_W + 2 * SMEM_HALO)   // 20
#define SMEM_BLK_H    (SMEM_TILE_H + 2 * SMEM_HALO)   // 20
#define SMEM_PIXELS   (SMEM_BLK_W  * SMEM_BLK_H)       // 400

__global__ void gaussianBlurSharedKernel(const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height, int channels)
{
    // 20×20 × 3 channels = 1200 bytes per block  (well within 48 KB shared mem)
    __shared__ unsigned char smem[SMEM_BLK_H * SMEM_BLK_W * 3];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * SMEM_TILE_W + tx;
    int nThreads = SMEM_TILE_W * SMEM_TILE_H;   // 256

    // ── Cooperative load: all threads fill the shared-memory tile ──
    // 400 pixels, 256 threads → each thread loads 1 or 2 pixels.
    for (int i = tid; i < SMEM_PIXELS; i += nThreads)
    {
        int sy = i / SMEM_BLK_W;
        int sx = i % SMEM_BLK_W;
        int gy = (int)blockIdx.y * SMEM_TILE_H + sy - SMEM_HALO;
        int gx = (int)blockIdx.x * SMEM_TILE_W + sx - SMEM_HALO;
        gy = min(max(gy, 0), height - 1);   // clamp-to-border padding
        gx = min(max(gx, 0), width - 1);
        for (int c = 0; c < channels; c++)
            smem[i * channels + c] = input[(gy * width + gx) * channels + c];
    }
    __syncthreads();   // ensure all data is visible before compute

    // ── Apply 5×5 kernel using shared memory ──
    int x = blockIdx.x * SMEM_TILE_W + tx;
    int y = blockIdx.y * SMEM_TILE_H + ty;
    if (x >= width || y >= height) return;

    int smem_cx = tx + SMEM_HALO;
    int smem_cy = ty + SMEM_HALO;

    for (int c = 0; c < channels; c++) {
        float acc = 0.f;
        for (int ky = -2; ky <= 2; ky++)
            for (int kx = -2; kx <= 2; kx++) {
                int si = ((smem_cy + ky) * SMEM_BLK_W + (smem_cx + kx)) * channels + c;
                acc += (float)smem[si] * c_gaussian[(ky + 2) * 5 + (kx + 2)];
            }
        output[(y * width + x) * channels + c] =
            (unsigned char)fminf(fmaxf(acc, 0.f), 255.f);
    }
}

void launchGaussianBlurSharedKernel(const unsigned char* d_in, unsigned char* d_out,
    int w, int h, int ch)
{
    dim3 block(SMEM_TILE_W, SMEM_TILE_H);
    dim3 grid = makeGrid(w, h, SMEM_TILE_W, SMEM_TILE_H);
    gaussianBlurSharedKernel << <grid, block >> > (d_in, d_out, w, h, ch);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Async (no sync) version for CUDA Streams pipelining
void launchGaussianBlurSharedAsync(const unsigned char* d_in, unsigned char* d_out,
    int w, int h, int ch, cudaStream_t stream)
{
    dim3 block(SMEM_TILE_W, SMEM_TILE_H);
    dim3 grid = makeGrid(w, h, SMEM_TILE_W, SMEM_TILE_H);
    // Launch on the specified stream — no device sync here
    gaussianBlurSharedKernel << <grid, block, 0, stream >> > (d_in, d_out, w, h, ch);
}


// ============================================================
//  7.  BILATERAL FILTER  ★ NEW ★
//
//  Edge-preserving smoothing.  Unlike Gaussian blur (purely
//  spatial), bilateral weighting also depends on pixel-value
//  similarity, so strong edges are protected.
//
//  weight(p,q) = G_spatial(||p-q||) × G_range(|I_p - I_q|)
//
//  sigma_s large  → blurs over a wider spatial neighbourhood
//  sigma_r large  → less sensitive to intensity differences
//                   (approaches plain Gaussian blur at ∞)
// ============================================================
__global__ void bilateralFilterKernel(const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height, int channels,
    float sigma_s, float sigma_r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int   radius = (int)ceilf(2.5f * sigma_s);
    float inv2ss = 1.f / (2.f * sigma_s * sigma_s);
    float inv2sr = 1.f / (2.f * sigma_r * sigma_r);

    for (int c = 0; c < channels; c++) {
        float wsum = 0.f, norm = 0.f;
        float Ip = (float)input[(y * width + x) * channels + c];

        for (int dy = -radius; dy <= radius; dy++)
            for (int dx = -radius; dx <= radius; dx++) {
                int   nx = min(max(x + dx, 0), width - 1);
                int   ny = min(max(y + dy, 0), height - 1);
                float Iq = (float)input[(ny * width + nx) * channels + c];
                float diff = Ip - Iq;
                // Spatial × range weight
                float w = expf(-(float)(dx * dx + dy * dy) * inv2ss
                    - diff * diff * inv2sr);
                wsum += Iq * w;
                norm += w;
            }
        output[(y * width + x) * channels + c] =
            (unsigned char)fminf(fmaxf(wsum / norm, 0.f), 255.f);
    }
}

void launchBilateralFilterKernel(const unsigned char* d_in, unsigned char* d_out,
    int w, int h, int ch, float ss, float sr)
{
    bilateralFilterKernel << <makeGrid(w, h), BLOCK16 >> > (d_in, d_out, w, h, ch, ss, sr);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ============================================================
//  8.  CANNY EDGE DETECTION  ★ NEW ★  (full multi-pass GPU pipeline)
//
//  All passes run on-device with zero CPU involvement between them.
//  Memory allocations are done once by the launcher.
//
//  PASS 1  cannyGradientKernel     Sobel Gx/Gy → magnitude + direction (0°/45°/90°/135°)
//  PASS 2  nonMaxSuppressKernel    Non-maximum suppression (edge thinning)
//  PASS 3  doubleThresholdKernel   Classify: strong (255) / weak (64) / none (0)
//  PASS 4+ hysteresisKernel        Iterative: promote weak→strong if adjacent to strong
//  PASS 5  hysteresisCleanupKernel Remove remaining weak pixels
// ============================================================

// ── Pass 1: gradient magnitude + quantised direction ─────────
__global__ void cannyGradientKernel(const unsigned char* input,
    float* magnitude,
    unsigned char* direction,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        magnitude[y * width + x] = 0.f;
        direction[y * width + x] = 0;
        return;
    }

    int gx = -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)]
        - 2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)]
        - input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

    int gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
        + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

    magnitude[y * width + x] = sqrtf((float)(gx * gx + gy * gy));

    // Quantise angle into 4 compass directions
    float angle = atan2f((float)gy, (float)gx) * (180.f / 3.14159265f);
    if (angle < 0.f) angle += 180.f;

    unsigned char dir;
    if (angle < 22.5f || angle >= 157.5f) dir = 0;   // horizontal  ─
    else if (angle < 67.5f)                    dir = 1;   // diagonal   ╱
    else if (angle < 112.5f)                    dir = 2;   // vertical   │
    else                                         dir = 3;   // diagonal   ╲
    direction[y * width + x] = dir;
}

// ── Pass 2: non-maximum suppression (thin edges to 1px) ──────
__global__ void nonMaxSuppressKernel(const float* magnitude,
    float* nms_out,
    const unsigned char* direction,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        nms_out[y * width + x] = 0.f; return;
    }

    float mag = magnitude[y * width + x];
    int   dir = direction[y * width + x];
    float n1, n2;

    switch (dir) {
    case 0:  n1 = magnitude[y * width + (x - 1)];  // E/W
        n2 = magnitude[y * width + (x + 1)]; break;
    case 1:  n1 = magnitude[(y - 1) * width + (x + 1)];  // NE/SW
        n2 = magnitude[(y + 1) * width + (x - 1)]; break;
    case 2:  n1 = magnitude[(y - 1) * width + x];  // N/S
        n2 = magnitude[(y + 1) * width + x]; break;
    default: n1 = magnitude[(y - 1) * width + (x - 1)];  // NW/SE
        n2 = magnitude[(y + 1) * width + (x + 1)]; break;
    }
    nms_out[y * width + x] = (mag >= n1 && mag >= n2) ? mag : 0.f;
}

// ── Pass 3: double threshold ──────────────────────────────────
#define CANNY_STRONG 255
#define CANNY_WEAK    64
#define CANNY_NONE     0

__global__ void doubleThresholdKernel(const float* nms,
    unsigned char* out,
    int width, int height,
    float lo, float hi)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float v = nms[y * width + x];
    if (v >= hi) out[y * width + x] = CANNY_STRONG;
    else if (v >= lo) out[y * width + x] = CANNY_WEAK;
    else              out[y * width + x] = CANNY_NONE;
}

// ── Pass 4: hysteresis – one propagation sweep ───────────────
__global__ void hysteresisKernel(unsigned char* edges, int* changed,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) return;
    if (edges[y * width + x] != CANNY_WEAK) return;

    // Promote weak edge to strong if any 8-neighbour is strong
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            if (edges[(y + dy) * width + (x + dx)] == CANNY_STRONG) {
                edges[y * width + x] = CANNY_STRONG;
                atomicAdd(changed, 1);
                return;
            }
}

// ── Pass 5: discard remaining weak pixels ────────────────────
__global__ void hysteresisCleanupKernel(unsigned char* edges, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height && edges[y * width + x] == CANNY_WEAK)
        edges[y * width + x] = CANNY_NONE;
}

void launchCannyKernel(const unsigned char* d_in, unsigned char* d_out,
    int w, int h, float lo, float hi)
{
    dim3 grid = makeGrid(w, h);
    size_t nPx = (size_t)w * h;

    // Pass 0: Gaussian pre-blur to reduce noise
    unsigned char* d_blur;
    CUDA_CHECK(cudaMalloc(&d_blur, nPx));
    launchGaussianBlurKernel(d_in, d_blur, w, h, 1);

    // Pass 1: gradient
    float* d_mag;
    unsigned char* d_dir;
    CUDA_CHECK(cudaMalloc(&d_mag, nPx * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dir, nPx));
    cannyGradientKernel << <grid, BLOCK16 >> > (d_blur, d_mag, d_dir, w, h);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pass 2: NMS
    float* d_nms;
    CUDA_CHECK(cudaMalloc(&d_nms, nPx * sizeof(float)));
    nonMaxSuppressKernel << <grid, BLOCK16 >> > (d_mag, d_nms, d_dir, w, h);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pass 3: double threshold → d_out gets strong/weak/none labels
    doubleThresholdKernel << <grid, BLOCK16 >> > (d_nms, d_out, w, h, lo, hi);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pass 4: hysteresis – iterate until no more weak→strong promotions
    int* d_changed;
    int  h_changed;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    for (int iter = 0; iter < 128; iter++) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        hysteresisKernel << <grid, BLOCK16 >> > (d_out, d_changed, w, h);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
            cudaMemcpyDeviceToHost));
        if (h_changed == 0) break;   // converged
    }

    // Pass 5: cleanup
    hysteresisCleanupKernel << <grid, BLOCK16 >> > (d_out, w, h);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_blur); cudaFree(d_mag); cudaFree(d_dir);
    cudaFree(d_nms);  cudaFree(d_changed);
}


// ============================================================
//  9.  MORPHOLOGICAL EROSION  ★ NEW ★
//      Minimum over a (2·radius+1)² square structuring element.
//      Dark regions expand; bright regions shrink.
// ============================================================
__global__ void erosionKernel(const unsigned char* input,
    unsigned char* output,
    int width, int height, int channels, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        unsigned char minVal = 255;
        for (int dy = -radius; dy <= radius; dy++)
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);
                unsigned char v = input[(ny * width + nx) * channels + c];
                if (v < minVal) minVal = v;
            }
        output[(y * width + x) * channels + c] = minVal;
    }
}

void launchErosionKernel(const unsigned char* d_in, unsigned char* d_out,
    int w, int h, int ch, int radius)
{
    erosionKernel << <makeGrid(w, h), BLOCK16 >> > (d_in, d_out, w, h, ch, radius);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ============================================================
//  10. MORPHOLOGICAL DILATION  ★ NEW ★
//      Maximum over a (2·radius+1)² square structuring element.
//      Bright regions expand; dark regions shrink.
// ============================================================
__global__ void dilationKernel(const unsigned char* input,
    unsigned char* output,
    int width, int height, int channels, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        unsigned char maxVal = 0;
        for (int dy = -radius; dy <= radius; dy++)
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);
                unsigned char v = input[(ny * width + nx) * channels + c];
                if (v > maxVal) maxVal = v;
            }
        output[(y * width + x) * channels + c] = maxVal;
    }
}

void launchDilationKernel(const unsigned char* d_in, unsigned char* d_out,
    int w, int h, int ch, int radius)
{
    dilationKernel << <makeGrid(w, h), BLOCK16 >> > (d_in, d_out, w, h, ch, radius);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ============================================================
//  11. UNSHARP MASKING  ★ NEW ★
//
//  Sharpening by subtracting a blurred copy:
//      out = clamp( in + strength × (in – GaussianBlur(in)), 0, 255 )
//
//  Uses the shared-memory Gaussian internally for better performance.
//  strength  0.5 = subtle,  1.0 = moderate,  2.0+ = strong
// ============================================================
__global__ void unsharpMaskKernel(const unsigned char* orig,
    const unsigned char* blurred,
    unsigned char* output,
    int width, int height, int channels,
    float strength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        int   idx = (y * width + x) * channels + c;
        float o = (float)orig[idx];
        float b = (float)blurred[idx];
        float v = o + strength * (o - b);
        output[idx] = (unsigned char)fminf(fmaxf(v, 0.f), 255.f);
    }
}

void launchUnsharpMaskKernel(const unsigned char* d_in, unsigned char* d_out,
    int w, int h, int ch, float strength)
{
    size_t sz = (size_t)w * h * ch;

    unsigned char* d_blur;
    CUDA_CHECK(cudaMalloc(&d_blur, sz));

    // Use SMEM Gaussian for the blur step
    launchGaussianBlurSharedKernel(d_in, d_blur, w, h, ch);

    unsharpMaskKernel << <makeGrid(w, h), BLOCK16 >> > (d_in, d_blur, d_out, w, h, ch, strength);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_blur);
}
