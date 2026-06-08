// main.cpp  –  GPU Image Processing Software  v2.0
// All OpenCV highgui calls live on one dedicated display thread.
// The main thread only does console I/O and GPU work.

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

#include "kernel.h"

// ============================================================
//  Display thread
//  Rule: every cv::namedWindow / cv::imshow / cv::waitKey /
//        cv::destroyAllWindows must be called from THIS thread.
//  The main thread writes to g_ds under the mutex; the display
//  thread reads it, calls imshow, then calls waitKey(30) to
//  keep the Win32 message pump running.
// ============================================================
struct DisplayState
{
    std::string title;
    cv::Mat     img;
    bool        needsShow = false;
    bool        needsDestroy = false;
};

static DisplayState      g_ds;
static std::mutex        g_dsMtx;
static std::atomic<bool> g_dtRunning(false);
static std::thread       g_dtThread;

static void displayThreadLoop()
{
    while (g_dtRunning.load())
    {
        {
            std::lock_guard<std::mutex> lk(g_dsMtx);

            if (g_ds.needsDestroy)
            {
                cv::destroyAllWindows();
                g_ds.needsDestroy = false;
            }

            if (g_ds.needsShow && !g_ds.img.empty())
            {
                cv::namedWindow(g_ds.title,
                    cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

                int dw = g_ds.img.cols, dh = g_ds.img.rows;
                if (dw > 1280 || dh > 720)
                {
                    float s = std::min(1280.0f / dw, 720.0f / dh);
                    dw = int(dw * s);
                    dh = int(dh * s);
                }
                cv::resizeWindow(g_ds.title, dw, dh);
                cv::imshow(g_ds.title, g_ds.img);
                g_ds.needsShow = false;
            }
        }
        cv::waitKey(30);
    }
    cv::destroyAllWindows();
}

static void showImage(const std::string& title, const cv::Mat& img)
{
    std::lock_guard<std::mutex> lk(g_dsMtx);
    g_ds.title = title;
    g_ds.img = img.clone();
    g_ds.needsShow = true;
}

static void destroyAll()
{
    std::lock_guard<std::mutex> lk(g_dsMtx);
    g_ds.needsDestroy = true;
    g_ds.needsShow = false;
}

static void startDisplayThread()
{
    g_dtRunning = true;
    g_dtThread = std::thread(displayThreadLoop);
}

static void stopDisplayThread()
{
    g_dtRunning = false;
    if (g_dtThread.joinable())
        g_dtThread.join();
}

// ============================================================
//  GPU info
// ============================================================
static void printGPUInfo()
{
    int n = 0;
    cudaGetDeviceCount(&n);
    if (n == 0) { std::cout << "  [!] No CUDA GPU found.\n"; return; }
    for (int i = 0; i < n; i++)
    {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        std::cout << "\n  GPU " << i << ": " << p.name << "\n";
        std::cout << "    Compute Capability : " << p.major << "." << p.minor << "\n";
        std::cout << "    Global Memory      : " << p.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "    Multiprocessors    : " << p.multiProcessorCount << "\n";
        std::cout << "    Max Threads/Block  : " << p.maxThreadsPerBlock << "\n";
        std::cout << "    Warp Size          : " << p.warpSize << "\n";
    }
}

// ============================================================
//  Timing helpers
// ============================================================
using Clock = std::chrono::high_resolution_clock;
static double ms(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ============================================================
//  CUDA wrappers  –  original kernels
// ============================================================
static cv::Mat doGrayscale(const cv::Mat& src)
{
    CV_Assert(src.channels() == 3);
    int W = src.cols, H = src.rows;
    unsigned char* di, * dout;
    cudaMalloc(&di, (size_t)W * H * 3);
    cudaMalloc(&dout, (size_t)W * H);
    cudaMemcpy(di, src.data, (size_t)W * H * 3, cudaMemcpyHostToDevice);
    launchGrayscaleKernel(di, dout, W, H);
    cv::Mat dst(H, W, CV_8UC1);
    cudaMemcpy(dst.data, dout, (size_t)W * H, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return dst;
}

static cv::Mat doGaussianBlur(const cv::Mat& src)
{
    int W = src.cols, H = src.rows, C = src.channels();
    size_t sz = (size_t)W * H * C;
    unsigned char* di = nullptr, * dout = nullptr;

    if (cudaMalloc(&di, sz) != cudaSuccess ||
        cudaMalloc(&dout, sz) != cudaSuccess)
    {
        std::cout << "  [!] cudaMalloc failed.\n";
        cudaFree(di); return src.clone();
    }
    cudaMemcpy(di, src.data, sz, cudaMemcpyHostToDevice);
    launchGaussianBlurKernel(di, dout, W, H, C);
    cudaDeviceSynchronize();

    cv::Mat dst(H, W, src.type());
    cudaMemcpy(dst.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return dst;
}

static cv::Mat doSobel(const cv::Mat& src)
{
    cv::Mat g = (src.channels() == 3) ? doGrayscale(src) : src.clone();
    int W = g.cols, H = g.rows;
    size_t sz = (size_t)W * H;
    unsigned char* di, * dout;
    cudaMalloc(&di, sz); cudaMalloc(&dout, sz);
    cudaMemcpy(di, g.data, sz, cudaMemcpyHostToDevice);
    launchSobelKernel(di, dout, W, H);
    cv::Mat dst(H, W, CV_8UC1);
    cudaMemcpy(dst.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return dst;
}

static cv::Mat doBrightnessContrast(const cv::Mat& src, float alpha, int beta)
{
    int W = src.cols, H = src.rows, C = src.channels();
    size_t sz = (size_t)W * H * C;
    unsigned char* di, * dout;
    cudaMalloc(&di, sz); cudaMalloc(&dout, sz);
    cudaMemcpy(di, src.data, sz, cudaMemcpyHostToDevice);
    launchBrightnessContrastKernel(di, dout, W, H, C, alpha, beta);
    cv::Mat dst(H, W, src.type());
    cudaMemcpy(dst.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return dst;
}

static cv::Mat doHistEq(const cv::Mat& src)
{
    cv::Mat g = (src.channels() == 3) ? doGrayscale(src) : src.clone();
    int W = g.cols, H = g.rows;
    size_t sz = (size_t)W * H;
    unsigned char* di, * dout;
    cudaMalloc(&di, sz); cudaMalloc(&dout, sz);
    cudaMemcpy(di, g.data, sz, cudaMemcpyHostToDevice);
    launchHistogramEqualizationKernel(di, dout, W, H);
    cv::Mat dst(H, W, CV_8UC1);
    cudaMemcpy(dst.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return dst;
}

// ============================================================
//  CUDA wrappers  –  NEW kernels
// ============================================================

// ── Shared-memory Gaussian Blur ──────────────────────────────
static cv::Mat doGaussianBlurShared(const cv::Mat& src)
{
    int W = src.cols, H = src.rows, C = src.channels();
    size_t sz = (size_t)W * H * C;
    unsigned char* di = nullptr, * dout = nullptr;

    if (cudaMalloc(&di, sz) != cudaSuccess ||
        cudaMalloc(&dout, sz) != cudaSuccess)
    {
        std::cout << "  [!] cudaMalloc failed.\n";
        cudaFree(di); return src.clone();
    }
    cudaMemcpy(di, src.data, sz, cudaMemcpyHostToDevice);
    launchGaussianBlurSharedKernel(di, dout, W, H, C);

    cv::Mat dst(H, W, src.type());
    cudaMemcpy(dst.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return dst;
}

// ── Bilateral Filter ─────────────────────────────────────────
static cv::Mat doBilateralFilter(const cv::Mat& src, float sigma_s, float sigma_r)
{
    int W = src.cols, H = src.rows, C = src.channels();
    size_t sz = (size_t)W * H * C;
    unsigned char* di = nullptr, * dout = nullptr;

    if (cudaMalloc(&di, sz) != cudaSuccess ||
        cudaMalloc(&dout, sz) != cudaSuccess)
    {
        std::cout << "  [!] cudaMalloc failed.\n";
        cudaFree(di); return src.clone();
    }
    cudaMemcpy(di, src.data, sz, cudaMemcpyHostToDevice);
    launchBilateralFilterKernel(di, dout, W, H, C, sigma_s, sigma_r);

    cv::Mat dst(H, W, src.type());
    cudaMemcpy(dst.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return dst;
}

// ── Canny Edge Detection ─────────────────────────────────────
// Input may be colour or grayscale; Canny expects single-channel.
static cv::Mat doCanny(const cv::Mat& src, float low_thresh, float high_thresh)
{
    cv::Mat g = (src.channels() == 3) ? doGrayscale(src) : src.clone();
    int W = g.cols, H = g.rows;
    size_t sz = (size_t)W * H;
    unsigned char* di = nullptr, * dout = nullptr;

    if (cudaMalloc(&di, sz) != cudaSuccess ||
        cudaMalloc(&dout, sz) != cudaSuccess)
    {
        std::cout << "  [!] cudaMalloc failed.\n";
        cudaFree(di); return src.clone();
    }
    cudaMemcpy(di, g.data, sz, cudaMemcpyHostToDevice);
    launchCannyKernel(di, dout, W, H, low_thresh, high_thresh);

    cv::Mat dst(H, W, CV_8UC1);
    cudaMemcpy(dst.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return dst;
}

// ── Morphological Erosion ────────────────────────────────────
static cv::Mat doErosion(const cv::Mat& src, int radius)
{
    int W = src.cols, H = src.rows, C = src.channels();
    size_t sz = (size_t)W * H * C;
    unsigned char* di = nullptr, * dout = nullptr;

    if (cudaMalloc(&di, sz) != cudaSuccess ||
        cudaMalloc(&dout, sz) != cudaSuccess)
    {
        std::cout << "  [!] cudaMalloc failed.\n";
        cudaFree(di); return src.clone();
    }
    cudaMemcpy(di, src.data, sz, cudaMemcpyHostToDevice);
    launchErosionKernel(di, dout, W, H, C, radius);

    cv::Mat dst(H, W, src.type());
    cudaMemcpy(dst.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return dst;
}

// ── Morphological Dilation ───────────────────────────────────
static cv::Mat doDilation(const cv::Mat& src, int radius)
{
    int W = src.cols, H = src.rows, C = src.channels();
    size_t sz = (size_t)W * H * C;
    unsigned char* di = nullptr, * dout = nullptr;

    if (cudaMalloc(&di, sz) != cudaSuccess ||
        cudaMalloc(&dout, sz) != cudaSuccess)
    {
        std::cout << "  [!] cudaMalloc failed.\n";
        cudaFree(di); return src.clone();
    }
    cudaMemcpy(di, src.data, sz, cudaMemcpyHostToDevice);
    launchDilationKernel(di, dout, W, H, C, radius);

    cv::Mat dst(H, W, src.type());
    cudaMemcpy(dst.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return dst;
}

// ── Unsharp Masking ──────────────────────────────────────────
static cv::Mat doUnsharpMask(const cv::Mat& src, float strength)
{
    int W = src.cols, H = src.rows, C = src.channels();
    size_t sz = (size_t)W * H * C;
    unsigned char* di = nullptr, * dout = nullptr;

    if (cudaMalloc(&di, sz) != cudaSuccess ||
        cudaMalloc(&dout, sz) != cudaSuccess)
    {
        std::cout << "  [!] cudaMalloc failed.\n";
        cudaFree(di); return src.clone();
    }
    cudaMemcpy(di, src.data, sz, cudaMemcpyHostToDevice);
    launchUnsharpMaskKernel(di, dout, W, H, C, strength);

    cv::Mat dst(H, W, src.type());
    cudaMemcpy(dst.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return dst;
}

// ============================================================
//  Helper: guard against operating on an empty image
// ============================================================
static bool requireImage(const cv::Mat& img)
{
    if (img.empty())
    {
        std::cout << "  [!] Load an image first (option 1).\n";
        return false;
    }
    return true;
}

// ============================================================
//  Menu
// ============================================================
static void printMenu(const cv::Mat& img)
{
    std::cout << "\n+----------------------------------------------------+\n";
    std::cout << "|        GPU Image Processing Software  v2.0        |\n";
    std::cout << "+----------------------------------------------------+\n";
    if (img.empty())
        std::cout << "|  Status  : No image loaded                         |\n";
    else
        std::cout << "|  Image   : " << img.cols << "x" << img.rows
        << "  (" << img.channels() << " ch)"
        << std::string(20, ' ') << "|\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "| Load / Save / Info                                 |\n";
    std::cout << "|  1.  Load Image                                    |\n";
    std::cout << "|  2.  Save Current Image                            |\n";
    std::cout << "|  3.  Show GPU Info                                 |\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "| Classic Filters                                    |\n";
    std::cout << "|  4.  Grayscale Conversion         (GPU)            |\n";
    std::cout << "|  5.  Gaussian Blur  – Naive       (GPU)            |\n";
    std::cout << "|  6.  Gaussian Blur  – Shared Mem  (GPU) faster     |\n";
    std::cout << "|  7.  Sobel Edge Detection         (GPU)            |\n";
    std::cout << "|  8.  Brightness & Contrast        (GPU)            |\n";
    std::cout << "|  9.  Histogram Equalization       (GPU)            |\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "| Advanced Filters                                   |\n";
    std::cout << "|  10. Bilateral Filter             (GPU)  new       |\n";
    std::cout << "|  11. Canny Edge Detection         (GPU)  new       |\n";
    std::cout << "|  12. Morphological Erosion        (GPU)  new       |\n";
    std::cout << "|  13. Morphological Dilation       (GPU)  new       |\n";
    std::cout << "|  14. Unsharp Masking (Sharpen)    (GPU)  new       |\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "|  0.  Exit                                          |\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "  Enter choice: ";
}

// ============================================================
//  Shared display-update helper
// ============================================================
static void applyAndShow(cv::Mat& current, const cv::Mat& result,
    const std::string& windowTitle)
{
    current = result;
    destroyAll();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    showImage(windowTitle, current);
}

// ============================================================
//  main
// ============================================================
int main()
{
    std::cout << "\n=== GPU Image Processing Software  v2.0 ===\n";
    printGPUInfo();

    startDisplayThread();

    cv::Mat current;
    int choice = 0;

    while (true)
    {
        printMenu(current);
        std::cin >> choice;

        if (std::cin.fail())
        {
            std::cin.clear();
            std::cin.ignore(1024, '\n');
            std::cout << "  [!] Please enter a number from the menu.\n";
            continue;
        }

        // ── 0: Exit ────────────────────────────────────────────
        if (choice == 0)
        {
            std::cout << "  Goodbye!\n";
            stopDisplayThread();
            break;
        }

        // ── 1: Load ────────────────────────────────────────────
        else if (choice == 1)
        {
            std::string path;
            std::cout << "  Enter full image path: ";
            std::cin.ignore();
            std::getline(std::cin, path);
            cv::Mat loaded = cv::imread(path);
            if (loaded.empty())
                std::cout << "  [!] Failed to load: " << path << "\n";
            else
            {
                current = loaded;
                std::cout << "  Loaded: " << current.cols << "x" << current.rows
                    << " (" << current.channels() << " ch)\n";
                applyAndShow(current, current, "Loaded Image");
            }
        }

        // ── 2: Save ────────────────────────────────────────────
        else if (choice == 2)
        {
            if (!requireImage(current)) continue;
            std::string outPath;
            std::cout << "  Enter save path (e.g. output.jpg): ";
            std::cin.ignore();
            std::getline(std::cin, outPath);
            if (cv::imwrite(outPath, current))
                std::cout << "  Saved: " << outPath << "\n";
            else
                std::cout << "  [!] Failed to save.\n";
        }

        // ── 3: GPU Info ────────────────────────────────────────
        else if (choice == 3)
        {
            printGPUInfo();
        }

        // ── 4: Grayscale ───────────────────────────────────────
        else if (choice == 4)
        {
            if (!requireImage(current)) continue;
            if (current.channels() != 3)
            {
                std::cout << "  [!] Image is already grayscale.\n"; continue;
            }
            auto t0 = Clock::now();
            cv::Mat r = doGrayscale(current);
            std::cout << "  Grayscale done in " << ms(t0) << " ms  (GPU)\n";
            applyAndShow(current, r, "Grayscale");
        }

        // ── 5: Gaussian Blur (naive) ───────────────────────────
        else if (choice == 5)
        {
            if (!requireImage(current)) continue;
            auto t0 = Clock::now();
            cv::Mat r = doGaussianBlur(current);
            std::cout << "  Gaussian Blur (naive) done in " << ms(t0) << " ms  (GPU)\n";
            applyAndShow(current, r, "Gaussian Blur – Naive");
        }

        // ── 6: Gaussian Blur (shared memory) NEW
        else if (choice == 6)
        {
            if (!requireImage(current)) continue;
            auto t0 = Clock::now();
            cv::Mat r = doGaussianBlurShared(current);
            std::cout << "  Gaussian Blur (shared-mem) done in " << ms(t0) << " ms  (GPU)\n";
            applyAndShow(current, r, "Gaussian Blur – Shared Mem");
        }

        // ── 7: Sobel ───────────────────────────────────────────
        else if (choice == 7)
        {
            if (!requireImage(current)) continue;
            auto t0 = Clock::now();
            cv::Mat r = doSobel(current);
            std::cout << "  Sobel Edge Detection done in " << ms(t0) << " ms  (GPU)\n";
            applyAndShow(current, r, "Sobel Edges");
        }

        // ── 8: Brightness & Contrast ───────────────────────────
        else if (choice == 8)
        {
            if (!requireImage(current)) continue;
            float alpha = 1.0f; int beta = 0;
            std::cout << "  Contrast  (alpha – 1.0 = no change, 1.5 = more contrast): ";
            std::cin >> alpha;
            std::cout << "  Brightness (beta  – 0   = no change,  50 = brighter)    : ";
            std::cin >> beta;
            auto t0 = Clock::now();
            cv::Mat r = doBrightnessContrast(current, alpha, beta);
            std::cout << "  Brightness/Contrast done in " << ms(t0) << " ms  (GPU)\n";
            applyAndShow(current, r, "Brightness / Contrast");
        }

        // ── 9: Histogram Equalization ──────────────────────────
        else if (choice == 9)
        {
            if (!requireImage(current)) continue;
            auto t0 = Clock::now();
            cv::Mat r = doHistEq(current);
            std::cout << "  Histogram Equalization done in " << ms(t0) << " ms  (GPU)\n";
            applyAndShow(current, r, "Histogram Equalization");
        }

        // ── 10: Bilateral Filter NEW 
        else if (choice == 10)
        {
            if (!requireImage(current)) continue;
            float sigma_s = 10.f, sigma_r = 75.f;
            std::cout << "  Spatial sigma  (sigma_s – e.g. 5.0–15.0, larger = wider blur radius) : ";
            std::cin >> sigma_s;
            std::cout << "  Range  sigma   (sigma_r – e.g. 50–100,   larger = less edge-preserve): ";
            std::cin >> sigma_r;
            std::cout << "  Running bilateral filter (may be slow on large images) ...\n";
            auto t0 = Clock::now();
            cv::Mat r = doBilateralFilter(current, sigma_s, sigma_r);
            std::cout << "  Bilateral Filter done in " << ms(t0) << " ms  (GPU)\n";
            applyAndShow(current, r, "Bilateral Filter");
        }

        // ── 11: Canny Edge Detection NEW
        else if (choice == 11)
        {
            if (!requireImage(current)) continue;
            float lo = 50.f, hi = 150.f;
            std::cout << "  Low  threshold (e.g. 50.0 – weak edges below this are discarded): ";
            std::cin >> lo;
            std::cout << "  High threshold (e.g. 150.0 – strong edges above this are kept)  : ";
            std::cin >> hi;
            if (lo >= hi)
            {
                std::cout << "  [!] Low threshold must be less than high threshold.\n";
                continue;
            }
            auto t0 = Clock::now();
            cv::Mat r = doCanny(current, lo, hi);
            std::cout << "  Canny Edge Detection done in " << ms(t0) << " ms  (GPU)\n";
            applyAndShow(current, r, "Canny Edges");
        }

        // ── 12: Morphological Erosion NEW 
        else if (choice == 12)
        {
            if (!requireImage(current)) continue;
            int radius = 1;
            std::cout << "  Structuring element radius (1 = 3×3, 2 = 5×5, 3 = 7×7): ";
            std::cin >> radius;
            if (radius < 1 || radius > 20)
            {
                std::cout << "  [!] Radius must be 1–20.\n"; continue;
            }
            auto t0 = Clock::now();
            cv::Mat r = doErosion(current, radius);
            std::cout << "  Erosion (radius=" << radius << ") done in "
                << ms(t0) << " ms  (GPU)\n";
            applyAndShow(current, r, "Morphological Erosion");
        }

        // ── 13: Morphological Dilation NEW 
        else if (choice == 13)
        {
            if (!requireImage(current)) continue;
            int radius = 1;
            std::cout << "  Structuring element radius (1 = 3×3, 2 = 5×5, 3 = 7×7): ";
            std::cin >> radius;
            if (radius < 1 || radius > 20)
            {
                std::cout << "  [!] Radius must be 1–20.\n"; continue;
            }
            auto t0 = Clock::now();
            cv::Mat r = doDilation(current, radius);
            std::cout << "  Dilation (radius=" << radius << ") done in "
                << ms(t0) << " ms  (GPU)\n";
            applyAndShow(current, r, "Morphological Dilation");
        }

        // ── 14: Unsharp Masking NEW
        else if (choice == 14)
        {
            if (!requireImage(current)) continue;
            float strength = 1.0f;
            std::cout << "  Sharpening strength (0.5 = subtle, 1.0 = moderate, 2.0+ = strong): ";
            std::cin >> strength;
            if (strength <= 0.f)
            {
                std::cout << "  [!] Strength must be positive.\n"; continue;
            }
            auto t0 = Clock::now();
            cv::Mat r = doUnsharpMask(current, strength);
            std::cout << "  Unsharp Masking (strength=" << strength << ") done in "
                << ms(t0) << " ms  (GPU)\n";
            applyAndShow(current, r, "Unsharp Mask – Sharpened");
        }

        else
        {
            std::cout << "  [!] Invalid choice. Please enter a number shown in the menu.\n";
        }
    }

    return 0;
}