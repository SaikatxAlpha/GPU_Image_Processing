// main.cpp  –  GPU Image Processing Software  v3.0
// Modern GUI: Dear ImGui + Win32 + DirectX 11
//
// ═══════════════════════════════════════════════════════════════════
//  QUICK-START  –  Adding Dear ImGui to the project
// ═══════════════════════════════════════════════════════════════════
//  1. Download https://github.com/ocornut/imgui  (Code → Download ZIP)
//  2. Unzip; copy these files next to this main.cpp:
//        imgui.h            imgui.cpp
//        imgui_draw.cpp     imgui_tables.cpp   imgui_widgets.cpp
//        imgui_internal.h   imconfig.h
//        imstb_rectpack.h   imstb_textedit.h   imstb_truetype.h
//  3. From the "backends" sub-folder also copy:
//        imgui_impl_win32.h   imgui_impl_win32.cpp
//        imgui_impl_dx11.h    imgui_impl_dx11.cpp
//  4. Visual Studio: Project → Add Existing Item → select all 7 .cpp files
//  5. Project Properties → Linker → Input → Additional Dependencies:
//        d3d11.lib   d3dcompiler.lib   dxgi.lib
//  6. Project Properties → Linker → System → SubSystem: Windows
// ═══════════════════════════════════════════════════════════════════

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#include <tchar.h>
#include <windows.h>
#include <commdlg.h>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "kernel.h"

#include <string>
#include <vector>
#include <deque>
#include <functional>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>

// Forward-declare ImGui Win32 message handler
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// ============================================================
//  D3D11 globals
// ============================================================
static ID3D11Device* g_pd3dDevice = nullptr;
static ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
static IDXGISwapChain* g_pSwapChain = nullptr;
static UINT                     g_ResizeWidth = 0;
static UINT                     g_ResizeHeight = 0;
static ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

bool   CreateDeviceD3D(HWND hWnd);
void   CleanupDeviceD3D();
void   CreateRenderTarget();
void   CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// ============================================================
//  Image Texture  (cv::Mat  →  D3D11 shader-resource view)
// ============================================================
static ID3D11ShaderResourceView* g_imageSRV = nullptr;
static int g_texW = 0, g_texH = 0;

static void UpdateImageTexture(const cv::Mat& mat)
{
    if (g_imageSRV) { g_imageSRV->Release(); g_imageSRV = nullptr; }
    g_texW = g_texH = 0;
    if (mat.empty() || !g_pd3dDevice) return;

    // D3D11 wants BGRA
    cv::Mat rgba;
    if (mat.channels() == 3) cv::cvtColor(mat, rgba, cv::COLOR_BGR2BGRA);
    else if (mat.channels() == 1) cv::cvtColor(mat, rgba, cv::COLOR_GRAY2BGRA);
    else                          rgba = mat.clone();

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = rgba.cols;
    td.Height = rgba.rows;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA id = {};
    id.pSysMem = rgba.data;
    id.SysMemPitch = rgba.cols * 4;

    ID3D11Texture2D* pTex = nullptr;
    if (FAILED(g_pd3dDevice->CreateTexture2D(&td, &id, &pTex))) return;

    D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
    sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sd.Texture2D.MipLevels = 1;

    g_pd3dDevice->CreateShaderResourceView(pTex, &sd, &g_imageSRV);
    pTex->Release();

    g_texW = rgba.cols;
    g_texH = rgba.rows;
}

// ============================================================
//  Application State
// ============================================================
using Clock = std::chrono::high_resolution_clock;

enum FilterID {
    F_NONE = 0,
    F_GRAYSCALE, F_BLUR_NAIVE, F_BLUR_SMEM,
    F_SOBEL, F_BRIGHTNESS, F_HISTEQ,
    F_BILATERAL, F_CANNY, F_EROSION, F_DILATION, F_UNSHARP
};

struct GpuInfo { std::string name; int memMB = 0, smCount = 0, major = 0, minor = 0; };

struct AppState
{
    cv::Mat       current;
    cv::Mat       original;       // backup for "Revert to Original"
    std::string   filePath;
    bool          imageLoaded = false;
    bool          processing = false;
    double        lastOpMs = 0.0;
    std::string   lastOpName;

    std::vector<GpuInfo> gpus;

    // Pending filter (for modal-based ops triggered from menu)
    int  pendingFilter = F_NONE;
    bool openModal = false;

    // Per-filter parameters  (persist across calls)
    float  bcAlpha = 1.0f;
    int    bcBeta = 0;
    float  bilateralSigmaS = 10.f;
    float  bilateralSigmaR = 75.f;
    float  cannyLo = 50.f;
    float  cannyHi = 150.f;
    int    morphRadius = 1;
    float  unsharpStrength = 1.0f;

    // Processing history / log
    std::deque<std::string> log;
    static constexpr int LOG_MAX = 200;

    void addLog(const std::string& s)
    {
        log.push_back(s);
        if ((int)log.size() > LOG_MAX) log.pop_front();
    }
};
static AppState g_app;

// ============================================================
//  CUDA wrappers
// ============================================================
static cv::Mat RunKernel(const char* name, std::function<cv::Mat()> fn)
{
    auto t0 = Clock::now();
    cv::Mat r = fn();
    double dt = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    g_app.lastOpMs = dt;
    g_app.lastOpName = name;

    std::ostringstream ss;
    ss << name << "  →  " << std::fixed << std::setprecision(1) << dt << " ms  (GPU)";
    g_app.addLog(ss.str());
    return r;
}

static cv::Mat KGrayscale(const cv::Mat& s)
{
    int W = s.cols, H = s.rows;
    unsigned char* di, * dout;
    cudaMalloc(&di, (size_t)W * H * 3);
    cudaMalloc(&dout, (size_t)W * H);
    cudaMemcpy(di, s.data, (size_t)W * H * 3, cudaMemcpyHostToDevice);
    launchGrayscaleKernel(di, dout, W, H);
    cv::Mat d(H, W, CV_8UC1);
    cudaMemcpy(d.data, dout, (size_t)W * H, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return d;
}

static cv::Mat KGaussian(const cv::Mat& s, bool smem)
{
    int W = s.cols, H = s.rows, C = s.channels();
    size_t sz = (size_t)W * H * C;
    unsigned char* di = nullptr, * dout = nullptr;
    if (cudaMalloc(&di, sz) != cudaSuccess ||
        cudaMalloc(&dout, sz) != cudaSuccess)
    {
        cudaFree(di); return s.clone();
    }
    cudaMemcpy(di, s.data, sz, cudaMemcpyHostToDevice);
    if (smem) launchGaussianBlurSharedKernel(di, dout, W, H, C);
    else      launchGaussianBlurKernel(di, dout, W, H, C);
    cv::Mat d(H, W, s.type());
    cudaMemcpy(d.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return d;
}

static cv::Mat KSobel(const cv::Mat& s)
{
    cv::Mat g = (s.channels() == 3) ? KGrayscale(s) : s.clone();
    int W = g.cols, H = g.rows; size_t sz = (size_t)W * H;
    unsigned char* di, * dout;
    cudaMalloc(&di, sz); cudaMalloc(&dout, sz);
    cudaMemcpy(di, g.data, sz, cudaMemcpyHostToDevice);
    launchSobelKernel(di, dout, W, H);
    cv::Mat d(H, W, CV_8UC1);
    cudaMemcpy(d.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return d;
}

static cv::Mat KBrightnessContrast(const cv::Mat& s, float alpha, int beta)
{
    int W = s.cols, H = s.rows, C = s.channels(); size_t sz = (size_t)W * H * C;
    unsigned char* di, * dout;
    cudaMalloc(&di, sz); cudaMalloc(&dout, sz);
    cudaMemcpy(di, s.data, sz, cudaMemcpyHostToDevice);
    launchBrightnessContrastKernel(di, dout, W, H, C, alpha, beta);
    cv::Mat d(H, W, s.type());
    cudaMemcpy(d.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return d;
}

static cv::Mat KHistEq(const cv::Mat& s)
{
    cv::Mat g = (s.channels() == 3) ? KGrayscale(s) : s.clone();
    int W = g.cols, H = g.rows; size_t sz = (size_t)W * H;
    unsigned char* di, * dout;
    cudaMalloc(&di, sz); cudaMalloc(&dout, sz);
    cudaMemcpy(di, g.data, sz, cudaMemcpyHostToDevice);
    launchHistogramEqualizationKernel(di, dout, W, H);
    cv::Mat d(H, W, CV_8UC1);
    cudaMemcpy(d.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return d;
}

static cv::Mat KBilateral(const cv::Mat& s, float ss, float sr)
{
    int W = s.cols, H = s.rows, C = s.channels(); size_t sz = (size_t)W * H * C;
    unsigned char* di = nullptr, * dout = nullptr;
    if (cudaMalloc(&di, sz) != cudaSuccess || cudaMalloc(&dout, sz) != cudaSuccess)
    {
        cudaFree(di); return s.clone();
    }
    cudaMemcpy(di, s.data, sz, cudaMemcpyHostToDevice);
    launchBilateralFilterKernel(di, dout, W, H, C, ss, sr);
    cv::Mat d(H, W, s.type());
    cudaMemcpy(d.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return d;
}

static cv::Mat KCanny(const cv::Mat& s, float lo, float hi)
{
    cv::Mat g = (s.channels() == 3) ? KGrayscale(s) : s.clone();
    int W = g.cols, H = g.rows; size_t sz = (size_t)W * H;
    unsigned char* di = nullptr, * dout = nullptr;
    if (cudaMalloc(&di, sz) != cudaSuccess || cudaMalloc(&dout, sz) != cudaSuccess)
    {
        cudaFree(di); return s.clone();
    }
    cudaMemcpy(di, g.data, sz, cudaMemcpyHostToDevice);
    launchCannyKernel(di, dout, W, H, lo, hi);
    cv::Mat d(H, W, CV_8UC1);
    cudaMemcpy(d.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return d;
}

static cv::Mat KMorph(const cv::Mat& s, int r, bool dilate)
{
    int W = s.cols, H = s.rows, C = s.channels(); size_t sz = (size_t)W * H * C;
    unsigned char* di = nullptr, * dout = nullptr;
    if (cudaMalloc(&di, sz) != cudaSuccess || cudaMalloc(&dout, sz) != cudaSuccess)
    {
        cudaFree(di); return s.clone();
    }
    cudaMemcpy(di, s.data, sz, cudaMemcpyHostToDevice);
    if (dilate) launchDilationKernel(di, dout, W, H, C, r);
    else        launchErosionKernel(di, dout, W, H, C, r);
    cv::Mat d(H, W, s.type());
    cudaMemcpy(d.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return d;
}

static cv::Mat KUnsharp(const cv::Mat& s, float strength)
{
    int W = s.cols, H = s.rows, C = s.channels(); size_t sz = (size_t)W * H * C;
    unsigned char* di = nullptr, * dout = nullptr;
    if (cudaMalloc(&di, sz) != cudaSuccess || cudaMalloc(&dout, sz) != cudaSuccess)
    {
        cudaFree(di); return s.clone();
    }
    cudaMemcpy(di, s.data, sz, cudaMemcpyHostToDevice);
    launchUnsharpMaskKernel(di, dout, W, H, C, strength);
    cv::Mat d(H, W, s.type());
    cudaMemcpy(d.data, dout, sz, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dout);
    return d;
}

// Apply result to working image + refresh texture
static void Apply(cv::Mat result)
{
    g_app.current = std::move(result);
    UpdateImageTexture(g_app.current);
}

// ============================================================
//  Win32 file dialogs
// ============================================================
static std::string FileOpenDlg()
{
    char buf[MAX_PATH] = {};
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "Images\0*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff\0All Files\0*.*\0\0";
    ofn.lpstrFile = buf;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    ofn.lpstrTitle = "Open Image";
    return GetOpenFileNameA(&ofn) ? buf : "";
}

static std::string FileSaveDlg()
{
    char buf[MAX_PATH] = "output.png";
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "PNG\0*.png\0JPEG\0*.jpg\0BMP\0*.bmp\0All Files\0*.*\0\0";
    ofn.lpstrFile = buf;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_OVERWRITEPROMPT;
    ofn.lpstrTitle = "Save Image As";
    ofn.lpstrDefExt = "png";
    return GetSaveFileNameA(&ofn) ? buf : "";
}

// ============================================================
//  Helpers
// ============================================================
static void LoadImage(const std::string& path)
{
    cv::Mat m = cv::imread(path);
    if (m.empty()) { g_app.addLog("ERROR: failed to load  " + path); return; }
    g_app.current = m;
    g_app.original = m.clone();
    g_app.filePath = path;
    g_app.imageLoaded = true;
    UpdateImageTexture(m);
    std::string fn = path.substr(path.find_last_of("/\\") + 1);
    std::ostringstream ss;
    ss << "Loaded  " << fn << "  (" << m.cols << " × " << m.rows
        << ",  " << m.channels() << " ch)";
    g_app.addLog(ss.str());
}

static void LoadGPUInfo()
{
    int n = 0; cudaGetDeviceCount(&n);
    for (int i = 0; i < n; i++) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, i);
        GpuInfo gi;
        gi.name = p.name;
        gi.memMB = (int)(p.totalGlobalMem / (1024 * 1024));
        gi.smCount = p.multiProcessorCount;
        gi.major = p.major;
        gi.minor = p.minor;
        g_app.gpus.push_back(gi);
    }
}

// ============================================================
//  ImGui Theme
// ============================================================
static void SetupTheme()
{
    ImGuiStyle& s = ImGui::GetStyle();
    ImVec4* c = s.Colors;

    c[ImGuiCol_WindowBg] = { 0.09f, 0.09f, 0.12f, 1.00f };
    c[ImGuiCol_ChildBg] = { 0.11f, 0.11f, 0.15f, 1.00f };
    c[ImGuiCol_PopupBg] = { 0.09f, 0.09f, 0.12f, 0.97f };
    c[ImGuiCol_Border] = { 0.24f, 0.24f, 0.32f, 1.00f };
    c[ImGuiCol_FrameBg] = { 0.18f, 0.18f, 0.24f, 1.00f };
    c[ImGuiCol_FrameBgHovered] = { 0.24f, 0.24f, 0.32f, 1.00f };
    c[ImGuiCol_FrameBgActive] = { 0.16f, 0.40f, 0.70f, 1.00f };
    c[ImGuiCol_TitleBg] = { 0.07f, 0.07f, 0.09f, 1.00f };
    c[ImGuiCol_TitleBgActive] = { 0.09f, 0.24f, 0.46f, 1.00f };
    c[ImGuiCol_MenuBarBg] = { 0.07f, 0.07f, 0.10f, 1.00f };
    c[ImGuiCol_ScrollbarGrab] = { 0.24f, 0.24f, 0.32f, 1.00f };
    c[ImGuiCol_CheckMark] = { 0.20f, 0.65f, 1.00f, 1.00f };
    c[ImGuiCol_SliderGrab] = { 0.18f, 0.54f, 0.95f, 1.00f };
    c[ImGuiCol_SliderGrabActive] = { 0.24f, 0.68f, 1.00f, 1.00f };
    c[ImGuiCol_Button] = { 0.16f, 0.32f, 0.56f, 1.00f };
    c[ImGuiCol_ButtonHovered] = { 0.22f, 0.44f, 0.74f, 1.00f };
    c[ImGuiCol_ButtonActive] = { 0.11f, 0.26f, 0.50f, 1.00f };
    c[ImGuiCol_Header] = { 0.18f, 0.36f, 0.60f, 0.55f };
    c[ImGuiCol_HeaderHovered] = { 0.22f, 0.44f, 0.72f, 0.80f };
    c[ImGuiCol_HeaderActive] = { 0.16f, 0.40f, 0.70f, 1.00f };
    c[ImGuiCol_Tab] = { 0.11f, 0.22f, 0.40f, 0.86f };
    c[ImGuiCol_TabHovered] = { 0.22f, 0.44f, 0.74f, 0.80f };
    c[ImGuiCol_TabActive] = { 0.16f, 0.34f, 0.62f, 1.00f };
    c[ImGuiCol_Separator] = { 0.22f, 0.22f, 0.30f, 1.00f };
    c[ImGuiCol_Text] = { 0.92f, 0.92f, 0.95f, 1.00f };
    c[ImGuiCol_TextDisabled] = { 0.42f, 0.42f, 0.52f, 1.00f };

    s.WindowRounding = 7.f;  s.ChildRounding = 5.f;
    s.FrameRounding = 4.f;  s.PopupRounding = 5.f;
    s.GrabRounding = 4.f;  s.TabRounding = 4.f;
    s.WindowPadding = { 10.f, 10.f };
    s.FramePadding = { 7.f,  4.f };
    s.ItemSpacing = { 8.f,  6.f };
    s.ScrollbarSize = 12.f;
}

// ============================================================
//  UI helpers
// ============================================================
static bool SideBtn(const char* label, bool enabled = true)
{
    if (!enabled) {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.38f);
        ImGui::PushStyleColor(ImGuiCol_Button, { 0.16f,0.16f,0.22f,1.f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, { 0.16f,0.16f,0.22f,1.f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, { 0.16f,0.16f,0.22f,1.f });
    }
    bool hit = ImGui::Button(label, { -1.f, 0.f });
    if (!enabled) { ImGui::PopStyleColor(3); ImGui::PopStyleVar(); return false; }
    return hit;
}

static void SectionHeader(const char* label, ImVec4 col = { 0.78f,0.78f,0.38f,1.f })
{
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    ImGui::Separator();
    ImGui::Spacing();
}

// ============================================================
//  Modal parameter popups
// ============================================================
static void DrawBrightnessModal()
{
    if (!ImGui::BeginPopupModal("Brightness & Contrast", nullptr,
        ImGuiWindowFlags_AlwaysAutoResize)) return;

    ImGui::TextDisabled("Adjust exposure and tone of the image.");
    ImGui::Spacing();
    ImGui::SetNextItemWidth(280.f);
    ImGui::SliderFloat("Contrast  (alpha)", &g_app.bcAlpha, 0.10f, 3.0f, "%.2f");
    ImGui::TextDisabled("  1.0 = unchanged,  >1.0 = more contrast");
    ImGui::Spacing();
    ImGui::SetNextItemWidth(280.f);
    ImGui::SliderInt("Brightness (beta)", &g_app.bcBeta, -128, 128);
    ImGui::TextDisabled("  0 = unchanged,  positive = brighter");
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Button, { 0.14f,0.48f,0.14f,1.f });
    if (ImGui::Button("Apply", { 110,0 })) {
        float a = g_app.bcAlpha; int b = g_app.bcBeta;
        Apply(RunKernel("Brightness / Contrast",
            [a, b] { return KBrightnessContrast(g_app.current, a, b); }));
        g_app.pendingFilter = F_NONE;
        ImGui::CloseCurrentPopup();
    }
    ImGui::PopStyleColor();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", { 110,0 })) {
        g_app.pendingFilter = F_NONE; ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

static void DrawBilateralModal()
{
    if (!ImGui::BeginPopupModal("Bilateral Filter", nullptr,
        ImGuiWindowFlags_AlwaysAutoResize)) return;

    ImGui::TextDisabled("Edge-preserving smoothing — blurs without crossing edges.");
    ImGui::Spacing();
    ImGui::SetNextItemWidth(280.f);
    ImGui::SliderFloat("Spatial sigma (ss)", &g_app.bilateralSigmaS, 1.f, 30.f, "%.1f");
    ImGui::TextDisabled("  Larger = wider blur radius");
    ImGui::Spacing();
    ImGui::SetNextItemWidth(280.f);
    ImGui::SliderFloat("Range  sigma (sr)", &g_app.bilateralSigmaR, 10.f, 200.f, "%.1f");
    ImGui::TextDisabled("  Larger = less edge preservation");
    ImGui::PushStyleColor(ImGuiCol_Text, { 1.f,0.75f,0.3f,1.f });
    ImGui::TextWrapped("Note: large spatial sigma can be slow on high-res images.");
    ImGui::PopStyleColor();
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Button, { 0.14f,0.48f,0.14f,1.f });
    if (ImGui::Button("Apply", { 110,0 })) {
        float ss = g_app.bilateralSigmaS, sr = g_app.bilateralSigmaR;
        Apply(RunKernel("Bilateral Filter",
            [ss, sr] { return KBilateral(g_app.current, ss, sr); }));
        g_app.pendingFilter = F_NONE; ImGui::CloseCurrentPopup();
    }
    ImGui::PopStyleColor();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", { 110,0 })) {
        g_app.pendingFilter = F_NONE; ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

static void DrawCannyModal()
{
    if (!ImGui::BeginPopupModal("Canny Edge Detection", nullptr,
        ImGuiWindowFlags_AlwaysAutoResize)) return;

    ImGui::TextDisabled("Full 5-pass GPU Canny pipeline.");
    ImGui::Spacing();
    ImGui::SetNextItemWidth(280.f);
    ImGui::SliderFloat("Low threshold", &g_app.cannyLo, 1.f, 300.f, "%.0f");
    ImGui::TextDisabled("  Weak edges below this value are discarded");
    ImGui::Spacing();
    ImGui::SetNextItemWidth(280.f);
    ImGui::SliderFloat("High threshold", &g_app.cannyHi, 10.f, 600.f, "%.0f");
    ImGui::TextDisabled("  Strong edges above this value are kept");
    if (g_app.cannyLo >= g_app.cannyHi)
        ImGui::TextColored({ 1.f,0.4f,0.4f,1.f }, "  Low must be less than High!");
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    bool canApply = g_app.cannyLo < g_app.cannyHi;
    if (!canApply) ImGui::BeginDisabled();
    ImGui::PushStyleColor(ImGuiCol_Button, { 0.14f,0.48f,0.14f,1.f });
    if (ImGui::Button("Apply", { 110,0 })) {
        float lo = g_app.cannyLo, hi = g_app.cannyHi;
        Apply(RunKernel("Canny Edge Detection",
            [lo, hi] { return KCanny(g_app.current, lo, hi); }));
        g_app.pendingFilter = F_NONE; ImGui::CloseCurrentPopup();
    }
    ImGui::PopStyleColor();
    if (!canApply) ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", { 110,0 })) {
        g_app.pendingFilter = F_NONE; ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

static void DrawMorphModal(bool dilate)
{
    const char* title = dilate ? "Morphological Dilation" : "Morphological Erosion";
    if (!ImGui::BeginPopupModal(title, nullptr,
        ImGuiWindowFlags_AlwaysAutoResize)) return;

    ImGui::TextDisabled(dilate
        ? "Expands bright regions (maximum filter)."
        : "Shrinks bright regions (minimum filter).");
    ImGui::Spacing();
    ImGui::SetNextItemWidth(280.f);
    ImGui::SliderInt("Radius", &g_app.morphRadius, 1, 20);
    int side = g_app.morphRadius * 2 + 1;
    ImGui::TextDisabled("  Structuring element: %d × %d", side, side);
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Button, { 0.14f,0.48f,0.14f,1.f });
    if (ImGui::Button("Apply", { 110,0 })) {
        int r = g_app.morphRadius; bool d = dilate;
        Apply(RunKernel(dilate ? "Morphological Dilation" : "Morphological Erosion",
            [r, d] { return KMorph(g_app.current, r, d); }));
        g_app.pendingFilter = F_NONE; ImGui::CloseCurrentPopup();
    }
    ImGui::PopStyleColor();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", { 110,0 })) {
        g_app.pendingFilter = F_NONE; ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

static void DrawUnsharpModal()
{
    if (!ImGui::BeginPopupModal("Unsharp Masking", nullptr,
        ImGuiWindowFlags_AlwaysAutoResize)) return;

    ImGui::TextDisabled("Sharpen by subtracting a blurred copy.");
    ImGui::Spacing();
    ImGui::SetNextItemWidth(280.f);
    ImGui::SliderFloat("Strength", &g_app.unsharpStrength, 0.10f, 4.0f, "%.2f");
    ImGui::TextDisabled("  0.5–1.0 = subtle,  1.5–2.5 = strong");
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Button, { 0.14f,0.48f,0.14f,1.f });
    if (ImGui::Button("Apply", { 110,0 })) {
        float st = g_app.unsharpStrength;
        Apply(RunKernel("Unsharp Masking",
            [st] { return KUnsharp(g_app.current, st); }));
        g_app.pendingFilter = F_NONE; ImGui::CloseCurrentPopup();
    }
    ImGui::PopStyleColor();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", { 110,0 })) {
        g_app.pendingFilter = F_NONE; ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

// ============================================================
//  Main UI render (called every frame)
// ============================================================
static void RenderUI()
{
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGuiIO& io = ImGui::GetIO();
    bool& hasImg = g_app.imageLoaded;

    // ── Full-screen host window ───────────────────────────────
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0.f, 0.f });
    ImGui::Begin("##Host", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_MenuBar |
        ImGuiWindowFlags_NoBringToDisplayFront);
    ImGui::PopStyleVar(3);

    // ── Menu bar ──────────────────────────────────────────────
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open Image…", "Ctrl+O")) {
                auto p = FileOpenDlg();
                if (!p.empty()) LoadImage(p);
            }
            if (ImGui::MenuItem("Save Image…", "Ctrl+S", false, hasImg)) {
                auto p = FileSaveDlg();
                if (!p.empty()) {
                    if (cv::imwrite(p, g_app.current))
                        g_app.addLog("Saved  →  " + p);
                    else
                        g_app.addLog("ERROR: could not save  " + p);
                }
            }
            if (ImGui::MenuItem("Revert to Original", nullptr, false,
                hasImg && !g_app.original.empty())) {
                g_app.current = g_app.original.clone();
                UpdateImageTexture(g_app.current);
                g_app.addLog("Reverted to original image.");
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Alt+F4")) PostQuitMessage(0);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Filters"))
        {
            if (ImGui::MenuItem("Grayscale", nullptr, false, hasImg && g_app.current.channels() == 3))
            {
                Apply(RunKernel("Grayscale", [] { return KGrayscale(g_app.current); }));
            }
            if (ImGui::MenuItem("Gaussian Blur (Naive)", nullptr, false, hasImg))
            {
                Apply(RunKernel("Gaussian (Naive)", [] { return KGaussian(g_app.current, false); }));
            }
            if (ImGui::MenuItem("Gaussian Blur (SMEM) ★", nullptr, false, hasImg))
            {
                Apply(RunKernel("Gaussian (SMEM)", [] { return KGaussian(g_app.current, true);  }));
            }
            if (ImGui::MenuItem("Sobel Edges", nullptr, false, hasImg))
            {
                Apply(RunKernel("Sobel Edge Detection", [] { return KSobel(g_app.current); }));
            }
            if (ImGui::MenuItem("Histogram Equalization", nullptr, false, hasImg))
            {
                Apply(RunKernel("Histogram Equalization", [] { return KHistEq(g_app.current); }));
            }
            if (ImGui::MenuItem("Brightness & Contrast…", nullptr, false, hasImg))
            {
                g_app.pendingFilter = F_BRIGHTNESS; g_app.openModal = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Bilateral Filter…", nullptr, false, hasImg))
            {
                g_app.pendingFilter = F_BILATERAL;  g_app.openModal = true;
            }
            if (ImGui::MenuItem("Canny Edge Detection…", nullptr, false, hasImg))
            {
                g_app.pendingFilter = F_CANNY;      g_app.openModal = true;
            }
            if (ImGui::MenuItem("Morphological Erosion…", nullptr, false, hasImg))
            {
                g_app.pendingFilter = F_EROSION;    g_app.openModal = true;
            }
            if (ImGui::MenuItem("Morphological Dilation…", nullptr, false, hasImg))
            {
                g_app.pendingFilter = F_DILATION;   g_app.openModal = true;
            }
            if (ImGui::MenuItem("Unsharp Masking…", nullptr, false, hasImg))
            {
                g_app.pendingFilter = F_UNSHARP;    g_app.openModal = true;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help"))
        {
            ImGui::MenuItem("GPU Image Processing  v3.0", nullptr, false, false);
            ImGui::MenuItem("Built with CUDA + OpenCV + Dear ImGui", nullptr, false, false);
            ImGui::EndMenu();
        }

        // Right-aligned timing indicator
        if (g_app.lastOpMs > 0.0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "Last op:  %.1f ms", g_app.lastOpMs);
            float tw = ImGui::CalcTextSize(buf).x + 16.f;
            ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - tw);
            ImGui::PushStyleColor(ImGuiCol_Text, { 0.40f,0.90f,0.45f,1.f });
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();
        }

        ImGui::EndMenuBar();
    }

    // Keyboard shortcuts
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O, false)) {
        auto p = FileOpenDlg(); if (!p.empty()) LoadImage(p);
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S, false) && hasImg) {
        auto p = FileSaveDlg();
        if (!p.empty()) {
            if (cv::imwrite(p, g_app.current)) g_app.addLog("Saved  →  " + p);
            else                               g_app.addLog("ERROR: save failed");
        }
    }

    // ── Layout dimensions ─────────────────────────────────────
    float winW = vp->WorkSize.x;
    float winH = vp->WorkSize.y;
    float menuH = ImGui::GetFrameHeight() + ImGui::GetStyle().WindowPadding.y * 2.f;
    float statusH = 28.f;
    float contentH = winH - menuH - statusH;
    float sideW = 228.f;
    float rightW = winW - sideW - 1.f;
    float imgPanH = contentH * 0.70f;
    float logPanH = contentH - imgPanH - ImGui::GetStyle().ItemSpacing.y;
    float contentTop = vp->WorkPos.y + menuH;

    // ── SIDEBAR ───────────────────────────────────────────────
    ImGui::SetNextWindowPos({ vp->WorkPos.x, contentTop });
    ImGui::SetNextWindowSize({ sideW, contentH });
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::Begin("##Sidebar", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    ImGui::PopStyleVar(2);

    // Image info badge
    if (hasImg) {
        ImGui::PushStyleColor(ImGuiCol_Text, { 0.55f,0.92f,0.55f,1.f });
        ImGui::Text("%d × %d  px", g_app.current.cols, g_app.current.rows);
        ImGui::PopStyleColor();
        ImGui::TextDisabled("%d channel%s", g_app.current.channels(),
            g_app.current.channels() > 1 ? "s" : "");
    }
    else {
        ImGui::TextDisabled("No image loaded");
    }
    ImGui::Spacing();

    // Open / Save / Revert
    ImGui::PushStyleColor(ImGuiCol_Button, { 0.16f,0.38f,0.62f,1.f });
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, { 0.22f,0.50f,0.78f,1.f });
    if (ImGui::Button("  Open Image", { -1.f,0.f })) {
        auto p = FileOpenDlg(); if (!p.empty()) LoadImage(p);
    }
    ImGui::PopStyleColor(2);

    ImGui::PushStyleColor(ImGuiCol_Button, { 0.14f,0.44f,0.14f,1.f });
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, { 0.20f,0.58f,0.20f,1.f });
    if (ImGui::Button("  Save Image", { -1.f,0.f }) && hasImg) {
        auto p = FileSaveDlg();
        if (!p.empty()) {
            if (cv::imwrite(p, g_app.current)) g_app.addLog("Saved  →  " + p);
        }
    }
    ImGui::PopStyleColor(2);

    if (SideBtn("  Revert to Original", hasImg && !g_app.original.empty())) {
        g_app.current = g_app.original.clone();
        UpdateImageTexture(g_app.current);
        g_app.addLog("Reverted to original image.");
    }

    // ── Basic filters ─────────────────────────────────────────
    SectionHeader("  Basic Filters");

    bool is3ch = hasImg && g_app.current.channels() == 3;
    if (SideBtn("Grayscale", is3ch)) {
        Apply(RunKernel("Grayscale", [] { return KGrayscale(g_app.current); }));
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && !is3ch && hasImg)
        ImGui::SetTooltip("Image is already grayscale");

    if (SideBtn("Gaussian Blur — Naive", hasImg))
        Apply(RunKernel("Gaussian (Naive)", [] { return KGaussian(g_app.current, false); }));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("5×5 kernel, global-memory reads");

    // Highlight the faster version
    ImGui::PushStyleColor(ImGuiCol_Button, { 0.12f,0.34f,0.58f,1.f });
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, { 0.18f,0.46f,0.74f,1.f });
    if (SideBtn("Gaussian Blur — SMEM  ★", hasImg))
        Apply(RunKernel("Gaussian (SMEM)", [] { return KGaussian(g_app.current, true); }));
    ImGui::PopStyleColor(2);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Same result, faster: shared-memory tiling");

    if (SideBtn("Sobel Edge Detection", hasImg))
        Apply(RunKernel("Sobel Edge Detection", [] { return KSobel(g_app.current); }));
    if (SideBtn("Brightness / Contrast…", hasImg))
    {
        g_app.pendingFilter = F_BRIGHTNESS; g_app.openModal = true;
    }
    if (SideBtn("Histogram Equalization", hasImg))
        Apply(RunKernel("Histogram Equalization", [] { return KHistEq(g_app.current); }));

    // ── Advanced filters ──────────────────────────────────────
    SectionHeader("  Advanced Filters", { 0.92f,0.62f,0.28f,1.f });

    if (SideBtn("Bilateral Filter…", hasImg))
    {
        g_app.pendingFilter = F_BILATERAL; g_app.openModal = true;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Blur that preserves edges");

    if (SideBtn("Canny Edge Detection…", hasImg))
    {
        g_app.pendingFilter = F_CANNY; g_app.openModal = true;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Full 5-pass GPU Canny pipeline");

    if (SideBtn("Morpho. Erosion…", hasImg))
    {
        g_app.pendingFilter = F_EROSION;  g_app.openModal = true;
    }
    if (SideBtn("Morpho. Dilation…", hasImg))
    {
        g_app.pendingFilter = F_DILATION; g_app.openModal = true;
    }
    if (SideBtn("Unsharp Masking…", hasImg))
    {
        g_app.pendingFilter = F_UNSHARP;  g_app.openModal = true;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Sharpen via blur subtraction");

    // ── GPU Info ──────────────────────────────────────────────
    SectionHeader("  GPU", { 0.55f,0.82f,1.00f,1.f });
    if (g_app.gpus.empty()) {
        ImGui::TextColored({ 1.f,0.5f,0.5f,1.f }, "No CUDA GPU detected");
    }
    else {
        for (auto& g : g_app.gpus) {
            ImGui::PushStyleColor(ImGuiCol_Text, { 0.88f,0.88f,0.88f,1.f });
            ImGui::TextWrapped("%s", g.name.c_str());
            ImGui::PopStyleColor();
            ImGui::TextDisabled("SM %d.%d  ·  %d SMs  ·  %d MB",
                g.major, g.minor, g.smCount, g.memMB);
        }
    }

    ImGui::End();   // Sidebar

    // ── RIGHT COLUMN: Image panel + Log panel ─────────────────
    ImGui::SetNextWindowPos({ vp->WorkPos.x + sideW, contentTop });
    ImGui::SetNextWindowSize({ rightW, contentH });
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 4.f,4.f });
    ImGui::Begin("##Right", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoScrollbar);
    ImGui::PopStyleVar(3);

    // ── Image panel ───────────────────────────────────────────
    ImGui::BeginChild("##ImagePanel", { rightW - 8.f, imgPanH }, true);
    if (!hasImg) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        const char* hint = "Open an image to get started   (File → Open Image,  or  Ctrl+O)";
        float tw = ImGui::CalcTextSize(hint).x;
        ImGui::SetCursorPos({ (avail.x - tw) * 0.5f, avail.y * 0.45f });
        ImGui::PushStyleColor(ImGuiCol_Text, { 0.30f,0.30f,0.42f,1.f });
        ImGui::TextUnformatted(hint);
        ImGui::PopStyleColor();
    }
    else if (g_imageSRV) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float scale = std::min(avail.x / (float)g_texW,
            avail.y / (float)g_texH);
        scale = std::min(scale, 2.0f);   // allow up to 2× zoom, but never stretch
        ImVec2 sz{ g_texW * scale, g_texH * scale };
        ImGui::SetCursorPos({ (avail.x - sz.x) * 0.5f, (avail.y - sz.y) * 0.5f });
        ImGui::Image((ImTextureID)g_imageSRV, sz);

        // Hover: show pixel value
        if (ImGui::IsItemHovered()) {
            ImVec2 mp = ImGui::GetMousePos();
            ImVec2 ip = ImGui::GetItemRectMin();
            int px = (int)((mp.x - ip.x) / scale);
            int py = (int)((mp.y - ip.y) / scale);
            px = std::max(0, std::min(px, g_texW - 1));
            py = std::max(0, std::min(py, g_texH - 1));
            ImGui::BeginTooltip();
            ImGui::Text("Pixel  (%d, %d)", px, py);
            if (g_app.current.channels() == 3) {
                auto v = g_app.current.at<cv::Vec3b>(py, px);
                ImGui::Text("B: %3d   G: %3d   R: %3d", v[0], v[1], v[2]);
            }
            else {
                ImGui::Text("L: %3d", g_app.current.at<uchar>(py, px));
            }
            ImGui::EndTooltip();
        }
    }
    ImGui::EndChild();   // ImagePanel

    // ── Log panel ─────────────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_ChildBg, { 0.08f,0.08f,0.11f,1.f });
    ImGui::BeginChild("##LogPanel", { rightW - 8.f, logPanH }, true,
        ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::PopStyleColor();

    ImGui::PushStyleColor(ImGuiCol_Text, { 0.78f,0.78f,0.38f,1.f });
    ImGui::TextUnformatted(" History");
    ImGui::PopStyleColor();
    ImGui::Separator();

    for (auto& entry : g_app.log) {
        ImGui::PushStyleColor(ImGuiCol_Text, { 0.72f,0.88f,0.72f,1.f });
        ImGui::Text("  %s", entry.c_str());
        ImGui::PopStyleColor();
    }
    // Auto-scroll
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();   // LogPanel

    ImGui::End();   // Right

    // ── Status bar ────────────────────────────────────────────
    float sbTop = vp->WorkPos.y + winH - statusH;
    ImGui::SetNextWindowPos({ vp->WorkPos.x, sbTop });
    ImGui::SetNextWindowSize({ winW, statusH });
    ImGui::PushStyleColor(ImGuiCol_WindowBg, { 0.06f,0.06f,0.08f,1.f });
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 10.f, 4.f });
    ImGui::Begin("##Status", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoScrollbar);
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();

    if (hasImg) {
        ImGui::PushStyleColor(ImGuiCol_Text, { 0.62f,0.88f,0.62f,1.f });
        ImGui::Text("%d × %d   ·   %d ch", g_app.current.cols,
            g_app.current.rows, g_app.current.channels());
        ImGui::PopStyleColor();
    }
    else {
        ImGui::TextDisabled("Ready — no image loaded");
    }

    if (!g_app.gpus.empty()) {
        char gpuBuf[128];
        snprintf(gpuBuf, sizeof(gpuBuf), "GPU: %s  (%d MB)",
            g_app.gpus[0].name.c_str(), g_app.gpus[0].memMB);
        float tw = ImGui::CalcTextSize(gpuBuf).x;
        ImGui::SameLine(winW - tw - 14.f);
        ImGui::PushStyleColor(ImGuiCol_Text, { 0.55f,0.80f,1.00f,1.f });
        ImGui::TextUnformatted(gpuBuf);
        ImGui::PopStyleColor();
    }
    ImGui::End();   // Status

    ImGui::End();   // Host

    // ── Open modal popups (must be called before BeginPopupModal) ──
    if (g_app.openModal && g_app.pendingFilter != F_NONE) {
        switch (g_app.pendingFilter) {
        case F_BRIGHTNESS: ImGui::OpenPopup("Brightness & Contrast");   break;
        case F_BILATERAL:  ImGui::OpenPopup("Bilateral Filter");         break;
        case F_CANNY:      ImGui::OpenPopup("Canny Edge Detection");     break;
        case F_EROSION:    ImGui::OpenPopup("Morphological Erosion");    break;
        case F_DILATION:   ImGui::OpenPopup("Morphological Dilation");   break;
        case F_UNSHARP:    ImGui::OpenPopup("Unsharp Masking");          break;
        }
        g_app.openModal = false;
    }

    // ── Draw modal contents ───────────────────────────────────
    DrawBrightnessModal();
    DrawBilateralModal();
    DrawCannyModal();
    DrawMorphModal(false);   // Erosion
    DrawMorphModal(true);    // Dilation
    DrawUnsharpModal();
}

// ============================================================
//  WinMain
// ============================================================
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int)
{
    WNDCLASSEXW wc = {
        sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L,
        hInstance, nullptr, nullptr, nullptr, nullptr,
        L"GPUImgProc3", nullptr
    };
    RegisterClassExW(&wc);

    HWND hwnd = CreateWindowW(
        wc.lpszClassName,
        L"GPU Image Processing  v3.0",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1400, 860,
        nullptr, nullptr, hInstance, nullptr);

    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        UnregisterClassW(wc.lpszClassName, hInstance);
        return 1;
    }

    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = "gpu_imgproc.ini";
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    SetupTheme();
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    // Load Segoe UI if available (looks much better than built-in)
    ImFontConfig fc;
    fc.OversampleH = 2; fc.OversampleV = 2;
    if (GetFileAttributesA("C:\\Windows\\Fonts\\segoeui.ttf") != INVALID_FILE_ATTRIBUTES)
        io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf", 15.0f, &fc);

    // App init
    LoadGPUInfo();
    g_app.addLog("GPU Image Processing  v3.0  —  ready.");
    if (g_app.gpus.empty())
        g_app.addLog("WARNING: no CUDA-capable GPU detected.");
    else
        for (auto& g : g_app.gpus)
            g_app.addLog("GPU: " + g.name + "  (" + std::to_string(g.memMB) + " MB)");

    // Main loop
    const float cc[4] = { 0.07f, 0.07f, 0.09f, 1.f };
    bool done = false;
    while (!done) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        if (done) break;

        if (g_ResizeWidth && g_ResizeHeight) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight,
                DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        RenderUI();

        ImGui::Render();
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, cc);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_pSwapChain->Present(1, 0);   // VSync on
    }

    // Cleanup
    if (g_imageSRV) { g_imageSRV->Release(); g_imageSRV = nullptr; }
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    CleanupDeviceD3D();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, hInstance);
    return 0;
}

// ============================================================
//  D3D11 helpers
// ============================================================
bool CreateDeviceD3D(HWND hWnd)
{
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate = { 60, 1 };
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    D3D_FEATURE_LEVEL fl;
    const D3D_FEATURE_LEVEL lvls[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    if (FAILED(D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        lvls, 2, D3D11_SDK_VERSION,
        &sd, &g_pSwapChain, &g_pd3dDevice, &fl, &g_pd3dDeviceContext)))
        return false;
    CreateRenderTarget();
    return true;
}
void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release();        g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release();  g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release();         g_pd3dDevice = nullptr; }
}
void CreateRenderTarget()
{
    ID3D11Texture2D* pBB;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBB));
    g_pd3dDevice->CreateRenderTargetView(pBB, nullptr, &g_mainRenderTargetView);
    pBB->Release();
}
void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) {
        g_mainRenderTargetView->Release();
        g_mainRenderTargetView = nullptr;
    }
}
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) return true;
    switch (msg) {
    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED) {
            g_ResizeWidth = LOWORD(lParam);
            g_ResizeHeight = HIWORD(lParam);
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) return 0;
        break;
    case WM_DESTROY:
        PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}