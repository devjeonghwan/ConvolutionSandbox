#include "ConvolutionSandbox.h"
#include "App.h"
#include "ThreadPool.h"
#include "SystemUtil.h"

#include <Windows.h>
#include <gdiplus.h>
#include <math.h>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <vector>
#include <immintrin.h> 

#pragma comment (lib,"Gdiplus.lib")

#define ALIGN(x, align)     (((x) + (align) - 1) & ~((align) - 1))
#define CLAMP(x, min, max)  (x < min ? min : (x > max ? max : x))

#define GIGA                1000000000

#define MAX_BOXES           50
#define BOX_SIZE            60
#define BOX_HAFL_SIZE       (BOX_SIZE / 2)
#define BOX_VELOCITY_RANGE  5

#define COLORS_COUNT 12
const float    COLORS[COLORS_COUNT][3] = {
    {1.0f, 0.0f, 0.0f},
    {1.0f, 1.0f, 0.0f},
    {1.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, 1.0f},
    {0.0f, 0.0f, 1.0f},

    {0.9f, 0.3f, 0.3f},
    {0.9f, 0.9f, 0.3f},
    {0.9f, 0.3f, 0.9f},
    {0.3f, 0.9f, 0.3f},
    {0.3f, 0.9f, 0.9f},
    {0.3f, 0.3f, 0.9f},
};

const float PI = 3.14159265359f;
const float PI2 = 2.0f * PI;

// Fields
AppConfig       g_appConfig;
ThreadPool* g_threadPool;

// Buffers
Image           g_backgroundImage;
Image           g_inputImage;
Kernel          g_kernel;
Image           g_resultImage;

// Statistics
size_t          g_NTimesCounter;

double          g_totalOperations; // Constant

double          g_NTimesComputeFps[STATISTICS_COUNT];
double          g_latestComputeFps;

double          g_NTimesTotalFps[STATISTICS_COUNT];
double          g_latestTotalFps;

double          g_NTimesGflops[STATISTICS_COUNT];
double          g_latestGflops;

// Handles
HWND            g_hWnd;
HDC             g_hDc;

HBITMAP         g_hBitmap;
BITMAPINFO      g_bitmapInfo;
BYTE* g_bitmapPixels;

// Animation Fields
size_t          g_animationFrame;
int             g_boxX[MAX_BOXES];
int             g_boxY[MAX_BOXES];
float           g_boxVelocityX[MAX_BOXES];
float           g_boxVelocityY[MAX_BOXES];

template <bool useGrayscale>
static inline void RenderBackground(float* data, size_t width, size_t height, size_t animationFrame)
{
    size_t blueChannel = 0;
    size_t greenChannel = height * width;
    size_t redChannel = greenChannel * 2;

    float time = (float)animationFrame / 20.0f;

    size_t globalIndex = 0;

    for (size_t y = 0; y < height; y++)
    {
        __m256 yVector = _mm256_set1_ps((float)y * 0.05f);
        __m256 cosineYVector = _mm256_cos_ps(yVector);
        __m256 sineYVector = _mm256_sin_ps(yVector);

        for (size_t x = 0; x < width; x += 8)
        {
            __m256 xVector = _mm256_setr_ps(
                (float)x + 0, (float)x + 1, (float)x + 2, (float)x + 3,
                (float)x + 4, (float)x + 5, (float)x + 6, (float)x + 7
            );

            __m256 scaledXVector = _mm256_mul_ps(xVector, _mm256_set1_ps(0.05f));
            __m256 cosineXVector = _mm256_cos_ps(scaledXVector);
            __m256 sineXVector = _mm256_sin_ps(scaledXVector);

            __m256 wave1 = _mm256_mul_ps(sineXVector, cosineYVector);
            __m256 wave2 = _mm256_mul_ps(sineYVector, cosineXVector);

            __m256 blueValue = _mm256_div_ps(_mm256_cos_ps(_mm256_mul_ps(_mm256_set1_ps(time), _mm256_mul_ps(wave1, _mm256_set1_ps(0.5f)))), _mm256_set1_ps(2.0f));
            __m256 redValue = _mm256_div_ps(_mm256_cos_ps(_mm256_mul_ps(_mm256_set1_ps(time), _mm256_mul_ps(wave2, _mm256_set1_ps(0.5f)))), _mm256_set1_ps(2.0f));
            __m256 greenValue = _mm256_div_ps(_mm256_sin_ps(_mm256_mul_ps(_mm256_set1_ps(time), _mm256_mul_ps(wave2, _mm256_set1_ps(0.5f)))), _mm256_set1_ps(2.0f));

            float blueValueArray[8];
            float redValueArray[8];
            float greenValueArray[8];

            _mm256_storeu_ps(blueValueArray, blueValue);
            _mm256_storeu_ps(redValueArray, redValue);
            _mm256_storeu_ps(greenValueArray, greenValue);

            for (int i = 0; i < 8; ++i)
            {
                size_t index = globalIndex++;

                if (useGrayscale)
                {
                    data[index] = (blueValueArray[i] + redValueArray[i] + greenValueArray[i]) / 3;
                }
                else
                {
                    data[blueChannel + index] = blueValueArray[i];
                    data[redChannel + index] = redValueArray[i];
                    data[greenChannel + index] = greenValueArray[i];
                }
            }
        }
    }
}

template <bool useGrayscale>
static inline void RenderBoxes(float* data, size_t width, size_t height)
{
    size_t blueChannel = 0;
    size_t greenChannel = g_inputImage.height * g_inputImage.width;
    size_t redChannel = greenChannel * 2;

    for (size_t i = 0; i < MAX_BOXES; ++i)
    {
        const size_t boxY = CLAMP(g_boxY[i], 0, height - BOX_SIZE);
        const size_t boxX = CLAMP(g_boxX[i], 0, width - BOX_SIZE);
        const size_t colorIndex = i % COLORS_COUNT;

        const float valueB = COLORS[colorIndex][0];
        const float valueG = COLORS[colorIndex][1];
        const float valueR = COLORS[colorIndex][2];

        if (useGrayscale)
        {
            for (size_t y = boxY; y < boxY + BOX_SIZE; y++)
            {
                size_t yIndex = y * width + boxX;

                for (size_t x = boxX; x < boxX + BOX_SIZE; x++)
                {
                    data[yIndex] = (valueB + valueG + valueR) / 3;

                    yIndex++;
                }
            }
        }
        else
        {
            for (size_t y = boxY; y < boxY + BOX_SIZE; y++)
            {
                size_t yIndex = y * width + boxX;

                for (size_t x = boxX; x < boxX + BOX_SIZE; x++)
                {
                    data[blueChannel + yIndex] = valueB;
                    data[greenChannel + yIndex] = valueG;
                    data[redChannel + yIndex] = valueR;

                    yIndex++;
                }
            }
        }
    }
}

static void UpdatePixels()
{
    size_t blueChannel = 0;
    size_t greenChannel = g_inputImage.height * g_inputImage.width;
    size_t redChannel = greenChannel * 2;

    if (g_backgroundImage.data != NULL)
    {
        memcpy_s(g_inputImage.data,
                 g_inputImage.height * g_inputImage.width * g_inputImage.channels * sizeof(float),
                 g_backgroundImage.data,
                 g_backgroundImage.height * g_backgroundImage.width * g_backgroundImage.channels * sizeof(float));
    }
    else
    {
        if (g_appConfig.useGrayscale)
        {
            RenderBackground<true>(g_inputImage.data, g_inputImage.width, g_inputImage.height, g_animationFrame);
        }
        else
        {
            RenderBackground<false>(g_inputImage.data, g_inputImage.width, g_inputImage.height, g_animationFrame);
        }
    }

    if (g_appConfig.useRenderOverlayBalls)
    {
        if (g_appConfig.useGrayscale)
        {
            RenderBoxes<true>(g_inputImage.data, g_inputImage.width, g_inputImage.height);
        }
        else
        {
            RenderBoxes<false>(g_inputImage.data, g_inputImage.width, g_inputImage.height);
        }

        for (size_t i = 0; i < MAX_BOXES; ++i)
        {
            g_boxX[i] += (int)g_boxVelocityX[i];
            g_boxY[i] += (int)g_boxVelocityY[i];
        }

        for (size_t i = 0; i < MAX_BOXES; ++i)
        {
            if (g_boxX[i] <= 0 || g_boxX[i] >= g_inputImage.width - BOX_SIZE)
            {
                g_boxVelocityX[i] = -g_boxVelocityX[i];
            }

            if (g_boxY[i] <= 0 || g_boxY[i] >= g_inputImage.height - BOX_SIZE)
            {
                g_boxVelocityY[i] = -g_boxVelocityY[i];
            }

            for (size_t j = i + 1; j < MAX_BOXES; ++j)
            {
                float distanceX = (float)abs(g_boxX[j] - g_boxX[i]) - BOX_SIZE;
                float distanceY = (float)abs(g_boxY[j] - g_boxY[i]) - BOX_SIZE;

                if (distanceX <= 0 && distanceY < 0)
                {
                    float tempX = g_boxVelocityX[i];
                    float tempY = g_boxVelocityY[i];
                    float deltaX = abs(distanceX);
                    float deltaY = abs(distanceY);

                    if (deltaX < deltaY)
                    {
                        if (g_boxX[i] < g_boxX[j])
                        {
                            g_boxX[i] -= (int)(deltaX / 2);
                            g_boxX[j] += (int)(deltaX / 2);
                        }
                        else
                        {
                            g_boxX[i] += (int)(deltaX / 2);
                            g_boxX[j] -= (int)(deltaX / 2);
                        }

                        g_boxVelocityX[i] = g_boxVelocityX[j];
                        g_boxVelocityX[j] = tempX;
                    }
                    else
                    {
                        if (g_boxY[i] < g_boxY[j])
                        {
                            g_boxY[i] -= (int)(deltaY / 2);
                            g_boxY[j] += (int)(deltaY / 2);
                        }
                        else
                        {
                            g_boxY[i] += (int)(deltaY / 2);
                            g_boxY[j] -= (int)(deltaY / 2);
                        }

                        g_boxVelocityY[i] = g_boxVelocityY[j];
                        g_boxVelocityY[j] = tempY;
                    }
                }
            }
        }
    }

    g_animationFrame += 1;

    auto computeStartTime = std::chrono::steady_clock::now();
    OnCompute(&g_inputImage, &g_kernel, &g_resultImage);
    auto computeEndTime = std::chrono::steady_clock::now();

    std::chrono::duration<double> computeElapsed = computeEndTime - computeStartTime;

    g_NTimesComputeFps[g_NTimesCounter] = 1.0f / computeElapsed.count();
    g_NTimesGflops[g_NTimesCounter] = (g_totalOperations * (1 / computeElapsed.count())) / GIGA;

    size_t bitmapWidth = ALIGN(g_resultImage.width, 4);

    for (size_t c = 0; c < 3; c++)
    {
        size_t resultIndex = g_appConfig.useGrayscale ? 0 : g_resultImage.height * g_resultImage.width * c;

        for (size_t y = 0; y < g_resultImage.height; y++)
        {
            size_t bitmapYIndex = (y * bitmapWidth) * 3 + c;
            size_t x = 0;

            for (; (x + 7) < g_resultImage.width; x += 8)
            {
                __m256 values = _mm256_loadu_ps(&g_resultImage.data[resultIndex]);

                __m256 zero = _mm256_set1_ps(0.0f);
                __m256 one = _mm256_set1_ps(1.0f);
                values = _mm256_min_ps(_mm256_max_ps(values, zero), one);

                __m256 scale = _mm256_set1_ps(255.0f);
                values = _mm256_mul_ps(values, scale);

                __m256i intValues = _mm256_cvtps_epi32(values);

                g_bitmapPixels[bitmapYIndex] = (BYTE)_mm256_extract_epi32(intValues, 0);
                g_bitmapPixels[bitmapYIndex + 3] = (BYTE)_mm256_extract_epi32(intValues, 1);
                g_bitmapPixels[bitmapYIndex + 6] = (BYTE)_mm256_extract_epi32(intValues, 2);
                g_bitmapPixels[bitmapYIndex + 9] = (BYTE)_mm256_extract_epi32(intValues, 3);
                g_bitmapPixels[bitmapYIndex + 12] = (BYTE)_mm256_extract_epi32(intValues, 4);
                g_bitmapPixels[bitmapYIndex + 15] = (BYTE)_mm256_extract_epi32(intValues, 5);
                g_bitmapPixels[bitmapYIndex + 18] = (BYTE)_mm256_extract_epi32(intValues, 6);
                g_bitmapPixels[bitmapYIndex + 21] = (BYTE)_mm256_extract_epi32(intValues, 7);

                resultIndex += 8;
                bitmapYIndex += 24;
            }

            for (; x < g_resultImage.width; x++)
            {
                float value = g_resultImage.data[resultIndex++];
                g_bitmapPixels[bitmapYIndex] = (BYTE)(CLAMP(value, 0.0f, 1.0f) * 255.0f);
                bitmapYIndex += 3;
            }
        }
    }
}

static double CalculateStatisticsValue(double* values, size_t n, size_t cutoff)
{
    if (n <= 2 * cutoff)
    {
        return 0.0;
    }

    std::vector<double> vec(values, values + n);

    std::sort(vec.begin(), vec.end());

    auto start = vec.begin() + cutoff;
    auto end = vec.end() - cutoff;

    double sum = std::accumulate(start, end, 0.0);

    size_t count = end - start;

    return sum / count;
}

static LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_CLOSE:
        DestroyWindow(hWnd);
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_PAINT:
    {
        auto totalStartTime = std::chrono::steady_clock::now();
        UpdatePixels();
        auto totalEndTime = std::chrono::steady_clock::now();

        std::chrono::duration<double> totalElapsed = totalEndTime - totalStartTime;

        g_NTimesTotalFps[g_NTimesCounter] = 1.0f / totalElapsed.count();
        g_NTimesCounter += 1;

        if (g_NTimesCounter >= STATISTICS_COUNT)
        {
            g_latestComputeFps = CalculateStatisticsValue(g_NTimesComputeFps, g_NTimesCounter, STATISTICS_CUTOFF_COUNT);
            g_latestTotalFps = CalculateStatisticsValue(g_NTimesTotalFps, g_NTimesCounter, STATISTICS_CUTOFF_COUNT);
            g_latestGflops = CalculateStatisticsValue(g_NTimesGflops, g_NTimesCounter, STATISTICS_CUTOFF_COUNT);

            g_NTimesCounter = 0;
        }

        PAINTSTRUCT paintStruct;
        HDC hdc = BeginPaint(hWnd, &paintStruct);
        HDC hdcMemory = CreateCompatibleDC(hdc);

        SelectObject(hdcMemory, g_hBitmap);
        BitBlt(hdc, 0, 0, g_resultImage.width, g_resultImage.height, hdcMemory, 0, 0, SRCCOPY);

        DeleteDC(hdcMemory);
        EndPaint(hWnd, &paintStruct);

        std::wstring title = L"Sandbox - [FPS: " + std::to_wstring(g_latestComputeFps) + L"] [GFLOP/s: " + std::to_wstring(g_latestGflops) + L"] [Render: " + std::to_wstring(g_latestTotalFps) + L"]";
        SetWindowText(g_hWnd, title.c_str());

        InvalidateRect(g_hWnd, NULL, FALSE);
        break;
    }
    default:
        return DefWindowProc(hWnd, uMsg, wParam, lParam);
    }
    return 0;
}

static float* LoadImageWithResizeGrayscale(const wchar_t* filename, uint16_t resizeWidth, uint16_t resizeHeight, bool useGrayscale)
{
    using namespace Gdiplus;

    float* imageData = NULL;

    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    BitmapData bitmapData;

    Bitmap* bitmap = nullptr;
    Bitmap* resizedBitmap = nullptr;

    GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    bitmap = new Bitmap(filename);

    if (bitmap->GetLastStatus() != Ok)
    {
        goto CLEAN_UP;
    }

    resizedBitmap = new Bitmap(resizeWidth, resizeHeight, bitmap->GetPixelFormat());

    if (resizedBitmap->GetLastStatus() != Ok)
    {
        goto CLEAN_UP;
    }

    {
        Graphics graphics(resizedBitmap);

        graphics.SetInterpolationMode(InterpolationModeHighQualityBicubic);
        graphics.SetSmoothingMode(SmoothingModeHighQuality);
        graphics.SetPixelOffsetMode(PixelOffsetModeHighQuality);

        graphics.DrawImage(bitmap, 0, 0, resizeWidth, resizeHeight);

        Rect rect(0, 0, resizeWidth, resizeHeight);

        resizedBitmap->LockBits(&rect, ImageLockModeRead, PixelFormat24bppRGB, &bitmapData);

        size_t targetChannels = useGrayscale ? 1 : 3;
#if USE_ALIGNED_MALLOC
        imageData = (float*)_aligned_malloc(resizeWidth * resizeHeight * targetChannels * sizeof(float), 32);
#else
        imageData = (float*)malloc(resizeWidth * resizeHeight * targetChannels * sizeof(float));
#endif

        if (imageData == NULL)
        {
            goto CLEAN_UP;
        }

        BYTE* source = (BYTE*)bitmapData.Scan0;

        if (useGrayscale)
        {
            for (size_t y = 0; y < resizeHeight; y++)
            {
                for (size_t x = 0; x < resizeWidth; x++)
                {
                    size_t sourceIndex = y * bitmapData.Stride + x * 3;
                    float value1 = (float)source[sourceIndex] / 255.0f;
                    float value2 = (float)source[sourceIndex + 1] / 255.0f;
                    float value3 = (float)source[sourceIndex + 2] / 255.0f;

                    size_t destinationIndex = (y * resizeWidth) + x;

                    imageData[destinationIndex] = (value1 + value2 + value3) / 3;
                }
            }
        }
        else
        {
            for (size_t c = 0; c < 3; c++)
            {
                for (size_t y = 0; y < resizeHeight; y++)
                {
                    for (size_t x = 0; x < resizeWidth; x++)
                    {
                        size_t sourceIndex = y * bitmapData.Stride + x * 3 + c;
                        size_t destinationIndex = (c * resizeHeight * resizeWidth) + (y * resizeWidth) + x;

                        imageData[destinationIndex] = (float)source[sourceIndex] / 255.0f;
                    }
                }
            }
        }

        resizedBitmap->UnlockBits(&bitmapData);
    }

CLEAN_UP:
    if (bitmap != NULL)
    {
        delete bitmap;
    }

    if (resizedBitmap != NULL)
    {
        delete resizedBitmap;
    }

    GdiplusShutdown(gdiplusToken);

    return imageData;
}

int WINAPI wWinMain(_In_        HINSTANCE   hInstance,
                    _In_opt_    HINSTANCE   hPrevInstance,
                    _In_        LPWSTR      lpCmdLine,
                    _In_        int         nShowCmd)
{
    // Default Values
    g_appConfig.useRenderOverlayBalls = true;
    g_appConfig.useThreadPool = false;
    g_appConfig.useGrayscale = false;

    g_appConfig.sourceImagePath = NULL;

    g_appConfig.inputImageWidth = 1440;
    g_appConfig.inputImageHeight = 900;

    g_appConfig.kernelSize = 5;

    // Initialize
    OnInitialize(&g_appConfig);

    if (!SetMainThreadAffinityToPerformanceCores())
    {
        MessageBox(NULL, L"Failed to set thread affinity mask.", L"Error", MB_ICONEXCLAMATION | MB_OK);

        return 0;
    }

    g_threadPool = ThreadPoolCreate(g_appConfig.useThreadPool ? GetLogicalProcessorCount() : 0, 1024 * 1024);

    if (g_threadPool == NULL)
    {
        MessageBox(NULL, L"Failed to initialize thread pool.", L"Error", MB_ICONEXCLAMATION | MB_OK);

        return 0;
    }

    g_backgroundImage.width = -1;
    g_backgroundImage.height = -1;
    g_backgroundImage.channels = -1;
    g_backgroundImage.data = NULL;

    if (g_appConfig.sourceImagePath != NULL)
    {
        g_backgroundImage.width = ALIGN(g_appConfig.inputImageWidth, 4);
        g_backgroundImage.height = g_appConfig.inputImageHeight;
        g_backgroundImage.channels = g_appConfig.useGrayscale ? 1 : 3;

        g_backgroundImage.data = LoadImageWithResizeGrayscale(g_appConfig.sourceImagePath, g_backgroundImage.width, g_backgroundImage.height, g_appConfig.useGrayscale);

        if (g_backgroundImage.data == NULL)
        {
            MessageBox(NULL, L"Failed to load image.", L"Error", MB_ICONEXCLAMATION | MB_OK);

            return 0;
        }
    }

    g_animationFrame = 0;

    if (g_appConfig.useRenderOverlayBalls)
    {
        const int padding = BOX_SIZE;
        int currentX = 0;
        int currentY = 0;

        for (size_t i = 0; i < MAX_BOXES; ++i)
        {
            g_boxX[i] = currentX;
            g_boxY[i] = currentY;
            g_boxVelocityX[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * BOX_VELOCITY_RANGE;
            g_boxVelocityY[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * BOX_VELOCITY_RANGE;

            currentX += BOX_SIZE + padding;

            if (currentX >= g_appConfig.inputImageWidth - BOX_SIZE)
            {
                currentX = 0;
                currentY += BOX_SIZE + padding;

                if (currentY >= g_appConfig.inputImageHeight - BOX_SIZE)
                {
                    MessageBox(NULL, L"Failed to create boxes for animation. (Too many boxes or Too big box size)", L"Error", MB_ICONEXCLAMATION | MB_OK);

                    return 0;
                }
            }
        }

        if (currentY == 0)
        {
            for (size_t i = 0; i < MAX_BOXES; ++i)
            {
                g_boxX[i] += (g_appConfig.inputImageWidth / 2) - ((currentX - BOX_SIZE + padding) / 2);
            }
        }

        for (size_t i = 0; i < MAX_BOXES; ++i)
        {
            g_boxY[i] += (g_appConfig.inputImageHeight / 2) - ((currentY - BOX_SIZE + padding) / 2);
        }
    }

    if (g_appConfig.inputImageWidth < g_appConfig.kernelSize || g_appConfig.inputImageHeight < g_appConfig.kernelSize)
    {
        MessageBox(NULL, L"Failed to create result buffer. (Too small input image)", L"Error", MB_ICONEXCLAMATION | MB_OK);

        return 0;
    }

    g_inputImage.width = g_appConfig.inputImageWidth;
    g_inputImage.height = g_appConfig.inputImageHeight;
    g_inputImage.channels = g_appConfig.useGrayscale ? 1 : 3;

#if USE_ALIGNED_MALLOC
    g_inputImage.data = (float*)_aligned_malloc(g_inputImage.width * g_inputImage.height * g_inputImage.channels * sizeof(float), 32);
#else
    g_inputImage.data = (float*)malloc(g_inputImage.width * g_inputImage.height * g_inputImage.channels * sizeof(float));
#endif

    g_resultImage.width = (g_appConfig.inputImageWidth - g_appConfig.kernelSize) + 1;
    g_resultImage.height = (g_appConfig.inputImageHeight - g_appConfig.kernelSize) + 1;
    g_resultImage.channels = g_inputImage.channels;

#if USE_ALIGNED_MALLOC
    g_resultImage.data = (float*)_aligned_malloc(g_resultImage.width * g_resultImage.height * g_resultImage.channels * sizeof(float), 32);
#else
    g_resultImage.data = (float*)malloc(g_resultImage.width * g_resultImage.height * g_resultImage.channels * sizeof(float));
#endif

    g_kernel.size = g_appConfig.kernelSize;

#if USE_ALIGNED_MALLOC
    g_kernel.data = (float*)_aligned_malloc(g_kernel.size * g_kernel.size * sizeof(float), 32);
#else
    g_kernel.data = (float*)malloc(g_kernel.size * g_kernel.size * sizeof(float));
#endif

    uint16_t centerOfKernel = g_kernel.size / 2;
    for (uint16_t index = 0; index < (g_kernel.size * g_kernel.size); index++)
    {
        g_kernel.data[index] = 0;
    }
    g_kernel.data[(centerOfKernel * g_kernel.size) + centerOfKernel] = 1;

    g_totalOperations = g_resultImage.channels * g_resultImage.height * g_resultImage.width * (2 * g_kernel.size * g_kernel.size);

    WNDCLASSEX windowClass;

    windowClass.cbSize = sizeof(WNDCLASSEX);
    windowClass.style = 0;
    windowClass.lpfnWndProc = WndProc;
    windowClass.cbClsExtra = 0;
    windowClass.cbWndExtra = 0;
    windowClass.hInstance = hInstance;
    windowClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    windowClass.hbrBackground = (HBRUSH)(COLOR_WINDOW);
    windowClass.lpszMenuName = NULL;
    windowClass.lpszClassName = L"ConvolutionSandboxWindowClass";
    windowClass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

    if (!RegisterClassEx(&windowClass))
    {
        MessageBox(NULL, L"Failed to register window.", L"Error", MB_ICONEXCLAMATION | MB_OK);

        return 0;
    }

    g_hWnd = CreateWindowEx(WS_EX_CLIENTEDGE,
                            windowClass.lpszClassName,
                            L"Sandbox",
                            WS_OVERLAPPED | WS_SYSMENU,
                            CW_USEDEFAULT,
                            CW_USEDEFAULT,
                            ALIGN(g_resultImage.width, 4), g_resultImage.height,
                            NULL, NULL, hInstance, NULL);

    if (g_hWnd == NULL)
    {
        MessageBox(NULL, L"Failed to create window.", L"Error", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    g_hDc = GetDC(g_hWnd);

    g_bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    g_bitmapInfo.bmiHeader.biWidth = ALIGN(g_resultImage.width, 4);
    g_bitmapInfo.bmiHeader.biHeight = -g_resultImage.height;
    g_bitmapInfo.bmiHeader.biPlanes = 1;
    g_bitmapInfo.bmiHeader.biBitCount = 24;
    g_bitmapInfo.bmiHeader.biCompression = BI_RGB;
    g_bitmapInfo.bmiHeader.biSizeImage = 0;
    g_bitmapInfo.bmiHeader.biXPelsPerMeter = 0;
    g_bitmapInfo.bmiHeader.biYPelsPerMeter = 0;
    g_bitmapInfo.bmiHeader.biClrUsed = 0;
    g_bitmapInfo.bmiHeader.biClrImportant = 0;

    g_hBitmap = CreateDIBSection(g_hDc, &g_bitmapInfo, DIB_RGB_COLORS, (void**)&g_bitmapPixels, NULL, 0);

    if (g_hBitmap == NULL || g_bitmapPixels == NULL)
    {
        MessageBox(NULL, L"Failed to create bitmap.", L"Error", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    OnPrepare(&g_kernel);

    RECT rectClient, rectWindow;
    int posX, posY;

    GetClientRect(g_hWnd, &rectClient);
    GetWindowRect(g_hWnd, &rectWindow);

    posX = (GetSystemMetrics(SM_CXSCREEN) - rectWindow.right + rectWindow.left) / 2;
    posY = (GetSystemMetrics(SM_CYSCREEN) - rectWindow.bottom + rectWindow.top) / 2;

    MoveWindow(g_hWnd, posX, posY, rectWindow.right - rectWindow.left, rectWindow.bottom - rectWindow.top, TRUE);

    ShowWindow(g_hWnd, nShowCmd);
    UpdateWindow(g_hWnd);

    InvalidateRect(g_hWnd, NULL, TRUE);

    MSG msg;

    while (GetMessage(&msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    OnRelease();

#if USE_ALIGNED_MALLOC
    _aligned_free(g_kernel.data);
    _aligned_free(g_resultImage.data);
    _aligned_free(g_inputImage.data);

    if (g_backgroundImage.data != NULL)
    {
        _aligned_free(g_backgroundImage.data);
    }
#else
    free(g_kernel.data);
    free(g_resultImage.data);
    free(g_inputImage.data);

    if (g_backgroundImage.data != NULL)
    {
        free(g_inputImage.data);
    }
#endif

    DeleteObject(g_hBitmap);
    ReleaseDC(g_hWnd, g_hDc);

    ThreadPoolWait(g_threadPool);
    ThreadPoolDestroy(g_threadPool);

    return (int)msg.wParam;
}

void PassThroughImage(Image* sourceImage, Image* destinationImage)
{
    size_t resultIndex = 0;

    for (size_t c = 0; c < destinationImage->channels; c++)
    {
        size_t inputChannelIndex = c * sourceImage->height;

        for (size_t y = 0; y < destinationImage->height; y++)
        {
            size_t inputYIndex = (inputChannelIndex + y) * sourceImage->width;

            for (size_t x = 0; x < destinationImage->width; x++)
            {
                destinationImage->data[resultIndex++] = sourceImage->data[inputYIndex++];
            }
        }
    }
}

size_t GetLogicalProcessorCount()
{
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);

    return sysInfo.dwNumberOfProcessors;
}

void ExecuteThread(void (*function)(void* argument), void* argument)
{
    ThreadPoolAddTask(g_threadPool, function, argument);
}

void WaitForAllThread()
{
    ThreadPoolWait(g_threadPool);
}
