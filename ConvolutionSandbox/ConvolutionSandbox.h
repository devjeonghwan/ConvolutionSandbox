#pragma once

#include <cinttypes>

#define USE_ALIGNED_MALLOC          TRUE

#define STATISTICS_COUNT            40
#define STATISTICS_CUTOFF_COUNT     4

struct Image
{
    uint16_t width;
    uint16_t height;
    uint16_t channels;

    float* data;
};

struct Kernel
{
    uint16_t size;

    float* data;
};

struct AppConfig
{
    bool useRenderOverlayBalls;
    bool useThreadPool;
    bool useGrayscale;

    const wchar_t* sourceImagePath;

    uint16_t inputImageWidth;
    uint16_t inputImageHeight;

    uint16_t kernelSize;
};

void PassThroughImage(Image* sourceImage, Image* destinationImage);

size_t GetLogicalProcessorCount();

void ExecuteThread(void (*function)(void* argument), void* argument);

void WaitForAllThread();