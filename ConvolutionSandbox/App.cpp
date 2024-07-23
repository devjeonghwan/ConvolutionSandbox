#include "App.h"
#include "ConvolutionSandbox.h"
#include "Convolutions.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <intrin.h>

void OnInitialize(AppConfig* appConfig)
{
    appConfig->sourceImagePath = L"wafer.jpg";

    appConfig->useThreadPool = true;
    appConfig->useGrayscale = false;
    appConfig->useRenderOverlayBalls = true;

    appConfig->kernelSize = 8;
}

void OnPrepare(Kernel* kernel)
{
    /*
    for (uint16_t index = 0; index < (kernel->size * kernel->size); index++)
    {
        kernel->data[index] = 1.0f / (kernel->size * kernel->size);
    }
    */

    /*
    uint16_t centerOfKernel = kernel->size / 2;
    for (uint16_t index = 0; index < (kernel->size * kernel->size); index++)
    {
        kernel->data[index] = -1;
    }
    kernel->data[(centerOfKernel * kernel->size) + centerOfKernel] = kernel->size * kernel->size - 1;
    */
}

void OnCompute(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    // 0.0149 GFLOP/s
    // In Python

    // 1.32 GFLOP/s
    // computeConvolution_Naive(inputImage, kernel, resultImage);

    // 6.51 GFLOP/s
    // computeConvolution_Naive_Mem2Reg(inputImage, kernel, resultImage);

    // 2.11 GFLOP/s
    // computeConvolution_Naive_Mem2Reg_WeirdCacheOptimization1(inputImage, kernel, resultImage);

    // 6.06 GFLOP/s
    // computeConvolution_Naive_Mem2Reg_WeirdCacheOptimization2(inputImage, kernel, resultImage);

    // 7.59 GFLOP/s
    // computeConvolution_Naive_Mem2Reg_CacheOptimization(inputImage, kernel, resultImage);

    // 9.49 GFLOP/s
    // computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion(inputImage, kernel, resultImage); 

    // 7.92 GFLOP/s
    // computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll4(inputImage, kernel, resultImage);

    // 15.06 GFLOP/s	
    // computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll4_SIMD_SSE(inputImage, kernel, resultImage);

    // 18.99 GFLOP/s
    // computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll8_SIMD_AVX(inputImage, kernel, resultImage);

    // 138.1 GFLOP/s
    // computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll4_SIMD_SSE_MultiThread(inputImage, kernel, resultImage);

    // 159.8 GFLOP/s
    computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll8_SIMD_AVX_MultiThread(inputImage, kernel, resultImage);
}

void OnRelease()
{
}
