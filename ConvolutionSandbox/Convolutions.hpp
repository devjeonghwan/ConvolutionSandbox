#pragma once
#include <stdio.h>
#include <intrin.h>
#include "ConvolutionSandbox.h"

inline void computeConvolution_Naive(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    for (size_t x = 0; x < resultImage->width; x++)
    {
        for (size_t y = 0; y < resultImage->height; y++)
        {
            for (size_t c = 0; c < resultImage->channels; c++)
            {
                resultImage->data[(c * resultImage->height + y) * resultImage->width + x] = 0;

                for (size_t ky = 0; ky < kernel->size; ky++)
                {
                    for (size_t kx = 0; kx < kernel->size; kx++)
                    {
                        resultImage->data[(c * resultImage->height + y) * resultImage->width + x] += 
                            inputImage->data[(c * inputImage->height + (y + ky)) * inputImage->width + x + kx] * kernel->data[ky * kernel->size + kx];
                    }
                }
            }
        }
    }
}


inline void computeConvolution_Naive_Mem2Reg(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    for (size_t x = 0; x < resultImage->width; x++)
    {
        for (size_t y = 0; y < resultImage->height; y++)
        {
            for (size_t c = 0; c < resultImage->channels; c++)
            {
                float sum = 0.0f;

                for (size_t ky = 0; ky < kernel->size; ky++)
                {
                    for (size_t kx = 0; kx < kernel->size; kx++)
                    {
                        sum += inputImage->data[(c * inputImage->height + (y + ky)) * inputImage->width + x + kx] * kernel->data[ky * kernel->size + kx];
                    }
                }

                resultImage->data[(c * resultImage->height + y) * resultImage->width + x] = sum;
            }
        }
    }
}


inline void computeConvolution_Naive_Mem2Reg_WeirdCacheOptimization1(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    for (size_t ky = 0; ky < kernel->size; ky++)
    {
        for (size_t x = 0; x < resultImage->width; x++)
        {
            for (size_t y = 0; y < resultImage->height; y++)
            {
                for (size_t c = 0; c < resultImage->channels; c++)
                {
                    float sum = ky == 0 ? 0.0f : resultImage->data[(c * resultImage->height + y) * resultImage->width + x];

                    for (size_t kx = 0; kx < kernel->size; kx++)
                    {
                        sum += inputImage->data[(c * inputImage->height + (y + ky)) * inputImage->width + x + kx] * kernel->data[ky * kernel->size + kx];
                    }

                    resultImage->data[(c * resultImage->height + y) * resultImage->width + x] = sum;
                }
            }
        }
    }
}


inline void computeConvolution_Naive_Mem2Reg_WeirdCacheOptimization2(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    for (size_t ky = 0; ky < kernel->size; ky++)
    {
        for (size_t c = 0; c < resultImage->channels; c++)
        {
            for (size_t y = 0; y < resultImage->height; y++)
            {
                for (size_t x = 0; x < resultImage->width; x++)
                {
                    float sum = ky == 0 ? 0.0f : resultImage->data[(c * resultImage->height + y) * resultImage->width + x];

                    for (size_t kx = 0; kx < kernel->size; kx++)
                    {
                        sum += inputImage->data[(c * inputImage->height + (y + ky)) * inputImage->width + x + kx] * kernel->data[ky * kernel->size + kx];
                    }

                    resultImage->data[(c * resultImage->height + y) * resultImage->width + x] = sum;
                }
            }
        }
    }
}


inline void computeConvolution_Naive_Mem2Reg_CacheOptimization(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    for (size_t c = 0; c < resultImage->channels; c++)
    {
        for (size_t y = 0; y < resultImage->height; y++)
        {
            for (size_t x = 0; x < resultImage->width; x++)
            {
                float sum = 0.0f;

                for (size_t ky = 0; ky < kernel->size; ky++)
                {
                    for (size_t kx = 0; kx < kernel->size; kx++)
                    {
                        sum += inputImage->data[(c * inputImage->height + (y + ky)) * inputImage->width + x + kx] * kernel->data[ky * kernel->size + kx];
                    }
                }

                resultImage->data[(c * resultImage->height + y) * resultImage->width + x] = sum;
            }
        }
    }
}


inline void computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    size_t inputWidthWithoutKernelSize = inputImage->width - kernel->size;
    size_t resultIndex = 0;

    for (size_t c = 0; c < resultImage->channels; c++)
    {
        size_t inputChannelIndex = c * inputImage->height;

        for (size_t y = 0; y < resultImage->height; y++)
        {
            size_t inputYIndex = (inputChannelIndex + y) * inputImage->width;

            for (size_t x = 0; x < resultImage->width; x++)
            {
                size_t localInputIndex = inputYIndex + x;
                size_t localKernelIndex = 0;

                float sum = 0.0f;

                for (size_t ky = 0; ky < kernel->size; ky++)
                {
                    for (size_t kx = 0; kx < kernel->size; kx++)
                    {
                        sum += inputImage->data[localInputIndex++] * kernel->data[localKernelIndex++];
                    }

                    localInputIndex += inputWidthWithoutKernelSize;
                }

                resultImage->data[resultIndex++] = sum;
            }
        }
    }
}


inline void computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll4(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    size_t inputWidthWithoutKernelSize = inputImage->width - kernel->size;
    size_t resultIndex = 0;

    for (size_t c = 0; c < resultImage->channels; c++)
    {
        size_t inputChannelIndex = c * inputImage->height;

        for (size_t y = 0; y < resultImage->height; y++)
        {
            size_t inputYIndex = (inputChannelIndex + y) * inputImage->width;

            for (size_t x = 0; x < resultImage->width; x++)
            {
                size_t localInputIndex = inputYIndex + x;
                size_t localKernelIndex = 0;

                float sum = 0.0f;

                for (size_t ky = 0; ky < kernel->size; ky++)
                {
                    size_t kx = 0;

                    for (; kx + 3 < kernel->size; kx += 4)
                    {
                        float v1 = inputImage->data[localInputIndex++] * kernel->data[localKernelIndex++];
                        float v2 = inputImage->data[localInputIndex++] * kernel->data[localKernelIndex++];
                        float v3 = inputImage->data[localInputIndex++] * kernel->data[localKernelIndex++];
                        float v4 = inputImage->data[localInputIndex++] * kernel->data[localKernelIndex++];

                        sum += v1 + v2 + v3 + v4;
                    }

                    for (; kx < kernel->size; kx++)
                    {
                        sum += inputImage->data[localInputIndex++] * kernel->data[localKernelIndex++];
                    }

                    localInputIndex += inputWidthWithoutKernelSize;
                }

                resultImage->data[resultIndex++] = sum;
            }
        }
    }
}


inline void computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll4_SIMD_SSE(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    size_t inputWidthWithoutKernelSize = inputImage->width - kernel->size;
    size_t resultIndex = 0;

    for (size_t c = 0; c < resultImage->channels; c++)
    {
        size_t inputChannelIndex = c * inputImage->height;

        for (size_t y = 0; y < resultImage->height; y++)
        {
            size_t inputYIndex = (inputChannelIndex + y) * inputImage->width;

            for (size_t x = 0; x < resultImage->width; x++)
            {
                size_t localInputIndex = inputYIndex + x;
                size_t localKernelIndex = 0;

                __m128 sumVector = _mm_setzero_ps();
                float sum = 0.0f;

                for (size_t ky = 0; ky < kernel->size; ky++)
                {
                    size_t kx = 0;

                    for (; kx + 3 < kernel->size; kx += 4)
                    {
                        __m128 inputVector = _mm_load_ps(&inputImage->data[localInputIndex]);
                        __m128 kernelVector = _mm_load_ps(&kernel->data[localKernelIndex]);

                        sumVector = _mm_fmadd_ps(inputVector, kernelVector, sumVector);

                        localInputIndex += 4;
                        localKernelIndex += 4;
                    }

                    for (; kx < kernel->size; kx++)
                    {
                        sum += inputImage->data[localInputIndex++] * kernel->data[localKernelIndex++];
                    }

                    localInputIndex += inputWidthWithoutKernelSize;
                }

                float sumArray[4];
                _mm_store_ps(sumArray, sumVector);

                resultImage->data[resultIndex++] = sumArray[0] + sumArray[1] + sumArray[2] + sumArray[3] + sum;
            }
        }
    }
}


inline void computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll8_SIMD_AVX(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    size_t inputWidthWithoutKernelSize = inputImage->width - kernel->size;
    size_t resultIndex = 0;

    for (size_t c = 0; c < resultImage->channels; c++)
    {
        size_t inputChannelIndex = c * inputImage->height;

        for (size_t y = 0; y < resultImage->height; y++)
        {
            size_t inputYIndex = (inputChannelIndex + y) * inputImage->width;

            for (size_t x = 0; x < resultImage->width; x++)
            {
                size_t localInputIndex = inputYIndex + x;
                size_t localKernelIndex = 0;

                __m256 sumVector = _mm256_setzero_ps();
                float sum = 0.0f;

                for (size_t ky = 0; ky < kernel->size; ky++)
                {
                    size_t kx = 0;

                    for (; kx + 7 < kernel->size; kx += 8)
                    {
                        __m256 inputVector = _mm256_load_ps(&inputImage->data[localInputIndex]);
                        __m256 kernelVector = _mm256_load_ps(&kernel->data[localKernelIndex]);

                        sumVector = _mm256_fmadd_ps(inputVector, kernelVector, sumVector);

                        localInputIndex += 8;
                        localKernelIndex += 8;
                    }

                    for (; kx < kernel->size; kx++)
                    {
                        sum += inputImage->data[localInputIndex++] * kernel->data[localKernelIndex++];
                    }

                    localInputIndex += inputWidthWithoutKernelSize;
                }

                float sumArray[8];
                _mm256_storeu_ps(sumArray, sumVector);

                resultImage->data[resultIndex++] = sumArray[0] + sumArray[1] + sumArray[2] + sumArray[3] + sumArray[4] + sumArray[5] + sumArray[6] + sumArray[7] + sum;
            }
        }
    }
}


#define MULTI_THREAD_COUNT 24

struct ThreadTaskInfo
{
    size_t id;

    Image* inputImage;
    Kernel* kernel;
    Image* resultImage;
};


void _computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll4_SIMD_SSE_MultiThread(void* argument)
{
    ThreadTaskInfo* taskInfo = (ThreadTaskInfo*) argument;

    Image* inputImage = taskInfo->inputImage;
    Kernel* kernel = taskInfo->kernel;
    Image* resultImage = taskInfo->resultImage;

    size_t taskVolume = resultImage->height / MULTI_THREAD_COUNT;

    size_t startResultY = taskInfo->id * taskVolume;
    size_t endResultY = taskInfo->id == MULTI_THREAD_COUNT - 1 ? resultImage->height : startResultY + taskVolume;

    size_t inputWidthWithoutKernelSize = inputImage->width - kernel->size;
    
    for (size_t c = 0; c < resultImage->channels; c++)
    {
        size_t resultIndex = (c * resultImage->height + startResultY) * resultImage->width;
        size_t inputChannelIndex = c * inputImage->height;

        for (size_t y = startResultY; y < endResultY; y++)
        {
            size_t inputYIndex = (inputChannelIndex + y) * inputImage->width;

            for (size_t x = 0; x < resultImage->width; x++)
            {
                size_t localInputIndex = inputYIndex + x;
                size_t localKernelIndex = 0;

                __m128 sumVector = _mm_setzero_ps();
                float sum = 0.0f;

                for (size_t ky = 0; ky < kernel->size; ky++)
                {
                    size_t kx = 0;

                    for (; kx + 3 < kernel->size; kx += 4)
                    {
                        __m128 inputVector = _mm_load_ps(&inputImage->data[localInputIndex]);
                        __m128 kernelVector = _mm_load_ps(&kernel->data[localKernelIndex]);

                        sumVector = _mm_fmadd_ps(inputVector, kernelVector, sumVector);

                        localInputIndex += 4;
                        localKernelIndex += 4;
                    }

                    for (; kx < kernel->size; kx++)
                    {
                        sum += inputImage->data[localInputIndex++] * kernel->data[localKernelIndex++];
                    }

                    localInputIndex += inputWidthWithoutKernelSize;
                }

                float sumArray[4];
                _mm_store_ps(sumArray, sumVector);

                resultImage->data[resultIndex++] = sumArray[0] + sumArray[1] + sumArray[2] + sumArray[3] + sum;
            }
        }
    }
}

inline void computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll4_SIMD_SSE_MultiThread(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    ThreadTaskInfo taskInfos[MULTI_THREAD_COUNT];

    for (size_t index = 0; index < MULTI_THREAD_COUNT; index++)
    {
        taskInfos[index].id = index;
        taskInfos[index].inputImage = inputImage;
        taskInfos[index].kernel = kernel;
        taskInfos[index].resultImage = resultImage;

        ExecuteThread(_computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll4_SIMD_SSE_MultiThread, &taskInfos[index]);
    }
    
    WaitForAllThread();
}


void _computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll8_SIMD_AVX_MultiThread(void* argument)
{
    ThreadTaskInfo* taskInfo = (ThreadTaskInfo*)argument;

    Image* inputImage = taskInfo->inputImage;
    Kernel* kernel = taskInfo->kernel;
    Image* resultImage = taskInfo->resultImage;

    size_t taskVolume = resultImage->height / MULTI_THREAD_COUNT;

    size_t startResultY = taskInfo->id * taskVolume;
    size_t endResultY = taskInfo->id == MULTI_THREAD_COUNT - 1 ? resultImage->height : startResultY + taskVolume;

    size_t inputWidthWithoutKernelSize = inputImage->width - kernel->size;

    for (size_t c = 0; c < resultImage->channels; c++)
    {
        size_t resultIndex = (c * resultImage->height + startResultY) * resultImage->width;
        size_t inputChannelIndex = c * inputImage->height;

        for (size_t y = startResultY; y < endResultY; y++)
        {
            size_t inputYIndex = (inputChannelIndex + y) * inputImage->width;

            for (size_t x = 0; x < resultImage->width; x++)
            {
                size_t localInputIndex = inputYIndex + x;
                size_t localKernelIndex = 0;

                __m256 sumVector = _mm256_setzero_ps();
                float sum = 0.0f;

                for (size_t ky = 0; ky < kernel->size; ky++)
                {
                    size_t kx = 0;

                    for (; kx + 7 < kernel->size; kx += 8)
                    {
                        __m256 inputVector = _mm256_load_ps(&inputImage->data[localInputIndex + kx]);
                        __m256 kernelVector = _mm256_load_ps(&kernel->data[localKernelIndex + kx]);

                        sumVector = _mm256_fmadd_ps(inputVector, kernelVector, sumVector);
                    }

                    for (; kx < kernel->size; kx++)
                    {
                        sum += inputImage->data[localInputIndex + kx] * kernel->data[localKernelIndex + kx];
                    }

                    localInputIndex += inputImage->width;
                    localKernelIndex += kernel->size;
                }

                float sumArray[8];
                _mm256_store_ps(sumArray, sumVector);

                resultImage->data[resultIndex++] = sumArray[0] + sumArray[1] + sumArray[2] + sumArray[3] + sumArray[4] + sumArray[5] + sumArray[6] + sumArray[7] + sum;
            }
        }
    }
}

inline void computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll8_SIMD_AVX_MultiThread(Image* inputImage, Kernel* kernel, Image* resultImage)
{
    ThreadTaskInfo taskInfos[MULTI_THREAD_COUNT];

    for (size_t index = 0; index < MULTI_THREAD_COUNT; index++)
    {
        taskInfos[index].id = index;
        taskInfos[index].inputImage = inputImage;
        taskInfos[index].kernel = kernel;
        taskInfos[index].resultImage = resultImage;

        ExecuteThread(_computeConvolution_Naive_Mem2Reg_CacheOptimization_LoopInvariantCodeMotion_LoopUnroll8_SIMD_AVX_MultiThread, &taskInfos[index]);
    }

    WaitForAllThread();
}