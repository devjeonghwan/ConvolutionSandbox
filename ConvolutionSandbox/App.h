#pragma once
#include "ConvolutionSandbox.h"

void OnInitialize(AppConfig* appConfig);

void OnPrepare(Kernel* kernel);

void OnCompute(Image* inputImage, Kernel* kernel, Image* resultImage);

void OnRelease();