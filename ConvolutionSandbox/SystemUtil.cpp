#include "SystemUtil.h"

bool SetMainThreadAffinityToPerformanceCores()
{
    HANDLE process = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();

    DWORD length = 0;

    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(length);

    if (buffer == NULL)
    {
        return false;
    }

    GetLogicalProcessorInformationEx(RelationProcessorCore, buffer, &length);

    DWORD_PTR performanceCoreMask = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX ptr = buffer;

    while ((BYTE*)ptr < (BYTE*)buffer + length)
    {
        for (WORD i = 0; i < ptr->Processor.GroupCount; ++i)
        {
            if (ptr->Processor.GroupMask[i].Group == 0)
            {
                performanceCoreMask |= ptr->Processor.GroupMask[i].Mask;
            }
        }

        ptr = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)((BYTE*)ptr + ptr->Size);
    }

    free(buffer);

    DWORD_PTR result = SetThreadAffinityMask(thread, performanceCoreMask);

    return result != NULL;
}