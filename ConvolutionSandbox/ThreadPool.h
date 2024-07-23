#pragma once
#include <windows.h>
#include <stdio.h>
#include "SystemUtil.h"

typedef struct
{
    void (*function)(void* arg);
    void* arg;
} ThreadPoolTask;

typedef struct
{
    HANDLE* threads;
    ThreadPoolTask* taskQueue;
    size_t taskQueueSize;
    size_t taskCount;

    CRITICAL_SECTION lock;
    size_t activeThreads;
    size_t maxThreads;
    BOOL shutdown;
} ThreadPool;

DWORD WINAPI ThreadFunction(LPVOID param)
{
    ThreadPool* pool = (ThreadPool*)param;

    while (TRUE)
    {
        EnterCriticalSection(&pool->lock);
        if (pool->taskCount == 0 && pool->shutdown)
        {
            LeaveCriticalSection(&pool->lock);
            break;
        }
        if (pool->taskCount == 0)
        {
            LeaveCriticalSection(&pool->lock);
            Sleep(0);
            continue;
        }

        ThreadPoolTask task = pool->taskQueue[--pool->taskCount];
        LeaveCriticalSection(&pool->lock);

        task.function(task.arg);

        InterlockedDecrement(&pool->activeThreads);
    }
    return 0;
}

ThreadPool* ThreadPoolCreate(size_t maxThreads, size_t taskQueueSize)
{
    ThreadPool* pool = (ThreadPool*)malloc(sizeof(ThreadPool));
    if (pool == NULL)
    {
        return NULL;
    }

    pool->threads = (HANDLE*)malloc(sizeof(HANDLE) * maxThreads);
    if (pool->threads == NULL)
    {
        free(pool);
        return NULL;
    }

    pool->taskQueue = (ThreadPoolTask*)malloc(sizeof(ThreadPoolTask) * taskQueueSize);
    if (pool->taskQueue == NULL)
    {
        free(pool->threads);
        free(pool);
        return NULL;
    }

    pool->taskQueueSize = taskQueueSize;
    pool->taskCount = 0;
    pool->activeThreads = 0;
    pool->maxThreads = maxThreads;
    pool->shutdown = FALSE;

    InitializeCriticalSection(&pool->lock);

    for (size_t i = 0; i < maxThreads; i++)
    {
        pool->threads[i] = CreateThread(NULL, 0, ThreadFunction, pool, 0, NULL);

        if (pool->threads[i] == NULL)
        {
            for (size_t j = 0; j < i; j++)
            {
                CloseHandle(pool->threads[j]);
            }
            DeleteCriticalSection(&pool->lock);
            free(pool->taskQueue);
            free(pool->threads);
            free(pool);
            return NULL;
        }
    }

    return pool;
}

void ThreadPoolAddTask(ThreadPool* pool, void (*function)(void*), void* arg)
{
    EnterCriticalSection(&pool->lock);

    if (pool->taskCount < pool->taskQueueSize)
    {
        ThreadPoolTask task;
        task.function = function;
        task.arg = arg;
        pool->taskQueue[pool->taskCount++] = task;
        InterlockedIncrement(&pool->activeThreads);
    }

    LeaveCriticalSection(&pool->lock);
}

void ThreadPoolWait(ThreadPool* pool)
{
    while (TRUE)
    {
        if (InterlockedCompareExchange(&pool->taskCount, 0, 0) == 0 && InterlockedCompareExchange(&pool->activeThreads, 0, 0) == 0)
        {
            break;
        }

        Sleep(0);
    }
}

void ThreadPoolDestroy(ThreadPool* pool)
{
    EnterCriticalSection(&pool->lock);
    pool->shutdown = TRUE;
    LeaveCriticalSection(&pool->lock);

    for (size_t i = 0; i < pool->maxThreads; i++)
    {
        WaitForSingleObject(pool->threads[i], INFINITE);
        CloseHandle(pool->threads[i]);
    }

    DeleteCriticalSection(&pool->lock);
    free(pool->threads);
    free(pool->taskQueue);
    free(pool);
}

