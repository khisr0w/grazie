/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  Mo 20 Mär 2023 20:56:15 CET                                  |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#include "memory.h"

// TODO(Abid): The custum allocators should be defined here
#define Free(ptr) free(ptr)
#define Malloc(ptr) malloc(ptr)
#define Calloc(ptr, size) calloc(ptr, size)

// NOTE(Abid): To be defined by the user upon including the header files!
#define MEMORYFOOTPRINTBYTES Gigabyte(2)

internal void InitGrazie(size_t MemSizeAlloc);

internal grazie_state *
GrazieState()
{
    local_persist grazie_state GrazieState = {0};
    local_persist boolean IsInitialized = false;

    if(!IsInitialized) 
    {
        InitGrazie(MEMORYFOOTPRINTBYTES);
        IsInitialized = true;
    }

    return &GrazieState;
}

internal void
InitGrazie(size_t MemSizeAlloc)
{
    // TODO(Abid): BUGGGGGGGG!!!!! 0_0
    grazie_state *State = GrazieState();
    State->MemPtr = malloc(MemSizeAlloc);
    State->MemMaxByteSize = MemSizeAlloc;
    State->MemByteUsed = 0;
}

