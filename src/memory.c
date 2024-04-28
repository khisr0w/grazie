/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  Mo 20 Mär 2023 20:56:15 CET                                  |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#include "memory.h"

/* TODO(Abid): The custum allocators should be defined here */
#define _Free(ptr) free(ptr)
#define _Malloc(ptr) malloc(ptr)
#define _Calloc(ptr, size) calloc(ptr, size)
#define _Realloc(ptr, size) realloc(ptr, size)

#define PlatformAlloc(Size)

typedef struct {
    usize UsedSize;
    usize MaxSize;
    void *Ptr;
} arena;

inline internal arena
AllocateArena(usize BytesToAllocate) {

}
