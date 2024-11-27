/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  Mo 20 Mär 2023 20:56:15 CET                                   |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#include "memory.h"

/* TODO(Abid): The custum allocators should be defined here */
#define Free(ptr) free(ptr)
#define Malloc(ptr) malloc(ptr)
#define _Calloc(ptr, size) calloc(ptr, size)
#define _Realloc(ptr, size) realloc(ptr, size)

inline internal void *
gzPlatoformMemAllocate(usize Size) {
    void *Result = NULL;

#ifdef GRAZIE_PLT_WIN
    Result = VirtualAlloc(NULL, Size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if(Result == NULL) {
        GetLastError();
        exit(EXIT_FAILURE);
    }
#endif 

#ifdef GRAZIE_PLT_LINUX
    Result = mmap(NULL, Size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if(Result == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }
#endif 

    return Result;
}

internal mem_arena
gzMemArenaAllocate(usize BytesToAllocate) {
    mem_arena Arena = {0};
    Arena.Size = BytesToAllocate;
    Arena.Ptr = gzPlatoformMemAllocate(Arena.Size);
    Arena.Used = 0;

    return Arena;
}

inline internal temp_memory
gz_mem_temp_begin(mem_arena *Arena) {
    temp_memory Result = {0};

    Result.Arena = Arena;
    Result.Used = Arena->Used;

    ++Arena->TempCount;

    return Result;
}

inline internal void
gz_mem_temp_end(temp_memory TempMem) {
    mem_arena *Arena = TempMem.Arena;
    assert(Arena->Used >= TempMem.Used, "something was freed when it shouldn't have been");
    assert(Arena->TempCount > 0, "no temp memory registered for it to end");
    Arena->Used = TempMem.Used;
    --Arena->TempCount;
}

internal inline usize
gzMemAligmentOffset(mem_arena *Arena, usize Alignment) {
	usize AlignmentOffset = 0;
	usize ResultPointer = (usize)Arena->Ptr + Arena->Used;

	usize AlignmentMask = Alignment - 1;
	if(ResultPointer & AlignmentMask) {
		AlignmentOffset = Alignment - (ResultPointer & AlignmentMask);
	}

	return AlignmentOffset;
}

#define gzMemPushStruct(Arena, Type) (Type *)gzMemPushSize(Arena, sizeof(Type))
#define gzMemPushArray(Arena, Type, Count) (Type *)gzMemPushSize(Arena, (Count)*sizeof(Type))
internal void *
gzMemPushSize(mem_arena *Arena, usize Size) {
    assert(Arena->Used + Size < Arena->Size, "not enough arena memory");
    void *Result = (u8 *)Arena->Ptr + Arena->Used;
    Arena->Used += Size;

    return Result;
}
