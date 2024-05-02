/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  Mo 20 Mär 2023 20:56:15 CET                                  |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#include "memory.h"

/* TODO(Abid): The custum allocators should be defined here */
#define Free(ptr) free(ptr)
#define Malloc(ptr) malloc(ptr)
#define _Calloc(ptr, size) calloc(ptr, size)
#define Realloc(ptr, size) realloc(ptr, size)

#ifdef GRAZIE_PLT_WIN
#define PlatformAlloc(Size) Assert(0, "Implement windows allocation here")
#endif 
#ifdef GRAZIE_PLT_WIN

#define PlatformAllocte(Size) memset
#endif 

inline internal void *
PlatoformAllocate(usize Size) {
    void *Result = NULL;

#ifdef GRAZIE_PLT_WIN
    Assert(0, "Implement windows allocation here")
#endif 

#ifdef GRAZIE_PLT_LINUX
    Result = mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if(Result == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }
#endif 

    return Result;
}

internal mem_arena
AllocateArena(usize BytesToAllocate) {
    mem_arena Arena = {0};
    Arena.Ptr = PlatoformAllocate(Arena.MaxSize);
    Arena.Used = 0;
    Arena.MaxSize = BytesToAllocate;

    return Arena;
}

inline internal temp_memory
BeginTempMemory(mem_arena *Arena) {
    temp_memory Result = {0};

    Result.Arena = Arena;
    Result.Used = Arena->Used;

    ++Arena->TempCount;

    return Result;
}

inline internal void
EndTempMemory(temp_memory TempMem) {
    mem_arena *Arena = TempMem.Arena;
    Assert(Arena->Used >= TempMem.Used, "something was freed when it shouldn't have been");
    Arena->Used = TempMem.Used;
    Assert(Arena->TempCount > 0, "no temp memory registered for it to end");
    --Arena->TempCount;
}

internal inline usize
AligmentOffset(mem_arena *Arena, usize Alignment) {
	usize AlignmentOffset = 0;
	usize ResultPointer = (usize)Arena->Ptr + Arena->Used;

	usize AlignmentMask = Alignment - 1;
	if(ResultPointer & AlignmentMask) {
		AlignmentOffset = Alignment - (ResultPointer & AlignmentMask);
	}

	return AlignmentOffset;
}

#define PushStruct(Arena, Type) (Type *)PushSize(Arena, sizeof(Type))
#define PushArray(Arena, Type, Count) (Type *)PushSize(Arena, (Count)*sizeof(Type))
internal void *
PushSize(mem_arena *Arena, usize Size) {
    Assert(Arena->Used + Size < Arena->MaxSize, "not enough arena memory");
    void *Result = Arena->Ptr + Arena->Used;
    Arena->Used += Size;

    return Result;
}

