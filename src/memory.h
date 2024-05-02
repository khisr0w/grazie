/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  Mo 20 Mär 2023 20:58:37 CET                                   |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#if !defined(MEMORY_H)

typedef struct {
    usize Used;
    usize MaxSize;
    void *Ptr;
    
    u32 TempCount;
} mem_arena;

typedef struct {
    mem_arena *Arena;
    usize Used;
} temp_memory;

#define MEMORY_H
#endif
