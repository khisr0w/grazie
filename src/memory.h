/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  Mo 20 Mär 2023 20:58:37 CET                                  |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#if !defined(MEMORY_H)

typedef struct
{
    // NOTE(Abid): Memory
    int8_t *MemPtr;
    size_t MemMaxByteSize;
    size_t MemByteUsed;

} grazie_state;



#define MEMORY_H
#endif
