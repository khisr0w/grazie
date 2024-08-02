/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  8/1/2024 11:16:31 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

#if !defined(RAND_H)

typedef struct {
    u64 V;

    i32 NumU8Reserves;
    i32 NumU16Reserves;
    u64 U8Reserves;
    u64 U16Reserves;
    f64 F64Reserve;
    bool IsInit;
} rand_state;

#define RAND_H
#endif
