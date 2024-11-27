/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/16/2023 6:25:57 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#if !defined(UTILS_H)

/* TODO(Abid): Add a release flag to remove all the asserts. */

#ifdef GRAZIE_DEBUG
#endif

#ifdef GRAZIE_ASSERT
#define assert(Expr, ErrorStr, ...) \
    if((Expr)) { } \
    else { \
        fprintf(stderr, "ASSERTION ERROR (%s:%d): " ErrorStr "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__); \
        *(i32 *)0 = 0; \
    }
#else
#define assert(Expr, ErrorStr, ...) \
    if((Expr)) { } \
    else { \
        fprintf(stderr, "ASSERTION ERROR (%s:%d): " ErrorStr "\nExiting...\n", \
                __FILE__, __LINE__, ##__VA_ARGS__); \
        exit(EXIT_FAILURE); \
    }
#endif



#define gz_array_length(Array) (sizeof(Array)/sizeof(Array[0]))

/* NOTE(Abid): Byte Macros */
#define gzKilobyte(Value) ((Value)*1024LL)
#define gzMegabyte(Value) (gzKilobyte(Value)*1024LL)
#define gzGigabyte(Value) (gzMegabyte(Value)*1024LL)
#define gzTerabyte(Value) (gzGigabyte(Value)*1024LL)

/* NOTE(Abid): Get Stride and Sizes, WARNING(Abid): The indexing starts from the right side */
#define GetStrideR(A, IDX) ((A)->Header->Strides[(A)->Header->Dim-(IDX)-1])
#define GetSizeR(A, IDX) ((A)->Header->Sizes[(A)->Header->Dim-(IDX)-1])

/* NOTE(Abid): typedef and static define for ease of use */
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef int64_t i64;
typedef int32_t i32;
typedef int8_t i8;
typedef double f64;
typedef float f32;
typedef uintptr_t uintptr;
typedef int8_t bool;
typedef size_t usize;
#define true 1
#define false 0
#define internal static
#define local_persist static
#define global_var static

/* NOTE(Abid): Defines if op history should be preserved for gradient calculation */
#define GRAD_PRESERVE(Value) __GetSetGradState(true, Value)
#define GRAD_PRESERVE_TOGGLE() __GetSetGradState(true, !IS_GRAD_PRESERVE())
#define IS_GRAD_PRESERVE() (__GetSetGradState(false, 0))

inline internal bool
__GetSetGradState(bool Set, bool NewState) {
    local_persist bool ShouldGrad = true;

    if(Set) ShouldGrad = NewState;

    return ShouldGrad;
}

typedef struct {
    f64 Latest;
    f64 Sum;
    f64 SumSquared;
    u64 Count;
    f64 Max;
    f64 Min;
} f64_stat;
inline internal void
gzStatAccumulate(f64 Value, f64_stat *Stat) {
    if(Stat->Count == 0) {
        Stat->Max = Value;
        Stat->Min = Value;
    }

    Stat->Latest = Value;
    Stat->SumSquared += Value*Value;
    Stat->Sum += Value;
    ++Stat->Count;
}

inline internal f64
gzStatMean(f64_stat *Stat) {
    assert(Stat->Count > 0, "cannot calculate mean for count < 1");
    return Stat->Sum / Stat->Count;
}

inline internal f64
gzStatVar(f64_stat *Stat) {
    assert(Stat->Count > 0, "Cannot calculate variance for count < 1.");
    f64 StatMean = gzStatMean(Stat);
    return (Stat->SumSquared - 2 * StatMean * (Stat->Sum) + Stat->Count*(StatMean*StatMean)) / fmax(Stat->Count - 1, 1);
}

inline internal f64
gzStatStd(f64_stat *Stat) {
    assert(Stat->Count > 0, "Cannot calculate std for count < 1.");
    return sqrt(gzStatVar(Stat));
}

#define UTILS_H
#endif
