/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/16/2023 6:25:57 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#if !defined(UTILS_H)

#ifdef GRAZIE_ASSERT
#define Assert(Expr, ErrorStr) if(!(Expr)) {fprintf(stderr, "ASSERTION ERROR (%s:%d): " ErrorStr "\n", __FILE__, __LINE__); *(i32 *)0 = 0; }
#else
#define Assert(Expr, ErrorStr) if(!(Expr)) {fprintf(stderr, "ASSERTION ERROR (%s:%d): " ErrorStr "\nExiting...\n", __FILE__, __LINE__); exit(-1);}
#endif

#ifdef GRAZIE_DEBUG
#define InvalidCodePath *(i32 *)0 = 0
#else
#define InvalidCodePath do { fprintf(stderr, "Invalid Path (%s:%d): \nExiting...\n", __FILE__, __LINE__); exit(-1); } while(0)
#endif


#define ArrayLength(Array) (sizeof(Array)/sizeof(Array[0]))

/* NOTE(Abid): Byte Macros */
#define Kilobyte(Value) ((Value)*1024LL)
#define Megabyte(Value) (Kilobyte(Value)*1024LL)
#define Gigabyte(Value) (Megabyte(Value)*1024LL)
#define Terabyte(Value) (Gigabyte(Value)*1024LL)

/* NOTE(Abid): Get Stride and Sizes, WARNING(Abid): The indexing starts from the right side */
#define GetStrideR(A, IDX) ((A)->Header->Strides[(A)->Header->Dim-(IDX)-1])
#define GetSizeR(A, IDX) ((A)->Header->Sizes[(A)->Header->Dim-(IDX)-1])

/* NOTE(Abid): typedef and static define for ease of use */
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t i32;
typedef int64_t i64;
typedef float f32;
typedef double f64;
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
internal inline bool
__GetSetGradState(bool Set, bool NewState) {
    local_persist bool ShouldGrad = true;

    if(Set) ShouldGrad = NewState;

    return ShouldGrad;
}

internal inline u32
__PlatformRandom() {
    u32 Number;
#if GRAZIE_PLT_WIN
    /* TODO(Abid): The routine returns the status of the generation process and must be asserted.
     *             Link: https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/rand-s?view=msvc-170 */
    rand_s(&Number);
#endif

#if PLT_LINUX
    /* TODO(Abid): Use https://pubs.opengroup.org/onlinepubs/007908799/xsh/drand48.html */
    Assert(0, "not yet implemented");
#endif

    return Number;
}

/* NOTE(Abid): Following will return values from `From` up to, and excluding, `Until`. */
internal inline i32
RandUniformInt32(i32 From, i32 Until) {
    Assert(From <= Until, "starting interval cannot be larger than ending interval");

    i32 Diff = Until - From;
    double RandDouble = (double)__PlatformRandom() / ((double)UINT_MAX + 1);
    return From + (i32)(RandDouble * Diff);
}

internal inline f32
RandUniformFloat32(f32 From, f32 Until) {
    Assert(From <= Until, "starting interval cannot be larger than ending interval");

    f32 Diff = Until - From;
    double RandDouble = (double)__PlatformRandom() / ((double)UINT_MAX + 1);
    return From + (f32)(RandDouble * Diff);
}

#define UTILS_H
#endif
