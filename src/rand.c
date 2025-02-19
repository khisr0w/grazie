/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  8/1/2024 11:15:23 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

#include "rand.h"
/* NOTE(Abid): The following implementation has been derived from the Numerical Recipes.
 * Period = 10^12 */

global_var rand_state __gzGLOBALRandState = {
        .V = 4101842887655102017LL,
        .NumU8Reserves = 0,
        .NumU16Reserves = 0,
        .U8Reserves = 0,
        .U16Reserves = 0,
        .F64Reserve = 0,
        .IsInit = false,
};

/* NOTE(Abid): Override the RNG seed number. */
inline internal void
gzRandSeed(u64 Seed) {
    /* NOTE(Abid): Reserve bits for U8 and U16 routines. */

    __gzGLOBALRandState.V ^= Seed;
    /* NOTE(Abid): gzRandU64() routine here. */
    __gzGLOBALRandState.V ^= __gzGLOBALRandState.V >> 21;
    __gzGLOBALRandState.V ^= __gzGLOBALRandState.V << 35;
    __gzGLOBALRandState.V ^= __gzGLOBALRandState.V >> 4;
    __gzGLOBALRandState.V *= 2685821657736338717LL;
    __gzGLOBALRandState.IsInit = true;
}
/* NOTE(Abid): Initialize seed with an OS-dependent cyptographic sequence generator
 *             and feed it to the generator as the default seed. */
internal void
__gzPlatformInitRandSeed() {
    u64 Seed = 0;

#ifdef GRAZIE_PLT_WIN
    /* NOTE(Abid): Get the cryptograhic number generator method */
    BCRYPT_ALG_HANDLE CNGProvider;
    NTSTATUS Result = BCryptOpenAlgorithmProvider(&CNGProvider, BCRYPT_RNG_ALGORITHM, NULL, 0);
    assert(Result == ((NTSTATUS)0x00000000L), "could not init cryptographic algorithm provider");

    /* NOTE(Abid): Generate random Seed value. */
    BCryptGenRandom(CNGProvider, (PUCHAR)&Seed, 8, 0);

    /* NOTE(Abid): Close CNG Provider */
    BCryptCloseAlgorithmProvider(CNGProvider, 0);
#endif

#ifdef GRAZIE_PLT_LINUX
    /* TODO(Abid): Implement for linux. */
#endif
    gzRandSeed(Seed);
}

inline internal u64 
gzRandU64() {
    if(__gzGLOBALRandState.IsInit == false) __gzPlatformInitRandSeed();
    __gzGLOBALRandState.V ^= __gzGLOBALRandState.V >> 21;
    __gzGLOBALRandState.V ^= __gzGLOBALRandState.V << 35;
    __gzGLOBALRandState.V ^= __gzGLOBALRandState.V >> 4;
    return __gzGLOBALRandState.V * 2685821657736338717LL;
}
/* NOTE(Abid): Range in [Min, Max) */
/* TODO(Abid): Get rid of modulus in here (faster). */
inline internal u64
gzRandRangeU64(u64 Min, u64 Max) { return Min + gzRandU64() % (Max-Min); }
inline internal u32 gzRandU32() { return (u32)gzRandU64(); }
inline internal u16 gzRandU16() {
    if(__gzGLOBALRandState.NumU16Reserves--)
        return (u16)(__gzGLOBALRandState.U16Reserves >>= 16);
    __gzGLOBALRandState.U16Reserves = gzRandU64();
    __gzGLOBALRandState.NumU16Reserves = 3;

    return (u16)__gzGLOBALRandState.U16Reserves;
}
inline internal u8 gzRandU8() {
    if(__gzGLOBALRandState.NumU8Reserves--)
        return (u8)(__gzGLOBALRandState.U8Reserves >>= 8);
    __gzGLOBALRandState.U8Reserves = gzRandU64();
    __gzGLOBALRandState.NumU8Reserves = 7;

    return (u8)__gzGLOBALRandState.U8Reserves;
}
inline internal f64
gzRandF64() { return 5.42101086242752217e-20 * (f64)gzRandU64(); }

/* NOTE(Abid): Range [Min, Max] */
inline internal f64
gzRandRangeF64(f64 Min, f64 Max) { return Min + gzRandF64() * (Max - Min); }

inline internal f64
gzRandNormal(f64 Mean, f64 Std) {
    /* NOTE(Abid): Implementation of ratio-of-uniforms method by Leva
     * Paper: https://dl.acm.org/doi/pdf/10.1145/138351.138364 */
    f64 U, V, X, Y, Q;

    do {
        U = gzRandRangeF64(0, 1);
        V = 1.7156*(gzRandRangeF64(0, 1)-0.5);
        X = U - 0.449871;
        Y = fabs(V) + 0.386595;
        Q = (X*X) + Y*(0.19600*Y-0.25472*X);
    } while(Q > 0.27597 && (Q > 0.27846 || (V*V) > -4.*gz_log(U)*(U*U)));

    return Mean + Std * (V/U);
}


