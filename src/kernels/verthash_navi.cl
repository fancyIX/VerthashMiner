/*
 * Copyright 2021 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */


// AMD specific optimization for 64 bit rotate
#ifdef BIT_ALIGN
#define rotr64(x, n) ((n) < 32 ? (amd_bitalign((uint)((x) >> 32), (uint)(x), (uint)(n)) | ((ulong)amd_bitalign((uint)(x), (uint)((x) >> 32), (uint)(n)) << 32)) : (amd_bitalign((uint)(x), (uint)((x) >> 32), (uint)(n) - 32) | ((ulong)amd_bitalign((uint)((x) >> 32), (uint)(x), (uint)(n) - 32) << 32)))
#else
#define rotr64(x, n) rotate(x, (ulong)(64-n))
#endif

inline uint rotl32(uint x, uint n)
{
    return (((x) << (n)) | ((x) >> (32 - (n))));
}

inline uint fnv1a(const uint a, const uint b)
{
    uint res = (a ^ b) * 0x1000193U;
    return res;
}

// 2x precomputed SHA3 states
typedef union {
    ulong ul[50];
} kstate2x_t;

// shared hash to exchange between lanes during memory seeks stage
typedef union {
    uint2 u2[4];
} hash8_t;

// A combined SHA3 result used during memory seeks stage
typedef union {
    uint u[128];
    uint2 u2[64];
} sha3_state_t;

// Keccak constants
__constant ulong keccakf_rndc[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

// configured from host side(here just for unit testing)
//-----------------------------------------------------------------------------
// Kernel is configured from host side.

// Local work size. It can vary between hardware architectures.
//#define WORK_SIZE 64 // AMD
//#define WORK_SIZE 64 // NV

// Computed from the Verthash data file
//#define MDIV 71303125

// Extended validation uses 64 bit GPU side validation instead of 32 bit.
// It can be slightly more efficient with higher diff
//#define EXTENDED_VALIDATION

//! Full host side validation
//#define FULL_VALIDATION

__attribute__((reqd_work_group_size(WORK_SIZE, 1, 1)))
__kernel void verthash_4w(__global uint2* io_hashes,
                          __global kstate2x_t* restrict kStates,
                          __global uint2* restrict memory,
                          const uint in18,
                          const uint firstNonce
#ifdef FULL_VALIDATION
                          )
#else
                          , __global uint* targetResults,
    #ifdef EXTENDED_VALIDATION                          
                          const ulong target)
    #else // !EXTENDED_VALIDATION
                          const uint target)
    #endif
#endif // !FULL_VALIDATION
{
    // global id (1x work id)
    uint gid = get_global_id(0);
    // 4x lane group index(local)
    uint lgr4id = get_local_id(0) >> 2;
    // 4x lane group index(global) used as nonce result
    uint gr4id = gid >> 2;
    // sub group id(of 4x lane group)
    uint gr4e = gid & 3;

    //-----------------------------------------------------------------------------
    // SHA3 stage
    __global kstate2x_t* kstate = &kStates[gr4e];
    
    uint sha3St[32];
    uint nonce = firstNonce + gr4id;
    
    // 4 way kernel running 8xSHA3 passes(2x each lane)    
#define SHA_PASS(s3s) \
    { \
        ulong st[25] = { 0 }; \
        for(int i = 0; i < 25; ++i) \
        { \
            st[ i] = kstate->ul[25 * s3s + i]; \
        } \
        st[0] ^= as_ulong((uint2)(in18, nonce)); \
        st[1] ^= 0x06UL; \
        st[8] ^= 0x8000000000000000UL; \
        ulong u[5]; \
        ulong v,w; \
        for (int r = 0; r < 24; r++) \
        { \
            v    = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20]; \
            u[2] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21]; \
            u[3] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22]; \
            u[4] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23]; \
            w    = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24]; \
            u[0] = rotr64(u[2], 63) ^    w; \
            u[1] = rotr64(u[3], 63) ^    v; \
            u[2] = rotr64(u[4], 63) ^ u[2]; \
            u[3] = rotr64(   w, 63) ^ u[3]; \
            u[4] = rotr64(   v, 63) ^ u[4]; \
            st[0] ^= u[0]; st[5] ^= u[0]; st[10] ^= u[0]; st[15] ^= u[0]; st[20] ^= u[0]; \
            st[1] ^= u[1]; st[6] ^= u[1]; st[11] ^= u[1]; st[16] ^= u[1]; st[21] ^= u[1]; \
            st[2] ^= u[2]; st[7] ^= u[2]; st[12] ^= u[2]; st[17] ^= u[2]; st[22] ^= u[2]; \
            st[3] ^= u[3]; st[8] ^= u[3]; st[13] ^= u[3]; st[18] ^= u[3]; st[23] ^= u[3]; \
            st[4] ^= u[4]; st[9] ^= u[4]; st[14] ^= u[4]; st[19] ^= u[4]; st[24] ^= u[4]; \
            v = st[1]; \
            st[ 1] = rotr64(st[ 6], 20); \
            st[ 6] = rotr64(st[ 9], 44); \
            st[ 9] = rotr64(st[22],  3); \
            st[22] = rotr64(st[14], 25); \
            st[14] = rotr64(st[20], 46); \
            st[20] = rotr64(st[ 2],  2); \
            st[ 2] = rotr64(st[12], 21); \
            st[12] = rotr64(st[13], 39); \
            st[13] = rotr64(st[19], 56); \
            st[19] = rotr64(st[23],  8); \
            st[23] = rotr64(st[15], 23); \
            st[15] = rotr64(st[ 4], 37); \
            st[ 4] = rotr64(st[24], 50); \
            st[24] = rotr64(st[21], 62); \
            st[21] = rotr64(st[ 8],  9); \
            st[ 8] = rotr64(st[16], 19); \
            st[16] = rotr64(st[ 5], 28); \
            st[ 5] = rotr64(st[ 3], 36); \
            st[ 3] = rotr64(st[18], 43); \
            st[18] = rotr64(st[17], 49); \
            st[17] = rotr64(st[11], 54); \
            st[11] = rotr64(st[ 7], 58); \
            st[ 7] = rotr64(st[10], 61); \
            st[10] = rotr64(v, 63); \
            v = st[ 0]; w = st[ 1]; st[ 0] = bitselect(st[ 0] ^ st[ 2], st[ 0], st[ 1]); st[ 1] = bitselect(st[ 1] ^ st[ 3], st[ 1], st[ 2]); st[ 2] = bitselect(st[ 2] ^ st[ 4], st[ 2], st[ 3]); st[ 3] = bitselect(st[ 3] ^ v, st[ 3], st[ 4]); st[ 4] = bitselect(st[ 4] ^ w, st[ 4], v); \
            v = st[ 5]; w = st[ 6]; st[ 5] = bitselect(st[ 5] ^ st[ 7], st[ 5], st[ 6]); st[ 6] = bitselect(st[ 6] ^ st[ 8], st[ 6], st[ 7]); st[ 7] = bitselect(st[ 7] ^ st[ 9], st[ 7], st[ 8]); st[ 8] = bitselect(st[ 8] ^ v, st[ 8], st[ 9]); st[ 9] = bitselect(st[ 9] ^ w, st[ 9], v); \
            v = st[10]; w = st[11]; st[10] = bitselect(st[10] ^ st[12], st[10], st[11]); st[11] = bitselect(st[11] ^ st[13], st[11], st[12]); st[12] = bitselect(st[12] ^ st[14], st[12], st[13]); st[13] = bitselect(st[13] ^ v, st[13], st[14]); st[14] = bitselect(st[14] ^ w, st[14], v); \
            v = st[15]; w = st[16]; st[15] = bitselect(st[15] ^ st[17], st[15], st[16]); st[16] = bitselect(st[16] ^ st[18], st[16], st[17]); st[17] = bitselect(st[17] ^ st[19], st[17], st[18]); st[18] = bitselect(st[18] ^ v, st[18], st[19]); st[19] = bitselect(st[19] ^ w, st[19], v); \
            v = st[20]; w = st[21]; st[20] = bitselect(st[20] ^ st[22], st[20], st[21]); st[21] = bitselect(st[21] ^ st[23], st[21], st[22]); st[22] = bitselect(st[22] ^ st[24], st[22], st[23]); st[23] = bitselect(st[23] ^ v, st[23], st[24]); st[24] = bitselect(st[24] ^ w, st[24], v); \
            st[0] ^= keccakf_rndc[r]; \
        } \
        ((uint2 *)sha3St)[8 * s3s + 0] = as_uint2(st[0]); \
        ((uint2 *)sha3St)[8 * s3s + 1] = as_uint2(st[1]); \
        ((uint2 *)sha3St)[8 * s3s + 2] = as_uint2(st[2]); \
        ((uint2 *)sha3St)[8 * s3s + 3] = as_uint2(st[3]); \
        ((uint2 *)sha3St)[8 * s3s + 4] = as_uint2(st[4]); \
        ((uint2 *)sha3St)[8 * s3s + 5] = as_uint2(st[5]); \
        ((uint2 *)sha3St)[8 * s3s + 6] = as_uint2(st[6]); \
        ((uint2 *)sha3St)[8 * s3s + 7] = as_uint2(st[7]); \
    }

    SHA_PASS(0)
    SHA_PASS(1)

    //-----------------------------------------------------------------------------
    // Verthash IO memory seek stage
    
    // get SHA3 256 input
    uint2 up1;
    up1 = io_hashes[gid];
    
    uint value_accumulator = 0x811c9dc5;

    // reference computed on the host side.
    //const uint mdiv = ((datfile_sz - HASH_OUT_SIZE)/BYTE_ALIGNMENT) + 1;
    
    // 71303125 is by default, but can change in future releases
    const uint mdiv = MDIV;
    
    for(uint i = 0; i < 4096 / 128; ++i)
    {
        // generate seek indexes at runtime
        // After each iteration SHA3 combined state requires a bit rotate operation
        // Note that some hardware don't support bit rotate by dynamic amount in a single instruction
        // v1 uses Load -> rotate by 1 -> store
        // v2 uses Load -> rotate by dynamic factor(depends on iteration)
        
        // v1. Rotate by constant amount
        
#define GEN_SEEK(s3idx0) \
        { \
            uint seek_index; \
            uint blk = s3idx0 / 32; \
            uint ind = s3idx0 % 32; \
            if (blk == 0) { \
                __asm ( \
                "s_nop 0\n" \
                "v_mov_b32_dpp  %[d], %[s] quad_perm:[0,0,0,0]\n" \
                "s_nop 0" \
                : [d] "=v" (seek_index) \
                : [s] "v" (sha3St[ind])); \
            } \
            if (blk == 1) { \
                __asm ( \
                "s_nop 0\n" \
                "v_mov_b32_dpp  %[d], %[s] quad_perm:[1,1,1,1]\n" \
                "s_nop 0" \
                : [d] "=v" (seek_index) \
                : [s] "v" (sha3St[ind])); \
            } \
            if (blk == 2) { \
                __asm ( \
                "s_nop 0\n" \
                "v_mov_b32_dpp  %[d], %[s] quad_perm:[2,2,2,2]\n" \
                "s_nop 0" \
                : [d] "=v" (seek_index) \
                : [s] "v" (sha3St[ind])); \
            } \
            if (blk == 3) { \
                __asm ( \
                "s_nop 0\n" \
                "v_mov_b32_dpp  %[d], %[s] quad_perm:[3,3,3,3]\n" \
                "s_nop 0" \
                : [d] "=v" (seek_index) \
                : [s] "v" (sha3St[ind])); \
            } \
            uint state0mod = rotl32(seek_index, 1); \
            if (blk == gr4e) { \
                sha3St[ind] = state0mod; \
            } \
            const uint offset = (fnv1a(seek_index, value_accumulator) % mdiv) << 1; \
            const uint2 vvalue = memory[offset + gr4e]; \
            up1.x = fnv1a(up1.x, vvalue.x); \
            up1.y = fnv1a(up1.y, vvalue.y); \
            uint2 uu0; \
            uint2 uu1; \
            uint2 uu2; \
            uint2 uu3; \
            __asm ( \
            "s_nop 0\n" \
            "s_nop 0\n" \
            "v_mov_b32_dpp  %[d0x], %[vx] quad_perm:[0,0,0,0]\n" \
            "v_mov_b32_dpp  %[d0y], %[vy] quad_perm:[0,0,0,0]\n" \
            "v_mov_b32_dpp  %[d1x], %[vx] quad_perm:[1,1,1,1]\n" \
            "v_mov_b32_dpp  %[d1y], %[vy] quad_perm:[1,1,1,1]\n" \
            "v_mov_b32_dpp  %[d2x], %[vx] quad_perm:[2,2,2,2]\n" \
            "v_mov_b32_dpp  %[d2y], %[vy] quad_perm:[2,2,2,2]\n" \
            "v_mov_b32_dpp  %[d3x], %[vx] quad_perm:[3,3,3,3]\n" \
            "v_mov_b32_dpp  %[d3y], %[vy] quad_perm:[3,3,3,3]\n" \
            "s_nop 0\n" \
            "s_nop 0" \
            : [d0x] "=v" (uu0.x), \
                [d0y] "=v" (uu0.y), \
                [d1x] "=v" (uu1.x), \
                [d1y] "=v" (uu1.y), \
                [d2x] "=v" (uu2.x), \
                [d2y] "=v" (uu2.y), \
                [d3x] "=v" (uu3.x), \
                [d3y] "=v" (uu3.y) \
            : [vx] "v" (vvalue.x), \
                [vy] "v" (vvalue.y)); \
            value_accumulator = fnv1a(value_accumulator, uu0.x); \
            value_accumulator = fnv1a(value_accumulator, uu0.y); \
            value_accumulator = fnv1a(value_accumulator, uu1.x); \
            value_accumulator = fnv1a(value_accumulator, uu1.y); \
            value_accumulator = fnv1a(value_accumulator, uu2.x); \
            value_accumulator = fnv1a(value_accumulator, uu2.y); \
            value_accumulator = fnv1a(value_accumulator, uu3.x); \
            value_accumulator = fnv1a(value_accumulator, uu3.y); \
        }

        GEN_SEEK(0)
        GEN_SEEK(1)
        GEN_SEEK(2)
        GEN_SEEK(3)
        GEN_SEEK(4)
        GEN_SEEK(5)
        GEN_SEEK(6)
        GEN_SEEK(7)
        GEN_SEEK(8)
        GEN_SEEK(9)
        GEN_SEEK(10)
        GEN_SEEK(11)
        GEN_SEEK(12)
        GEN_SEEK(13)
        GEN_SEEK(14)
        GEN_SEEK(15)
        GEN_SEEK(16)
        GEN_SEEK(17)
        GEN_SEEK(18)
        GEN_SEEK(19)
        GEN_SEEK(20)
        GEN_SEEK(21)
        GEN_SEEK(22)
        GEN_SEEK(23)
        GEN_SEEK(24)
        GEN_SEEK(25)
        GEN_SEEK(26)
        GEN_SEEK(27)
        GEN_SEEK(28)
        GEN_SEEK(29)
        GEN_SEEK(30)
        GEN_SEEK(31)
        GEN_SEEK(32)
        GEN_SEEK(33)
        GEN_SEEK(34)
        GEN_SEEK(35)
        GEN_SEEK(36)
        GEN_SEEK(37)
        GEN_SEEK(38)
        GEN_SEEK(39)
        GEN_SEEK(40)
        GEN_SEEK(41)
        GEN_SEEK(42)
        GEN_SEEK(43)
        GEN_SEEK(44)
        GEN_SEEK(45)
        GEN_SEEK(46)
        GEN_SEEK(47)
        GEN_SEEK(48)
        GEN_SEEK(49)
        GEN_SEEK(50)
        GEN_SEEK(51)
        GEN_SEEK(52)
        GEN_SEEK(53)
        GEN_SEEK(54)
        GEN_SEEK(55)
        GEN_SEEK(56)
        GEN_SEEK(57)
        GEN_SEEK(58)
        GEN_SEEK(59)
        GEN_SEEK(60)
        GEN_SEEK(61)
        GEN_SEEK(62)
        GEN_SEEK(63)
        GEN_SEEK(64)
        GEN_SEEK(65)
        GEN_SEEK(66)
        GEN_SEEK(67)
        GEN_SEEK(68)
        GEN_SEEK(69)
        GEN_SEEK(70)
        GEN_SEEK(71)
        GEN_SEEK(72)
        GEN_SEEK(73)
        GEN_SEEK(74)
        GEN_SEEK(75)
        GEN_SEEK(76)
        GEN_SEEK(77)
        GEN_SEEK(78)
        GEN_SEEK(79)
        GEN_SEEK(80)
        GEN_SEEK(81)
        GEN_SEEK(82)
        GEN_SEEK(83)
        GEN_SEEK(84)
        GEN_SEEK(85)
        GEN_SEEK(86)
        GEN_SEEK(87)
        GEN_SEEK(88)
        GEN_SEEK(89)
        GEN_SEEK(90)
        GEN_SEEK(91)
        GEN_SEEK(92)
        GEN_SEEK(93)
        GEN_SEEK(94)
        GEN_SEEK(95)
        GEN_SEEK(96)
        GEN_SEEK(97)
        GEN_SEEK(98)
        GEN_SEEK(99)
        GEN_SEEK(100)
        GEN_SEEK(101)
        GEN_SEEK(102)
        GEN_SEEK(103)
        GEN_SEEK(104)
        GEN_SEEK(105)
        GEN_SEEK(106)
        GEN_SEEK(107)
        GEN_SEEK(108)
        GEN_SEEK(109)
        GEN_SEEK(110)
        GEN_SEEK(111)
        GEN_SEEK(112)
        GEN_SEEK(113)
        GEN_SEEK(114)
        GEN_SEEK(115)
        GEN_SEEK(116)
        GEN_SEEK(117)
        GEN_SEEK(118)
        GEN_SEEK(119)
        GEN_SEEK(120)
        GEN_SEEK(121)
        GEN_SEEK(122)
        GEN_SEEK(123)
        GEN_SEEK(124)
        GEN_SEEK(125)
        GEN_SEEK(126)
        GEN_SEEK(127)
    }
    
    // store result
    io_hashes[gid] = up1;

#ifndef FULL_VALIDATION
    //---------------------------------------------------
    // Save result as HTarg
    if(gr4e == 3)
    {
#ifdef EXTENDED_VALIDATION
        ulong up1_64 = as_ulong(up1);
        if(up1_64 <= target)
#else
        if(up1.y <= target)
#endif
        {
            uint ai = atomic_inc(targetResults);
            targetResults[ai+1] = gr4id; // final nonce
        }
    }
#endif // !FULL_VALIDATION

    barrier(CLK_GLOBAL_MEM_FENCE);
}
