
/**
 *  SangNom
 *
 *  port from AVISynth SangNom2
 *
 *  Original Author: Victor Efimov
 *
 *  Copyright (c) 2013 Victor Efimov
 *  This project is licensed under the MIT license. Binaries are GPL v2.
 *
 **/

#include <VapourSynth.h>
#include <VSHelper.h>

#include <cstdint>
#include <algorithm>
#include <string>
#include <smmintrin.h>

static const size_t alignment = 32;

typedef struct SangNomData
{
    VSNodeRef *node;
    const VSVideoInfo *vi;

    int order;
    int aa;
    bool planes[3];

    float aaf;  // float type of aa param
    int offset;
    int bufferStride;
    int bufferHeight;
} SangNomData;

enum SangNomOrderType
{
    SNOT_DFR = 0,   // double frame rate, user must call DoubleWeave() before use this
    SNOT_SFR_KT,    // single frame rate, keep top field
    SNOT_SFR_KB     // single frame rate, keep bottom field
};

static inline int VapourSynthFieldBasedToSangNomOrder(int fieldbased)
{
    if (fieldbased == 2) // tff
        return SNOT_SFR_KT;
    if (fieldbased == 1) // bff
        return SNOT_SFR_KB;
    return SNOT_SFR_KT;
}

template <typename T, bool aligned>
static inline __m128i simd_load_si128(const T *ptr) {
    if (aligned)
        return _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
}

template <typename T, bool aligned>
static inline void simd_store_si128(T *ptr, __m128i val) {
    if (aligned)
        _mm_store_si128(reinterpret_cast<__m128i*>(ptr), val);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), val);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i simd_load_one_to_left(const T *ptr) {

    static_assert((std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)) ||
                  (std::is_floating_point<T>::value && sizeof(T) == 4),
                  "error instantiation, T must be 8, 16, 32 bit int or 32 bit float");

    if (isBorder) {

        if (std::is_integral<T>::value) {
            if (sizeof(T) == 1) {
                auto mask = _mm_setr_epi8(0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_slli_si128(val, 1);
                auto andm = _mm_and_si128(val, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 2) {
                auto mask = _mm_setr_epi16(0xFFFF, 00, 00, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_slli_si128(val, 2);
                auto andm = _mm_and_si128(val, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 4) {
                auto mask = _mm_setr_epi32(0xFFFFFFFF, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_slli_si128(val, 4);
                auto andm = _mm_and_si128(val, mask);
                return _mm_or_si128(shifted, andm);
            }
        } else {
            if (sizeof(T) == 4) {
                // TODO float
            }
        }
    }

    return simd_load_si128<T, false>(ptr - 1);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i simd_load_two_to_left(const T *ptr) {

    static_assert((std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)) ||
                  (std::is_floating_point<T>::value && sizeof(T) == 4),
                  "error instantiation, T must be 8, 16, 32 bit int or 32 bit float");

    if (isBorder) {

        if (std::is_integral<T>::value) {
            if (sizeof(T) == 1) {
                auto mask = _mm_setr_epi8(0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_slli_si128(val, 2);
                auto unpck = _mm_unpacklo_epi8(val, val);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 2) {
                auto mask = _mm_setr_epi16(0xFFFF, 0xFFFF, 00, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_slli_si128(val, 4);
                auto unpck = _mm_unpacklo_epi16(val, val);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 4) {
                auto mask = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_slli_si128(val, 8);
                auto unpck = _mm_unpacklo_epi32(val, val);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            }
        } else {
            if (sizeof(T) == 4) {
                // TODO float
            }
        }
    }

    return simd_load_si128<T, false>(ptr - 2);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i simd_load_three_to_left(const T *ptr) {

    static_assert((std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)) ||
                  (std::is_floating_point<T>::value && sizeof(T) == 4),
                  "error instantiation, T must be 8, 16, 32 bit int or 32 bit float");

    if (isBorder) {

        if (std::is_integral<T>::value) {
            if (sizeof(T) == 1) {
                auto mask = _mm_setr_epi8(0xFF, 0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_slli_si128(val, 3);
                auto unpck = _mm_unpacklo_epi8(val, val);
                unpck = _mm_unpacklo_epi16(unpck, unpck);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 2) {
                auto mask = _mm_setr_epi16(0xFFFF, 0xFFFF, 0xFFFF, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_slli_si128(val, 6);
                auto unpck = _mm_unpacklo_epi16(val, val);
                unpck = _mm_unpacklo_epi32(unpck, unpck);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 4) {
                auto mask = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_slli_si128(val, 12);
                auto unpck = _mm_unpacklo_epi32(val, val);
                unpck = _mm_unpacklo_epi64(unpck, unpck);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            }
        } else {
            if (sizeof(T) == 4) {
                // TODO float
            }
        }
    }

    return simd_load_si128<T, false>(ptr - 3);
}

//note the difference between set and setr for left and right loading
template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i simd_load_one_to_right(const T *ptr) {

    static_assert((std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)) ||
                  (std::is_floating_point<T>::value && sizeof(T) == 4),
                  "error instantiation, T must be 8, 16, 32 bit int or 32 bit float");

    if (isBorder) {

        if (std::is_integral<T>::value) {
            if (sizeof(T) == 1) {
                auto mask = _mm_set_epi8(0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_srli_si128(val, 1);
                auto andm = _mm_and_si128(val, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 2) {
                auto mask = _mm_set_epi16(0xFFFF, 00, 00, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_srli_si128(val, 2);
                auto andm = _mm_and_si128(val, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 4) {
                auto mask = _mm_set_epi32(0xFFFFFFFF, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_srli_si128(val, 4);
                auto andm = _mm_and_si128(val, mask);
                return _mm_or_si128(shifted, andm);
            }
        } else {
            if (sizeof(T) == 4) {
                // TODO float
            }
        }
    }

    return simd_load_si128<T, false>(ptr + 1);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i simd_load_two_to_right(const T *ptr) {

    static_assert((std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)) ||
                  (std::is_floating_point<T>::value && sizeof(T) == 4),
                  "error instantiation, T must be 8, 16, 32 bit int or 32 bit float");

    if (isBorder) {

        if (std::is_integral<T>::value) {
            if (sizeof(T) == 1) {
                auto mask = _mm_set_epi8(0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_srli_si128(val, 2);
                auto unpck = _mm_unpackhi_epi8(val, val);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 2) {
                auto mask = _mm_set_epi16(0xFFFF, 0xFFFF, 00, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_srli_si128(val, 4);
                auto unpck = _mm_unpackhi_epi16(val, val);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 4) {
                auto mask = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_srli_si128(val, 8);
                auto unpck = _mm_unpackhi_epi32(val, val);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            }
        } else {
            if (sizeof(T) == 4) {
                // TODO float
            }
        }
    }

    return simd_load_si128<T, false>(ptr + 2);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i simd_load_three_to_right(const T *ptr) {

    static_assert((std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4)) ||
                  (std::is_floating_point<T>::value && sizeof(T) == 4),
                  "error instantiation, T must be 8, 16, 32 bit int or 32 bit float");

    if (isBorder) {

        if (std::is_integral<T>::value) {
            if (sizeof(T) == 1) {
                auto mask = _mm_set_epi8(0xFF, 0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_srli_si128(val, 3);
                auto unpck = _mm_unpackhi_epi8(val, val);
                unpck = _mm_unpackhi_epi16(unpck, unpck);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 2) {
                auto mask = _mm_set_epi16(0xFFFF, 0xFFFF, 0xFFFF, 00, 00, 00, 00, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_srli_si128(val, 6);
                auto unpck = _mm_unpackhi_epi16(val, val);
                unpck = _mm_unpackhi_epi32(unpck, unpck);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            } else if (sizeof(T) == 4) {
                auto mask = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 00);
                auto val = simd_load_si128<T, alignedLoad>(ptr);
                auto shifted = _mm_srli_si128(val, 12);
                auto unpck = _mm_unpackhi_epi32(val, val);
                unpck = _mm_unpackhi_epi64(unpck, unpck);
                auto andm = _mm_and_si128(unpck, mask);
                return _mm_or_si128(shifted, andm);
            }
        } else {
            if (sizeof(T) == 4) {
                // TODO float
            }
        }
    }

    return simd_load_si128<T, false>(ptr + 3);
}


enum Buffers {
    ADIFF_M3_P3 = 0,
    ADIFF_M2_P2 = 1,
    ADIFF_M1_P1 = 2,
    ADIFF_P0_M0 = 4,
    ADIFF_P1_M1 = 6,
    ADIFF_P2_M2 = 7,
    ADIFF_P3_M3 = 8,

    SG_FORWARD = 3,
    SG_REVERSE = 5
};

enum class BorderMode {
    LEFT,
    RIGHT,
    NONE
};

const int BUFFERS_COUNT = 9;

template <typename T>
static inline __m128i simd_abs_diff_epu(__m128i a, __m128i b);

template <>
inline __m128i simd_abs_diff_epu<uint8_t>(__m128i a, __m128i b) {
    auto positive = _mm_subs_epu8(a, b);
    auto negative = _mm_subs_epu8(b, a);
    return _mm_or_si128(positive, negative);
}

template <>
inline __m128i simd_abs_diff_epu<uint16_t>(__m128i a, __m128i b) {
    auto positive = _mm_subs_epu16(a, b);
    auto negative = _mm_subs_epu16(b, a);
    return _mm_or_si128(positive, negative);
}

template <typename T>
inline __m128i calculateSangnom(const __m128i& p1, const __m128i& p2, const __m128i& p3);

template <>
inline __m128i calculateSangnom<uint8_t>(const __m128i& p1, const __m128i& p2, const __m128i& p3) {

    auto zero = _mm_setzero_si128();

    auto temp_lo = _mm_unpacklo_epi8(p1, zero);
    auto temp_hi = _mm_unpackhi_epi8(p1, zero);

    temp_lo = _mm_slli_epi16(temp_lo, 2); //p1*4
    temp_hi = _mm_slli_epi16(temp_hi, 2);

    auto t2_lo = _mm_unpacklo_epi8(p2, zero);
    auto t2_hi = _mm_unpackhi_epi8(p2, zero);

    temp_lo = _mm_adds_epu16(temp_lo, t2_lo); //p1*4 + p2
    temp_hi = _mm_adds_epu16(temp_hi, t2_hi);

    t2_lo = _mm_slli_epi16(t2_lo, 2);
    t2_hi = _mm_slli_epi16(t2_hi, 2);

    temp_lo = _mm_adds_epu16(temp_lo, t2_lo); //p1*4 + p2*4 + p2 = p1*4 + p2*5
    temp_hi = _mm_adds_epu16(temp_hi, t2_hi);

    auto t3_lo = _mm_unpacklo_epi8(p3, zero);
    auto t3_hi = _mm_unpackhi_epi8(p3, zero);

    temp_lo = _mm_subs_epu16(temp_lo, t3_lo); //p1*4 + p2*5 - p3
    temp_hi = _mm_subs_epu16(temp_hi, t3_hi);

    temp_lo = _mm_srli_epi16(temp_lo, 3); //(p1*4 + p2*5 - p3) / 8
    temp_hi = _mm_srli_epi16(temp_hi, 3);

    return _mm_packus_epi16(temp_lo, temp_hi); //(p1*4 + p2*5 - p3) / 8
}

// NOTE: lack of _mm_adds_epu32 & _mm_subs_epu32
template <>
inline __m128i calculateSangnom<uint16_t>(const __m128i& p1, const __m128i& p2, const __m128i& p3) {

    auto zero = _mm_setzero_si128();

    auto temp_lo = _mm_unpacklo_epi16(p1, zero);
    auto temp_hi = _mm_unpackhi_epi16(p1, zero);

    temp_lo = _mm_slli_epi32(temp_lo, 2); //p1*4
    temp_hi = _mm_slli_epi32(temp_hi, 2);

    auto t2_lo = _mm_unpacklo_epi16(p2, zero);
    auto t2_hi = _mm_unpackhi_epi16(p2, zero);

    temp_lo = _mm_add_epi32(temp_lo, t2_lo); //p1*4 + p2
    temp_hi = _mm_add_epi32(temp_hi, t2_hi);

    t2_lo = _mm_slli_epi32(t2_lo, 2);
    t2_hi = _mm_slli_epi32(t2_hi, 2);

    temp_lo = _mm_add_epi32(temp_lo, t2_lo); //p1*4 + p2*4 + p2 = p1*4 + p2*5
    temp_hi = _mm_add_epi32(temp_hi, t2_hi);

    auto t3_lo = _mm_unpacklo_epi16(p3, zero);
    auto t3_hi = _mm_unpackhi_epi16(p3, zero);

    temp_lo = _mm_sub_epi32(temp_lo, t3_lo); //p1*4 + p2*5 - p3
    temp_hi = _mm_sub_epi32(temp_hi, t3_hi);

    temp_lo = _mm_max_epi32(temp_lo, zero); // NOTE: Prevent negative value
    temp_hi = _mm_max_epi32(temp_hi, zero);

    temp_lo = _mm_srli_epi32(temp_lo, 3); //(p1*4 + p2*5 - p3) / 8
    temp_hi = _mm_srli_epi32(temp_hi, 3);

    return _mm_packus_epi32(temp_lo, temp_hi); //(p1*4 + p2*5 - p3) / 8
}

template <typename T>
static inline __m128i blendAvgOnMinimalBuffer(const __m128i& a1, const __m128i& a2, const __m128i& buf, const __m128i& minv, const __m128i& acc, const __m128i& zero);

template <>
inline __m128i blendAvgOnMinimalBuffer<uint8_t>(const __m128i& a1, const __m128i& a2, const __m128i& buf, const __m128i& minv, const __m128i& acc, const __m128i& zero) {
    auto average = _mm_avg_epu8(a1, a2);
    //buffer is minimal
    auto mask = _mm_cmpeq_epi8(buf, minv);
    //blend
    auto avgPart = _mm_and_si128(mask, average);
    auto accPart = _mm_andnot_si128(mask, acc);
    return _mm_or_si128(avgPart, accPart);
}

template <>
inline __m128i blendAvgOnMinimalBuffer<uint16_t>(const __m128i& a1, const __m128i& a2, const __m128i& buf, const __m128i& minv, const __m128i& acc, const __m128i& zero) {
    auto average = _mm_avg_epu16(a1, a2);
    //buffer is minimal
    auto mask = _mm_cmpeq_epi16(buf, minv);
    //blend
    auto avgPart = _mm_and_si128(mask, average);
    auto accPart = _mm_andnot_si128(mask, acc);
    return _mm_or_si128(avgPart, accPart);
}

template <typename T, BorderMode border, bool alignedLoad, bool alignedStore>
static inline void prepareBuffersLine(const T* srcp, const T *srcpn2, T* buffers[BUFFERS_COUNT], int bufferOffset, int width) {

    for (int x = 0; x < width; x += (16 / sizeof(T))) {

        auto cur_minus_3   = simd_load_three_to_left<T, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto cur_minus_2   = simd_load_two_to_left<T, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto cur_minus_1   = simd_load_one_to_left<T, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto cur           = simd_load_si128<T, alignedLoad>(srcp + x);
        auto cur_plus_1    = simd_load_one_to_right<T, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto cur_plus_2    = simd_load_two_to_right<T, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto cur_plus_3    = simd_load_three_to_right<T, border == BorderMode::RIGHT, alignedLoad>(srcp + x);

        auto next_minus_3  = simd_load_three_to_left<T, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto next_minus_2  = simd_load_two_to_left<T, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto next_minus_1  = simd_load_one_to_left<T, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto next          = simd_load_si128<T, alignedLoad>(srcpn2 + x);
        auto next_plus_1   = simd_load_one_to_right<T, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto next_plus_2   = simd_load_two_to_right<T, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto next_plus_3   = simd_load_three_to_right<T, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);

        auto adiff_m3_p3 = simd_abs_diff_epu<T>(cur_minus_3, next_plus_3);
        simd_store_si128<T, alignedStore>(buffers[ADIFF_M3_P3] + bufferOffset + x, adiff_m3_p3);

        auto adiff_m2_p2 = simd_abs_diff_epu<T>(cur_minus_2, next_plus_2);
        simd_store_si128<T, alignedStore>(buffers[ADIFF_M2_P2] + bufferOffset + x, adiff_m2_p2);

        auto adiff_m1_p1 = simd_abs_diff_epu<T>(cur_minus_1, next_plus_1);
        simd_store_si128<T, alignedStore>(buffers[ADIFF_M1_P1] + bufferOffset + x, adiff_m1_p1);

        auto adiff_0     = simd_abs_diff_epu<T>(cur, next);
        simd_store_si128<T, alignedStore>(buffers[ADIFF_P0_M0] + bufferOffset + x, adiff_0);

        auto adiff_p1_m1 = simd_abs_diff_epu<T>(cur_plus_1, next_minus_1);
        simd_store_si128<T, alignedStore>(buffers[ADIFF_P1_M1] + bufferOffset + x, adiff_p1_m1);

        auto adiff_p2_m2 = simd_abs_diff_epu<T>(cur_plus_2, next_minus_2);
        simd_store_si128<T, alignedStore>(buffers[ADIFF_P2_M2] + bufferOffset + x, adiff_p2_m2);

        auto adiff_p3_m3 = simd_abs_diff_epu<T>(cur_plus_3, next_minus_3);
        simd_store_si128<T, alignedStore>(buffers[ADIFF_P3_M3] + bufferOffset + x, adiff_p3_m3);

        //////////////////////////////////////////////////////////////////////////
        auto temp1 = calculateSangnom<T>(cur_minus_1, cur, cur_plus_1);
        auto temp2 = calculateSangnom<T>(next_plus_1, next, next_minus_1);

        //abs((cur_minus_1*4 + cur*5 - cur_plus_1) / 8  - (next_plus_1*4 + next*5 - next_minus_1) / 8)
        auto absdiff_p1_p2 = simd_abs_diff_epu<T>(temp1, temp2);
        simd_store_si128<T, alignedStore>(buffers[SG_FORWARD] + bufferOffset + x, absdiff_p1_p2);
        //////////////////////////////////////////////////////////////////////////
        auto temp3 = calculateSangnom<T>(cur_plus_1, cur, cur_minus_1);
        auto temp4 = calculateSangnom<T>(next_minus_1, next, next_plus_1);

        //abs((cur_plus_1*4 + cur*5 - cur_minus_1) / 8  - (next_minus_1*4 + next*5 - next_plus_1) / 8)
        auto absdiff_p3_p4 = simd_abs_diff_epu<T>(temp3, temp4);
        simd_store_si128<T, alignedStore>(buffers[SG_REVERSE] + bufferOffset + x, absdiff_p3_p4);
        //////////////////////////////////////////////////////////////////////////
    }
}

template <typename T>
static inline void prepareBuffers(const T* srcp, T* buffers[BUFFERS_COUNT], int width, int height, int srcStride, int bufferStride) {

    auto srcpn2 = srcp + srcStride * 2;

    int bufferOffset = bufferStride;
    const int pixelPerInst = 16 / sizeof(T);
    const int widthMod = (width + (pixelPerInst - 1)) & ~(pixelPerInst - 1);

    for (int y = 0; y < height / 2 - 1; ++y) {

        prepareBuffersLine<T, BorderMode::LEFT, true, true>(srcp, srcpn2, buffers, bufferOffset, pixelPerInst);
        prepareBuffersLine<T, BorderMode::NONE, true, true>(srcp + pixelPerInst, srcpn2 + pixelPerInst, buffers, bufferOffset + pixelPerInst, widthMod - pixelPerInst);
        prepareBuffersLine<T, BorderMode::RIGHT, false, false>(srcp + width - pixelPerInst, srcpn2 + width - pixelPerInst, buffers, bufferOffset + width - pixelPerInst, pixelPerInst);

        srcp += srcStride * 2;
        srcpn2 += srcStride * 2;
        bufferOffset += bufferStride;
    }
}

template <BorderMode border>
static inline void finalizeBufferProcessingBlock(uint8_t* temp, uint8_t* srcpn, int x) {

    auto cur_minus_6_lo = simd_load_three_to_left<uint16_t, border == BorderMode::LEFT, true>(reinterpret_cast<uint16_t*>(temp + x * 2));
    auto cur_minus_4_lo = simd_load_two_to_left<uint16_t, border == BorderMode::LEFT, true>(reinterpret_cast<uint16_t*>(temp + x * 2));
    auto cur_minus_2_lo = simd_load_one_to_left<uint16_t, border == BorderMode::LEFT, true>(reinterpret_cast<uint16_t*>(temp + x * 2));
    auto cur_lo         = simd_load_si128<uint8_t, true>(temp + x * 2);
    auto cur_plus_2_lo  = simd_load_one_to_right<uint16_t, false, true>(reinterpret_cast<uint16_t*>(temp + x * 2));
    auto cur_plus_4_lo  = simd_load_two_to_right<uint16_t, false, true>(reinterpret_cast<uint16_t*>(temp + x * 2));
    auto cur_plus_6_lo  = simd_load_three_to_right<uint16_t, false, true>(reinterpret_cast<uint16_t*>(temp + x * 2));

    auto cur_minus_6_hi = simd_load_three_to_left<uint16_t, false, true>(reinterpret_cast<uint16_t*>(temp + x * 2 + 16));
    auto cur_minus_4_hi = simd_load_two_to_left<uint16_t, false, true>(reinterpret_cast<uint16_t*>(temp + x * 2 + 16));
    auto cur_minus_2_hi = simd_load_one_to_left<uint16_t, false, true>(reinterpret_cast<uint16_t*>(temp + x * 2 + 16));
    auto cur_hi         = simd_load_si128<uint8_t, true>(temp + x * 2 + 16);
    auto cur_plus_2_hi  = simd_load_one_to_right<uint16_t, border == BorderMode::RIGHT, true>(reinterpret_cast<uint16_t*>(temp + x * 2 + 16));
    auto cur_plus_4_hi  = simd_load_two_to_right<uint16_t, border == BorderMode::RIGHT, true>(reinterpret_cast<uint16_t*>(temp + x * 2 + 16));
    auto cur_plus_6_hi  = simd_load_three_to_right<uint16_t, border == BorderMode::RIGHT, true>(reinterpret_cast<uint16_t*>(temp + x * 2 + 16));

    auto sum_lo = _mm_adds_epu16(cur_minus_6_lo, cur_minus_4_lo);
    sum_lo = _mm_adds_epu16(sum_lo, cur_minus_2_lo);
    sum_lo = _mm_adds_epu16(sum_lo, cur_lo);
    sum_lo = _mm_adds_epu16(sum_lo, cur_plus_2_lo);
    sum_lo = _mm_adds_epu16(sum_lo, cur_plus_4_lo);
    sum_lo = _mm_adds_epu16(sum_lo, cur_plus_6_lo);

    sum_lo = _mm_srli_epi16(sum_lo, 4);

    auto sum_hi = _mm_adds_epu16(cur_minus_6_hi, cur_minus_4_hi);
    sum_hi = _mm_adds_epu16(sum_hi, cur_minus_2_hi);
    sum_hi = _mm_adds_epu16(sum_hi, cur_hi);
    sum_hi = _mm_adds_epu16(sum_hi, cur_plus_2_hi);
    sum_hi = _mm_adds_epu16(sum_hi, cur_plus_4_hi);
    sum_hi = _mm_adds_epu16(sum_hi, cur_plus_6_hi);

    sum_hi = _mm_srli_epi16(sum_hi, 4);

    auto result = _mm_packus_epi16(sum_lo, sum_hi);
    simd_store_si128<uint8_t, true>(srcpn + x, result);
}

// NOTE: lack of _mm_adds_epu32
template <BorderMode border>
static inline void finalizeBufferProcessingBlock(uint16_t* temp, uint16_t* srcpn, int x) {

    auto cur_minus_6_lo = simd_load_three_to_left<uint32_t, border == BorderMode::LEFT, true>(reinterpret_cast<uint32_t*>(temp + x * 2));
    auto cur_minus_4_lo = simd_load_two_to_left<uint32_t, border == BorderMode::LEFT, true>(reinterpret_cast<uint32_t*>(temp + x * 2));
    auto cur_minus_2_lo = simd_load_one_to_left<uint32_t, border == BorderMode::LEFT, true>(reinterpret_cast<uint32_t*>(temp + x * 2));
    auto cur_lo         = simd_load_si128<uint16_t, true>(temp + x * 2);
    auto cur_plus_2_lo  = simd_load_one_to_right<uint32_t, false, true>(reinterpret_cast<uint32_t*>(temp + x * 2));
    auto cur_plus_4_lo  = simd_load_two_to_right<uint32_t, false, true>(reinterpret_cast<uint32_t*>(temp + x * 2));
    auto cur_plus_6_lo  = simd_load_three_to_right<uint32_t, false, true>(reinterpret_cast<uint32_t*>(temp + x * 2));

    auto cur_minus_6_hi = simd_load_three_to_left<uint32_t, false, true>(reinterpret_cast<uint32_t*>(temp + x * 2 + 8));
    auto cur_minus_4_hi = simd_load_two_to_left<uint32_t, false, true>(reinterpret_cast<uint32_t*>(temp + x * 2 + 8));
    auto cur_minus_2_hi = simd_load_one_to_left<uint32_t, false, true>(reinterpret_cast<uint32_t*>(temp + x * 2 + 8));
    auto cur_hi         = simd_load_si128<uint16_t, true>(temp + x * 2 + 8);
    auto cur_plus_2_hi  = simd_load_one_to_right<uint32_t, border == BorderMode::RIGHT, true>(reinterpret_cast<uint32_t*>(temp + x * 2 + 8));
    auto cur_plus_4_hi  = simd_load_two_to_right<uint32_t, border == BorderMode::RIGHT, true>(reinterpret_cast<uint32_t*>(temp + x * 2 + 8));
    auto cur_plus_6_hi  = simd_load_three_to_right<uint32_t, border == BorderMode::RIGHT, true>(reinterpret_cast<uint32_t*>(temp + x * 2 + 8));

    auto sum_lo = _mm_add_epi32(cur_minus_6_lo, cur_minus_4_lo);
    sum_lo = _mm_add_epi32(sum_lo, cur_minus_2_lo);
    sum_lo = _mm_add_epi32(sum_lo, cur_lo);
    sum_lo = _mm_add_epi32(sum_lo, cur_plus_2_lo);
    sum_lo = _mm_add_epi32(sum_lo, cur_plus_4_lo);
    sum_lo = _mm_add_epi32(sum_lo, cur_plus_6_lo);

    sum_lo = _mm_srli_epi32(sum_lo, 4);

    auto sum_hi = _mm_add_epi32(cur_minus_6_hi, cur_minus_4_hi);
    sum_hi = _mm_add_epi32(sum_hi, cur_minus_2_hi);
    sum_hi = _mm_add_epi32(sum_hi, cur_hi);
    sum_hi = _mm_add_epi32(sum_hi, cur_plus_2_hi);
    sum_hi = _mm_add_epi32(sum_hi, cur_plus_4_hi);
    sum_hi = _mm_add_epi32(sum_hi, cur_plus_6_hi);

    sum_hi = _mm_srli_epi32(sum_hi, 4);

    auto result = _mm_packus_epi32(sum_lo, sum_hi);
    simd_store_si128<uint16_t, true>(srcpn + x, result);
}

static inline void processBuffer(uint8_t* buffer, uint8_t* temp, int stride, int height) {

    auto srcp = buffer;
    auto srcpn1 = srcp + stride;
    auto srcpn2 = srcpn1 + stride;

    for (int y = 0; y < height - 1; ++y) {

        auto zero = _mm_setzero_si128();

        for(int x = 0; x < stride; x += 16) {

            auto src = simd_load_si128<uint8_t, true>(srcp + x);
            auto srcn = simd_load_si128<uint8_t, true>(srcpn1 + x);
            auto srcn2 = simd_load_si128<uint8_t, true>(srcpn2 + x);

            auto src_lo = _mm_unpacklo_epi8(src, zero);
            auto srcn_lo = _mm_unpacklo_epi8(srcn, zero);
            auto srcn2_lo = _mm_unpacklo_epi8(srcn2, zero);

            auto src_hi     = _mm_unpackhi_epi8(src, zero);
            auto srcn_hi    = _mm_unpackhi_epi8(srcn, zero);
            auto srcn2_hi   = _mm_unpackhi_epi8(srcn2, zero);

            auto sum_lo = _mm_adds_epu16(src_lo, srcn_lo);
            sum_lo = _mm_adds_epu16(sum_lo, srcn2_lo);

            auto sum_hi = _mm_adds_epu16(src_hi, srcn_hi);
            sum_hi = _mm_adds_epu16(sum_hi, srcn2_hi);

            simd_store_si128<uint8_t, true>(temp + (x * 2), sum_lo);
            simd_store_si128<uint8_t, true>(temp + (x * 2) + 16, sum_hi);
        }

        finalizeBufferProcessingBlock<BorderMode::LEFT>(temp, srcpn1, 0);

        for (int x = 16; x < stride - 16; x += 16) {
            finalizeBufferProcessingBlock<BorderMode::NONE>(temp, srcpn1, x);
        }

        finalizeBufferProcessingBlock<BorderMode::RIGHT>(temp, srcpn1, stride - 16);

        srcp += stride;
        srcpn1 += stride;
        srcpn2 += stride;
    }
}

// NOTE: lack of _mm_adds_epu32
static inline void processBuffer(uint16_t* buffer, uint16_t* temp, int stride, int height) {

    auto srcp = buffer;
    auto srcpn1 = srcp + stride;
    auto srcpn2 = srcpn1 + stride;

    for (int y = 0; y < height - 1; ++y) {

        auto zero = _mm_setzero_si128();

        for(int x = 0; x < stride; x += 8) {

            auto src = simd_load_si128<uint16_t, true>(srcp + x);
            auto srcn = simd_load_si128<uint16_t, true>(srcpn1 + x);
            auto srcn2 = simd_load_si128<uint16_t, true>(srcpn2 + x);

            auto src_lo = _mm_unpacklo_epi16(src, zero);
            auto srcn_lo = _mm_unpacklo_epi16(srcn, zero);
            auto srcn2_lo = _mm_unpacklo_epi16(srcn2, zero);

            auto src_hi     = _mm_unpackhi_epi16(src, zero);
            auto srcn_hi    = _mm_unpackhi_epi16(srcn, zero);
            auto srcn2_hi   = _mm_unpackhi_epi16(srcn2, zero);

            auto sum_lo = _mm_add_epi32(src_lo, srcn_lo);
            sum_lo = _mm_add_epi32(sum_lo, srcn2_lo);

            auto sum_hi = _mm_add_epi32(src_hi, srcn_hi);
            sum_hi = _mm_add_epi32(sum_hi, srcn2_hi);

            simd_store_si128<uint16_t, true>(temp + (x * 2), sum_lo);
            simd_store_si128<uint16_t, true>(temp + (x * 2) + 8, sum_hi);
        }

        finalizeBufferProcessingBlock<BorderMode::LEFT>(temp, srcpn1, 0);

        for (int x = 8; x < stride - 8; x += 8) {
            finalizeBufferProcessingBlock<BorderMode::NONE>(temp, srcpn1, x);
        }

        finalizeBufferProcessingBlock<BorderMode::RIGHT>(temp, srcpn1, stride - 8);

        srcp += stride;
        srcpn1 += stride;
        srcpn2 += stride;
    }
}

template <BorderMode border, bool alignedLoad, bool alignedLoadBuffer, bool alignedStore>
static inline void finalizePlaneLine(const uint8_t* srcp, const uint8_t* srcpn2, uint8_t* dstpn, uint8_t* buffers[BUFFERS_COUNT], int bufferOffset, int width, const __m128i& aath) {

    auto zero = _mm_setzero_si128();

    for (int x = 0; x < width; x += 16) {
        auto buf0 = simd_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_M3_P3] + bufferOffset + x);
        auto buf1 = simd_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_M2_P2] + bufferOffset + x);
        auto buf2 = simd_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_M1_P1] + bufferOffset + x);
        auto buf3 = simd_load_si128<uint8_t, alignedLoadBuffer>(buffers[SG_FORWARD]  + bufferOffset + x);
        auto buf4 = simd_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_P0_M0] + bufferOffset + x);
        auto buf5 = simd_load_si128<uint8_t, alignedLoadBuffer>(buffers[SG_REVERSE]  + bufferOffset + x);
        auto buf6 = simd_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_P1_M1] + bufferOffset + x);
        auto buf7 = simd_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_P2_M2] + bufferOffset + x);
        auto buf8 = simd_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_P3_M3] + bufferOffset + x);

        auto cur_minus_3   = simd_load_three_to_left<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto cur_minus_2   = simd_load_two_to_left<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto cur_minus_1   = simd_load_one_to_left<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto cur           = simd_load_si128<uint8_t, alignedLoad>(srcp + x);
        auto cur_plus_1    = simd_load_one_to_right<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto cur_plus_2    = simd_load_two_to_right<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto cur_plus_3    = simd_load_three_to_right<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);

        auto next_minus_3  = simd_load_three_to_left<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto next_minus_2  = simd_load_two_to_left<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto next_minus_1  = simd_load_one_to_left<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto next          = simd_load_si128<uint8_t, alignedLoad>(srcpn2 + x);
        auto next_plus_1   = simd_load_one_to_right<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto next_plus_2   = simd_load_two_to_right<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto next_plus_3   = simd_load_three_to_right<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);

        auto minbuf = _mm_min_epu8(buf0, buf1);
        minbuf = _mm_min_epu8(minbuf, buf2);
        minbuf = _mm_min_epu8(minbuf, buf3);
        minbuf = _mm_min_epu8(minbuf, buf4);
        minbuf = _mm_min_epu8(minbuf, buf5);
        minbuf = _mm_min_epu8(minbuf, buf6);
        minbuf = _mm_min_epu8(minbuf, buf7);
        minbuf = _mm_min_epu8(minbuf, buf8);

        auto processed = _mm_setzero_si128();

        processed = blendAvgOnMinimalBuffer<uint8_t>(cur_minus_3, next_plus_3, buf0, minbuf, processed, zero);
        processed = blendAvgOnMinimalBuffer<uint8_t>(cur_plus_3, next_minus_3, buf8, minbuf, processed, zero);

        processed = blendAvgOnMinimalBuffer<uint8_t>(cur_minus_2, next_plus_2, buf1, minbuf, processed, zero);
        processed = blendAvgOnMinimalBuffer<uint8_t>(cur_plus_2, next_minus_2, buf7, minbuf, processed, zero);

        processed = blendAvgOnMinimalBuffer<uint8_t>(cur_minus_1, next_plus_1, buf2, minbuf, processed, zero);
        processed = blendAvgOnMinimalBuffer<uint8_t>(cur_plus_1, next_minus_1, buf6, minbuf, processed, zero);

        ////////////////////////////////////////////////////////////////////////////
        auto temp1 = calculateSangnom<uint8_t>(cur_minus_1, cur, cur_plus_1);
        auto temp2 = calculateSangnom<uint8_t>(next_plus_1, next, next_minus_1);

        processed = blendAvgOnMinimalBuffer<uint8_t>(temp1, temp2, buf3, minbuf, processed, zero);
        ////////////////////////////////////////////////////////////////////////////
        auto temp3 = calculateSangnom<uint8_t>(cur_plus_1, cur, cur_minus_1);
        auto temp4 = calculateSangnom<uint8_t>(next_minus_1, next, next_plus_1);

        processed = blendAvgOnMinimalBuffer<uint8_t>(temp3, temp4, buf5, minbuf, processed, zero);
        ////////////////////////////////////////////////////////////////////////////

        auto average = _mm_avg_epu8(cur, next);

        auto buf4IsMinimal = _mm_cmpeq_epi8(buf4, minbuf);

        auto takeAaa = _mm_subs_epu8(minbuf, aath);
        //this isn't strictly negation, don't optimize
        auto takeProcessed = _mm_cmpeq_epi8(takeAaa, zero);
        auto mask = _mm_andnot_si128(buf4IsMinimal, takeProcessed);

        //blending
        processed = _mm_and_si128(mask, processed);
        average = _mm_andnot_si128(mask, average);
        auto result = _mm_or_si128(processed, average);

        simd_store_si128<uint8_t, alignedStore>(dstpn + x, result);
    }
}

template <BorderMode border, bool alignedLoad, bool alignedLoadBuffer, bool alignedStore>
static inline void finalizePlaneLine(const uint16_t* srcp, const uint16_t* srcpn2, uint16_t* dstpn, uint16_t* buffers[BUFFERS_COUNT], int bufferOffset, int width, const __m128i& aath) {

    auto zero = _mm_setzero_si128();

    for (int x = 0; x < width; x += 8) {
        auto buf0 = simd_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_M3_P3] + bufferOffset + x);
        auto buf1 = simd_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_M2_P2] + bufferOffset + x);
        auto buf2 = simd_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_M1_P1] + bufferOffset + x);
        auto buf3 = simd_load_si128<uint16_t, alignedLoadBuffer>(buffers[SG_FORWARD]  + bufferOffset + x);
        auto buf4 = simd_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_P0_M0] + bufferOffset + x);
        auto buf5 = simd_load_si128<uint16_t, alignedLoadBuffer>(buffers[SG_REVERSE]  + bufferOffset + x);
        auto buf6 = simd_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_P1_M1] + bufferOffset + x);
        auto buf7 = simd_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_P2_M2] + bufferOffset + x);
        auto buf8 = simd_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_P3_M3] + bufferOffset + x);

        auto cur_minus_3   = simd_load_three_to_left<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto cur_minus_2   = simd_load_two_to_left<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto cur_minus_1   = simd_load_one_to_left<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto cur           = simd_load_si128<uint16_t, alignedLoad>(srcp + x);
        auto cur_plus_1    = simd_load_one_to_right<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto cur_plus_2    = simd_load_two_to_right<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto cur_plus_3    = simd_load_three_to_right<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);

        auto next_minus_3  = simd_load_three_to_left<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto next_minus_2  = simd_load_two_to_left<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto next_minus_1  = simd_load_one_to_left<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto next          = simd_load_si128<uint16_t, alignedLoad>(srcpn2 + x);
        auto next_plus_1   = simd_load_one_to_right<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto next_plus_2   = simd_load_two_to_right<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto next_plus_3   = simd_load_three_to_right<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);

        auto minbuf = _mm_min_epu16(buf0, buf1);
        minbuf = _mm_min_epu16(minbuf, buf2);
        minbuf = _mm_min_epu16(minbuf, buf3);
        minbuf = _mm_min_epu16(minbuf, buf4);
        minbuf = _mm_min_epu16(minbuf, buf5);
        minbuf = _mm_min_epu16(minbuf, buf6);
        minbuf = _mm_min_epu16(minbuf, buf7);
        minbuf = _mm_min_epu16(minbuf, buf8);

        auto processed = _mm_setzero_si128();

        processed = blendAvgOnMinimalBuffer<uint16_t>(cur_minus_3, next_plus_3, buf0, minbuf, processed, zero);
        processed = blendAvgOnMinimalBuffer<uint16_t>(cur_plus_3, next_minus_3, buf8, minbuf, processed, zero);

        processed = blendAvgOnMinimalBuffer<uint16_t>(cur_minus_2, next_plus_2, buf1, minbuf, processed, zero);
        processed = blendAvgOnMinimalBuffer<uint16_t>(cur_plus_2, next_minus_2, buf7, minbuf, processed, zero);

        processed = blendAvgOnMinimalBuffer<uint16_t>(cur_minus_1, next_plus_1, buf2, minbuf, processed, zero);
        processed = blendAvgOnMinimalBuffer<uint16_t>(cur_plus_1, next_minus_1, buf6, minbuf, processed, zero);

        ////////////////////////////////////////////////////////////////////////////
        auto temp1 = calculateSangnom<uint16_t>(cur_minus_1, cur, cur_plus_1);
        auto temp2 = calculateSangnom<uint16_t>(next_plus_1, next, next_minus_1);

        processed = blendAvgOnMinimalBuffer<uint16_t>(temp1, temp2, buf3, minbuf, processed, zero);
        ////////////////////////////////////////////////////////////////////////////
        auto temp3 = calculateSangnom<uint16_t>(cur_plus_1, cur, cur_minus_1);
        auto temp4 = calculateSangnom<uint16_t>(next_minus_1, next, next_plus_1);

        processed = blendAvgOnMinimalBuffer<uint16_t>(temp3, temp4, buf5, minbuf, processed, zero);
        ////////////////////////////////////////////////////////////////////////////

        auto average = _mm_avg_epu16(cur, next);

        auto buf4IsMinimal = _mm_cmpeq_epi16(buf4, minbuf);

        auto takeAaa = _mm_subs_epu16(minbuf, aath);
        //this isn't strictly negation, don't optimize
        auto takeProcessed = _mm_cmpeq_epi16(takeAaa, zero);
        auto mask = _mm_andnot_si128(buf4IsMinimal, takeProcessed);

        //blending
        processed = _mm_and_si128(mask, processed);
        average = _mm_andnot_si128(mask, average);
        auto result = _mm_or_si128(processed, average);

        simd_store_si128<uint16_t, alignedStore>(dstpn + x, result);
    }
}

template <typename T>
static inline void finalizePlane(const T* srcp, T* dstp, T* buffers[BUFFERS_COUNT], int srcStride, int dstStride, int bufferStride, int width, int height, float aa) {

    auto dstpn = dstp + dstStride;
    auto srcpn2 = srcp + srcStride * 2;

    __m128i aav;
    if (std::is_same<T, uint8_t>::value)
        aav = _mm_set1_epi8(aa);
    else if (std::is_same<T, uint16_t>::value)
        aav = _mm_set1_epi16(aa);
    int bufferOffset = bufferStride;
    const int pixelPerInst = 16 / sizeof(T);
    const int widthMod = (width + (pixelPerInst - 1)) & ~(pixelPerInst - 1);

    for (int y = 0; y < height / 2 - 1; ++y) {

        finalizePlaneLine<BorderMode::LEFT, true, true, true>(srcp, srcpn2, dstpn, buffers, bufferOffset, pixelPerInst, aav);

        finalizePlaneLine<BorderMode::NONE, true, true, true>(srcp + pixelPerInst, srcpn2 + pixelPerInst, dstpn + pixelPerInst, buffers, bufferOffset + pixelPerInst, widthMod - pixelPerInst, aav);

        finalizePlaneLine<BorderMode::RIGHT, false, false, false>(srcp + width - pixelPerInst, srcpn2 + width - pixelPerInst, dstpn + width - pixelPerInst, buffers, bufferOffset + width - pixelPerInst, pixelPerInst, aav);

        srcp += srcStride * 2;
        srcpn2 += srcStride * 2;
        dstpn += dstStride * 2;
        bufferOffset += bufferStride;
    }
}

template <typename T>
static inline void sangnom(const T *srcp, const int srcStride, T *dstp, const int dstStride, const int w, const int h, SangNomData *d, int plane, T *buffers[BUFFERS_COUNT], T *bufferLine)
{
    if (d->offset == 0) {
        // keep top field so the bottom line can't be interpolated
        // just copy the data from its correspond top field
        vs_bitblt(dstp + (h - 1) * dstStride, dstStride, srcp + (h - 2) * srcStride, srcStride, w, 1);
    } else {
        // keep bottom field so the top line can't be interpolated
        // just copy the data from its correspond bottom field
        vs_bitblt(dstp, dstStride, srcp + srcStride, srcStride, w, 1);
    }

    // prepare buffers
    prepareBuffers<T>(srcp + d->offset * srcStride, buffers, w, h, srcStride, d->bufferStride);

    // process buffers
    for (int i = 0; i < BUFFERS_COUNT; ++i)
        processBuffer(buffers[i], bufferLine, d->bufferStride, d->bufferHeight);

    // finalize plane
    finalizePlane(srcp + d->offset * srcStride, dstp + d->offset * dstStride, buffers, srcStride, dstStride, d->bufferStride, w, h, d->aaf);
}


static void VS_CC sangnomInit(VSMap *in, VSMap *out, void **instanceData, VSNode* node, VSCore *core, const VSAPI *vsapi)
{
    SangNomData *d = reinterpret_cast<SangNomData*> (*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef *VS_CC sangnomGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
    SangNomData *d = reinterpret_cast<SangNomData*> (*instanceData);

    if (activationReason == arInitial) {

        vsapi->requestFrameFilter(n, d->node, frameCtx);

    } else if (activationReason == arAllFramesReady) {

        auto src = vsapi->getFrameFilter(n, d->node, frameCtx);
        auto dst = vsapi->copyFrame(src, core);

        /////////////////////////////////////////////////////////////////////////////////////
        void *bufferPool;
        void *buffers[BUFFERS_COUNT];

        void *bufferLine;

        if (d->vi->format->sampleType == stInteger) {
            if (d->vi->format->bitsPerSample == 8) {
                bufferPool = vs_aligned_malloc<void>(sizeof(uint8_t) * d->bufferStride * (d->bufferHeight + 1) * BUFFERS_COUNT, alignment);
                // separate bufferpool to multiple pieces
                for (int i = 0; i < BUFFERS_COUNT; ++i)
                    buffers[i] = reinterpret_cast<uint8_t*>(bufferPool) + i * d->bufferStride * (d->bufferHeight + 1);
                bufferLine = vs_aligned_malloc<void>(sizeof(uint8_t) * d->bufferStride * 2, alignment);
            } else {
                bufferPool = vs_aligned_malloc<void>(sizeof(uint16_t) * d->bufferStride * (d->bufferHeight + 1) * BUFFERS_COUNT, alignment);
                // separate bufferpool to multiple pieces
                for (int i = 0; i < BUFFERS_COUNT; ++i)
                    buffers[i] = reinterpret_cast<uint16_t*>(bufferPool) + i * d->bufferStride * (d->bufferHeight + 1);
                bufferLine = vs_aligned_malloc<void>(sizeof(uint16_t) * d->bufferStride * 2, alignment);
            }
        } else {
            bufferPool = vs_aligned_malloc<void>(sizeof(float) * d->bufferStride * (d->bufferHeight + 1) * BUFFERS_COUNT, alignment);
            // separate bufferpool to multiple pieces
            for (int i = 0; i < BUFFERS_COUNT; ++i)
                buffers[i] = reinterpret_cast<float*>(bufferPool) + i * d->bufferStride * (d->bufferHeight + 1);
            bufferLine = vs_aligned_malloc<void>(sizeof(float) * d->bufferStride * 2, alignment);
        }
        /////////////////////////////////////////////////////////////////////////////////////

        for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {

            if (!d->planes[plane])
                continue;

            auto srcp = vsapi->getReadPtr(src, plane);
            auto srcStride = vsapi->getStride(src, plane) / d->vi->format->bytesPerSample;
            auto dstp = vsapi->getWritePtr(dst, plane);
            auto dstStride = vsapi->getStride(dst, plane) / d->vi->format->bytesPerSample;
            auto width = vsapi->getFrameWidth(src, plane);
            auto height = vsapi->getFrameHeight(src, plane);

            if (d->vi->format->sampleType == stInteger) {
                if (d->vi->format->bitsPerSample == 8)
                    sangnom<uint8_t>(srcp, srcStride, dstp, dstStride, width, height, d, plane, reinterpret_cast<uint8_t**>(buffers), reinterpret_cast<uint8_t*>(bufferLine));
                else
                    sangnom<uint16_t>(reinterpret_cast<const uint16_t*>(srcp), srcStride, reinterpret_cast<uint16_t*>(dstp), dstStride, width, height, d, plane, reinterpret_cast<uint16_t**>(buffers), reinterpret_cast<uint16_t*>(bufferLine));
            } else {
                //sangnom<float>(reinterpret_cast<const float*>(srcp), srcStride, reinterpret_cast<float*>(dstp), dstStride, width, height, d, plane, reinterpret_cast<float**>(buffers), reinterpret_cast<float*>(bufferLine));
            }
        }

        vs_aligned_free(bufferLine);
        vs_aligned_free(bufferPool);

        vsapi->freeFrame(src);
        return dst;
    }
    return nullptr;
}

static void VS_CC sangnomFree(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
    SangNomData *d = reinterpret_cast<SangNomData*> (instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

static void VS_CC sangnomCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    SangNomData *d = new SangNomData();

    int err;

    d->node = vsapi->propGetNode(in, "clip", 0, 0);
    d->vi = vsapi->getVideoInfo(d->node);

    try {
        if (d->vi->format->sampleType != stInteger || d->vi->format->bitsPerSample < 8 || d->vi->format->bitsPerSample > 16)
            throw std::string("only 8...16 bit integer support");

        d->order = vsapi->propGetInt(in, "order", 0, &err);
        if (err)
            d->order = VapourSynthFieldBasedToSangNomOrder(vsapi->propGetInt(in, "_FieldBased", 0, &err));
        if (err)
            d->order = SNOT_SFR_KT;

        if (d->order < 0  || d->order > 2)
            throw std::string("order must be 0 ... 2");

        d->aa = vsapi->propGetInt(in, "aa", 0, &err);
        if (err) d->aa = 64;

        if (d->aa < 0 || d->aa > 168)
            throw std::string("aa must be 0 ... 168");

        // tweak aa value for different format
        if (d->vi->format->sampleType == stInteger)
            d->aaf = d->aa << (d->vi->format->bitsPerSample - 8);
        else
            d->aaf = d->aa / 256.0f;


        for (int i = 0; i < 3; ++i)
            d->planes[i] = false;

        int m = vsapi->propNumElements(in, "planes");

        if (m <= 0) {
            for (int i = 0; i < 3; ++i) {
                d->planes[i] = true;
            }
        } else {
            for (int i = 0; i < m; ++i) {
                int p = vsapi->propGetInt(in, "planes", i, &err);
                if (p < 0 || p > d->vi->format->numPlanes - 1)
                    throw std::string("planes index out of bound");
                d->planes[p] = true;
            }
        }

    } catch (std::string &errorMsg) {
        vsapi->freeNode(d->node);
        vsapi->setError(out, std::string("SangNom: ").append(errorMsg).c_str());
        return;
    }

    if (d->order == SNOT_DFR) {
        d->offset = 0;  // keep top field
    } else {
        d->offset = !d->order;
    }

    d->bufferStride = (d->vi->width + alignment - 1) & ~(alignment - 1);
    d->bufferHeight = (d->vi->height + 1) >> 1;

    vsapi->createFilter(in, out, "sangnom", sangnomInit, sangnomGetFrame, sangnomFree, fmParallel, 0, d, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
    configFunc("com.mio.sangnom", "sangnom", "VapourSynth SangNom", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("SangNom", "clip:clip;"
                        "order:int:opt;"
                        "aa:int:opt;"
                        "planes:int[]:opt;",
                        sangnomCreate, nullptr, plugin);
}
