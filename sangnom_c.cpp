
/**
 *  SangNom
 *
 **/

#include <VapourSynth.h>
#include <VSHelper.h>

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>

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

enum Buffers {
    ADIFF_M3_P3 = 0,
    ADIFF_M2_P2 = 1,
    ADIFF_M1_P1 = 2,
    ADIFF_P0_M0 = 4,
    ADIFF_P1_M1 = 6,
    ADIFF_P2_M2 = 7,
    ADIFF_P3_M3 = 8,

    SG_FORWARD = 3,
    SG_REVERSE = 5,

    TOTAL_BUFFERS = 9,
};

template <typename T, typename IType>
static inline IType loadPixel(const T *srcp, int curPos, int offset, int width)
{
    int reqPos = curPos + offset;
    if (reqPos >= 0 && reqPos < width)
        return srcp[reqPos];
    if (reqPos >= 0)
        return srcp[width-1];
    return srcp[0];
}

template <typename T>
static inline T absDiff(const T &a, const T &b)
{
    return std::abs(a - b);
}

template <>
inline float absDiff<float>(const float &a, const float &b)
{
    return std::fabs(a - b);
}

template <typename T, typename IType>
static inline IType calculateSangNom(const T &p1, const T &p2, const T &p3)
{
    IType sum = p1 * 5 + p2 * 4 - p3;
    return sum / 8;
}

template <typename T>
static inline void copyField(const T *srcp, const int srcStride, T *dstp, const int dstStride, const int w, const int h, SangNomData *d)
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
}

template <typename T, typename IType>
static inline void prepareBuffers_c(const T *srcp, const int srcStride, const int w, const int h, const int bufferStride, T *buffers[TOTAL_BUFFERS])
{
    auto srcpn2 = srcp + srcStride * 2;

    int bufferOffset = bufferStride;

    for (int y = 0; y < h / 2 - 1; ++y) {

        for (int x = 0; x < w; ++x) {

            const IType currLineM3 = loadPixel<T, IType>(srcp, x, -3, w);
            const IType currLineM2 = loadPixel<T, IType>(srcp, x, -2, w);
            const IType currLineM1 = loadPixel<T, IType>(srcp, x, -1, w);
            const IType currLine   = srcp[x];
            const IType currLineP1 = loadPixel<T, IType>(srcp, x, 1, w);
            const IType currLineP2 = loadPixel<T, IType>(srcp, x, 2, w);
            const IType currLineP3 = loadPixel<T, IType>(srcp, x, 3, w);

            const IType nextLineM3 = loadPixel<T, IType>(srcpn2, x, -3, w);
            const IType nextLineM2 = loadPixel<T, IType>(srcpn2, x, -2, w);
            const IType nextLineM1 = loadPixel<T, IType>(srcpn2, x, -1, w);
            const IType nextLine   = srcpn2[x];
            const IType nextLineP1 = loadPixel<T, IType>(srcpn2, x, 1, w);
            const IType nextLineP2 = loadPixel<T, IType>(srcpn2, x, 2, w);
            const IType nextLineP3 = loadPixel<T, IType>(srcpn2, x, 3, w);

            const IType forwardSangNom1 = calculateSangNom<T, IType>(currLineM1, currLine, currLineP1);
            const IType forwardSangNom2 = calculateSangNom<T, IType>(nextLineP1, nextLine, nextLineM1);
            const IType backwardSangNom1 = calculateSangNom<T, IType>(currLineP1, currLine, currLineM1);
            const IType backwardSangNom2 = calculateSangNom<T, IType>(nextLineM1, nextLine, nextLineP1);

            buffers[ADIFF_M3_P3][bufferOffset + x] = absDiff(currLineM3, nextLineP3);
            buffers[ADIFF_M2_P2][bufferOffset + x] = absDiff(currLineM2, nextLineP2);
            buffers[ADIFF_M1_P1][bufferOffset + x] = absDiff(currLineM1, nextLineP1);
            buffers[ADIFF_P0_M0][bufferOffset + x] = absDiff(currLine, nextLine);
            buffers[ADIFF_P1_M1][bufferOffset + x] = absDiff(currLineP1, nextLineM1);
            buffers[ADIFF_P2_M2][bufferOffset + x] = absDiff(currLineP2, nextLineM2);
            buffers[ADIFF_P3_M3][bufferOffset + x] = absDiff(currLineP3, nextLineM3);

            buffers[SG_FORWARD][bufferOffset + x] = absDiff(forwardSangNom1, forwardSangNom2);
            buffers[SG_REVERSE][bufferOffset + x] = absDiff(backwardSangNom1, backwardSangNom2);
        }

        srcp += srcStride * 2;
        srcpn2 += srcStride * 2;
        bufferOffset += bufferStride;
    }
}

template <typename T, typename IType>
static inline void processBuffers_c(T *bufferp, IType *bufferTemp, const int bufferStride, const int bufferHeight, const int w)
{
    auto bufferpc = bufferp + bufferStride;
    auto bufferpp1 = bufferpc - bufferStride;
    auto bufferpn1 = bufferpc + bufferStride;
    auto bufferTempc = bufferTemp + bufferStride;

    for (int y = 0; y < bufferHeight - 1; ++y) {

        for (int x = 0; x < w; ++x) {
            bufferTempc[x] = bufferpp1[x] + bufferpc[x] + bufferpn1[x];
        }

        bufferpc += bufferStride;
        bufferpp1 += bufferStride;
        bufferpn1 += bufferStride;
        bufferTempc += bufferStride;
    }

    bufferpc = bufferp + bufferStride;
    bufferTempc = bufferTemp + bufferStride;

    for (int y = 0; y < bufferHeight - 1; ++y) {

        for (int x = 0; x < w; ++x) {

            const IType currLineM3 = loadPixel<IType, IType>(bufferTempc, x, -3, w);
            const IType currLineM2 = loadPixel<IType, IType>(bufferTempc, x, -2, w);
            const IType currLineM1 = loadPixel<IType, IType>(bufferTempc, x, -1, w);
            const IType currLine   = bufferTempc[x];
            const IType currLineP1 = loadPixel<IType, IType>(bufferTempc, x, 1, w);
            const IType currLineP2 = loadPixel<IType, IType>(bufferTempc, x, 2, w);
            const IType currLineP3 = loadPixel<IType, IType>(bufferTempc, x, 3, w);

            bufferpc[x] = (currLineM3 + currLineM2 + currLineM1 + currLine + currLineP1 + currLineP2 + currLineP3) / 16;
        }

        bufferpc += bufferStride;
        bufferTempc += bufferStride;
    }
}

template <typename T, typename IType>
static inline void finalizePlane_c(const T *srcp, const int srcStride, T *dstp, const int dstStride, const int w, const int h, const int bufferStride, const T aaf, T *buffers[TOTAL_BUFFERS])
{
    auto srcpn2 = srcp + srcStride * 2;
    auto dstpn = dstp + dstStride;

    int bufferOffset = bufferStride;

    for (int y = 0; y < h / 2 - 1; ++y) {

        for (int x = 0; x < w; ++x) {

            const IType currLineM3 = loadPixel<T, IType>(srcp, x, -3, w);
            const IType currLineM2 = loadPixel<T, IType>(srcp, x, -2, w);
            const IType currLineM1 = loadPixel<T, IType>(srcp, x, -1, w);
            const IType currLine   = srcp[x];
            const IType currLineP1 = loadPixel<T, IType>(srcp, x, 1, w);
            const IType currLineP2 = loadPixel<T, IType>(srcp, x, 2, w);
            const IType currLineP3 = loadPixel<T, IType>(srcp, x, 3, w);

            const IType nextLineM3 = loadPixel<T, IType>(srcpn2, x, -3, w);
            const IType nextLineM2 = loadPixel<T, IType>(srcpn2, x, -2, w);
            const IType nextLineM1 = loadPixel<T, IType>(srcpn2, x, -1, w);
            const IType nextLine   = srcpn2[x];
            const IType nextLineP1 = loadPixel<T, IType>(srcpn2, x, 1, w);
            const IType nextLineP2 = loadPixel<T, IType>(srcpn2, x, 2, w);
            const IType nextLineP3 = loadPixel<T, IType>(srcpn2, x, 3, w);

            const IType forwardSangNom1 = calculateSangNom<T, IType>(currLineM1, currLine, currLineP1);
            const IType forwardSangNom2 = calculateSangNom<T, IType>(nextLineP1, nextLine, nextLineM1);
            const IType backwardSangNom1 = calculateSangNom<T, IType>(currLineP1, currLine, currLineM1);
            const IType backwardSangNom2 = calculateSangNom<T, IType>(nextLineM1, nextLine, nextLineP1);

            IType buf[9];
            buf[0] = buffers[ADIFF_M3_P3][bufferOffset + x];
            buf[1] = buffers[ADIFF_M2_P2][bufferOffset + x];
            buf[2] = buffers[ADIFF_M1_P1][bufferOffset + x];
            buf[3] = buffers[SG_FORWARD][bufferOffset + x];
            buf[4] = buffers[ADIFF_P0_M0][bufferOffset + x];
            buf[5] = buffers[SG_REVERSE][bufferOffset + x];
            buf[6] = buffers[ADIFF_P1_M1][bufferOffset + x];
            buf[7] = buffers[ADIFF_P2_M2][bufferOffset + x];
            buf[8] = buffers[ADIFF_P3_M3][bufferOffset + x];

            int minIndex = 4;
            for (int i = 0; i < 9; ++i) {
                if (buf[i] < buf[minIndex])
                    minIndex = i;
            }

            if (minIndex == 4 || buf[minIndex] >= aaf) {
                dstpn[x] = (currLine + nextLine) / 2;
            } else if (minIndex == 0) {
                dstpn[x] = (currLineM3 + nextLineP3) / 2;
            } else if (minIndex == 1) {
                dstpn[x] = (currLineM2 + nextLineP2) / 2;
            } else if (minIndex == 2) {
                dstpn[x] = (currLineM1 + nextLineP1) / 2;
            } else if (minIndex == 3) {
                dstpn[x] = (forwardSangNom1 + forwardSangNom2) / 2;
            } else if (minIndex == 5) {
                dstpn[x] = (backwardSangNom1 + backwardSangNom2) / 2;
            } else if (minIndex == 6) {
                dstpn[x] = (currLineP1 + nextLineM1) / 2;
            } else if (minIndex == 7) {
                dstpn[x] = (currLineP2 + nextLineM2) / 2;
            } else if (minIndex == 8) {
                dstpn[x] = (currLineP3 + nextLineM3) / 2;
            }
        }

        srcp += srcStride * 2;
        srcpn2 += srcStride * 2;
        dstpn += dstStride * 2;
        bufferOffset += bufferStride;
    }
}

template <typename T, typename IType>
static inline void sangnom_c(const T *srcp, const int srcStride, T *dstp, const int dstStride, const int w, const int h, SangNomData *d, int plane, T *buffers[TOTAL_BUFFERS], IType *bufferTemp)
{
    copyField<T>(srcp, srcStride, dstp, dstStride, w, h, d);

    prepareBuffers_c<T, IType>(srcp + d->offset * srcStride, srcStride, w, h, d->bufferStride, buffers);

    for (int i = 0; i < TOTAL_BUFFERS; ++i)
        processBuffers_c<T, IType>(buffers[i], bufferTemp, d->bufferStride, d->bufferHeight, w);

    finalizePlane_c<T, IType>(srcp + d->offset * srcStride, srcStride, dstp + d->offset * dstStride, dstStride, w, h, d->bufferStride, d->aaf, buffers);
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
        void *buffers[TOTAL_BUFFERS];

        void *bufferTemp;

        if (d->vi->format->sampleType == stInteger) {
            if (d->vi->format->bitsPerSample == 8) {
                bufferPool = vs_aligned_malloc<void>(sizeof(uint8_t) * d->bufferStride * (d->bufferHeight + 1) * TOTAL_BUFFERS, alignment);
                // separate bufferpool to multiple pieces
                for (int i = 0; i < TOTAL_BUFFERS; ++i)
                    buffers[i] = reinterpret_cast<uint8_t*>(bufferPool) + i * d->bufferStride * (d->bufferHeight + 1);
                bufferTemp = vs_aligned_malloc<void>(sizeof(int16_t) * d->bufferStride * (d->bufferHeight + 1), alignment);
            } else {
                bufferPool = vs_aligned_malloc<void>(sizeof(uint16_t) * d->bufferStride * (d->bufferHeight + 1) * TOTAL_BUFFERS, alignment);
                // separate bufferpool to multiple pieces
                for (int i = 0; i < TOTAL_BUFFERS; ++i)
                    buffers[i] = reinterpret_cast<uint16_t*>(bufferPool) + i * d->bufferStride * (d->bufferHeight + 1);
                bufferTemp = vs_aligned_malloc<void>(sizeof(int32_t) * d->bufferStride * (d->bufferHeight + 1), alignment);
            }
        } else {
            bufferPool = vs_aligned_malloc<void>(sizeof(float) * d->bufferStride * (d->bufferHeight + 1) * TOTAL_BUFFERS, alignment);
            // separate bufferpool to multiple pieces
            for (int i = 0; i < TOTAL_BUFFERS; ++i)
                buffers[i] = reinterpret_cast<float*>(bufferPool) + i * d->bufferStride * (d->bufferHeight + 1);
            bufferTemp = vs_aligned_malloc<void>(sizeof(float) * d->bufferStride * (d->bufferHeight + 1), alignment);
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
                    sangnom_c<uint8_t, int16_t>(srcp, srcStride, dstp, dstStride, width, height, d, plane, reinterpret_cast<uint8_t**>(buffers), reinterpret_cast<int16_t*>(bufferTemp));
                else
                    sangnom_c<uint16_t, int32_t>(reinterpret_cast<const uint16_t*>(srcp), srcStride, reinterpret_cast<uint16_t*>(dstp), dstStride, width, height, d, plane, reinterpret_cast<uint16_t**>(buffers), reinterpret_cast<int32_t*>(bufferTemp));
            } else {
                sangnom_c<float, float>(reinterpret_cast<const float*>(srcp), srcStride, reinterpret_cast<float*>(dstp), dstStride, width, height, d, plane, reinterpret_cast<float**>(buffers), reinterpret_cast<float*>(bufferTemp));
            }
        }

        vs_aligned_free(bufferTemp);
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

        d->order = vsapi->propGetInt(in, "order", 0, &err);
        if (err)
            d->order = VapourSynthFieldBasedToSangNomOrder(vsapi->propGetInt(in, "_FieldBased", 0, &err));
        if (err)
            d->order = SNOT_SFR_KT;

        if (d->order < 0  || d->order > 2)
            throw std::string("order must be 0 ... 2");

        d->aa = vsapi->propGetInt(in, "aa", 0, &err);
        if (err) d->aa = 48;

        if (d->aa < 0 || d->aa > 128)
            throw std::string("aa must be 0 ... 128");
        // tweak aa value for different format
        if (d->vi->format->sampleType == stInteger)
            d->aaf = (d->aa * 21.0 / 16.0) * (1 << (d->vi->format->bitsPerSample - 8));
        else
            d->aaf = (d->aa * 21.0 / 16.0) / 256.0;


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
