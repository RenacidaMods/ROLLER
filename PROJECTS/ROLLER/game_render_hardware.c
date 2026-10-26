#include "game_render_hardware.h"
#include "game_render.h"
#include "roller.h"
#include "scene_render_gpu.h"
#include "carplans.h"    /* CarDesigns[], tPolygon, tCarDesign, car_flat_remap[] */
#include "car.h"         /* car_texmap[], CarBox, CarPol, team_col, driver_names, CarBox */
#include "graphics.h"    /* num_textures[] */
#include "3d.h"          /* cartex_vga[], Car[], localdata[], CarBox, g_pGameRenderer, names_on, winw/winh, viewx/y/z, xbase/ybase, scr_size, VIEWDIST, tcos/tsin, mirror, intro */
#include "transfrm.h"    /* vk1-vk9 */
#include "drawtrk3.h"    /* NamesLeft */
#include "func2.h"       /* mini_prt_string, language_buffer, screen_pointer, ascii_conv3 */
#include "func3.h"       /* tBlockHeader */
#include "frontend.h"    /* human_control, racers */
#include "moving.h"      /* replaytype */
#include "control.h"     /* getgroundz(), getroadz(), calculateseparatedcoordinatesystem() — real terrain queries, chunk-local coords */
#include "function.h"    /* getbankz() */
#include "loadtrak.h"    /* TrackInfo[], tTrackInfo — shoulder widths, surface types, roof height */

#include <SDL3/SDL.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

extern tColor palette[256];

static inline int hw_d2i(double d) {
    if (d >= (double)2147483647) return 2147483647;
    if (d <= (double)(-2147483647-1)) return (-2147483647-1);
    return (int)d;
}

#define GAME_RENDER_HW_NUM_CARS      16
#define GAME_RENDER_HW_MAX_CAR_DRAWS 32  /* 16 cars × up to 2 draws each */
#define GRHW_ANIM_FRAMES             5   /* wheel animation frames: 0-3 rolling, 4 = high-speed
                                            * (control.c sets byWheelAnimationFrame=4 at >=300 speed) */

/* --------------------------------------------------------------------------
 * Vertex layout matching SceneGPUMeshVertex in menu_render_gpu.c (same
 * shaders used for car pipeline in scene_render_gpu.c).
 * -------------------------------------------------------------------------- */
typedef struct {
    float position[3];
    float uv[2];
    float color[4];
    float normal[3];   /* smoothed corner normal (car shine), object space */
    float edge[2];     /* position across the panel, 0..1 each way (car shine bevel) */
    float bevel[3];    /* fully blended corner normal, used near panel edges */
} GRHWMeshVertex;

/* --------------------------------------------------------------------------
 * Per-car mesh cache entry
 * -------------------------------------------------------------------------- */
typedef struct {
    SDL_GPUBuffer  *vertexBufs[GRHW_ANIM_FRAMES]; /* one per wheel animation frame */
    SDL_GPUBuffer  *indexBuf;                      /* shared across all frames */
    SDL_GPUTexture *atlas;
    int             indexCount;
    int             bodyIndexCount;
    bool            built;
    bool            advancedCars; /* TEX_OFF_ADVANCED_CARS state when built */
    /* Car[carIdx].byCarDesignIdx when built. A race never changes a slot's
     * design, so this cache used to be correct by circumstance; the editor's
     * test car (E3A-S6) reuses one slot for every design the user picks, and
     * without this the mesh stayed whichever design was chosen first. */
    int             designIdx;
} GRHWCarMesh;

struct GameRendererHardware {
    SDL_GPUDevice *device;
    GRHWCarMesh    meshes[GAME_RENDER_HW_NUM_CARS];
};

/* Smoothed scrX/scrY for each car's name tag (EMA filter, -1 = uninitialised).
 * Keyed by [viewSlot][carIdx]: in 2-player mode the same car's tag is drawn
 * once per player's view (from that player's own camera), so a single
 * carIdx-only array would blend two unrelated camera-relative computations
 * together via the EMA -- see [[project_car_name_tags]]. */
#define GAME_RENDER_HW_MAX_VIEW_SLOTS 2
static int s_tagScrX[GAME_RENDER_HW_MAX_VIEW_SLOTS][GAME_RENDER_HW_NUM_CARS] = {
    { -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 },
    { -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 }
};
static int s_tagScrY[GAME_RENDER_HW_MAX_VIEW_SLOTS][GAME_RENDER_HW_NUM_CARS] = {
    { -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 },
    { -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 }
};

void game_render_hw_reset_name_tags(void)
{
    for (int slot = 0; slot < GAME_RENDER_HW_MAX_VIEW_SLOTS; slot++) {
        for (int car = 0; car < GAME_RENDER_HW_NUM_CARS; car++) {
            s_tagScrX[slot][car] = -1;
            s_tagScrY[slot][car] = -1;
        }
    }
}

/* Cached per-car name-tag label texture, so real GPU billboards (depth-tested
 * against the actual scene, unlike the old flat scrbuf-HUD text draw) don't
 * need to rasterise glyphs every frame -- only when the label string itself
 * changes (name/position rarely change frame to frame). Composed from the
 * exact same glyph data mini_prt_string used (ascii_conv3 + rev_vga[0]), so
 * it looks pixel-identical to the old SW-drawn text. */
#define GAME_RENDER_HW_TAG_LABEL_MAX 32
typedef struct {
    char            label[GAME_RENDER_HW_TAG_LABEL_MAX];
    SDL_GPUTexture *tex;
    int             texW, texH;
} GRHWTagLabelCache;
static GRHWTagLabelCache s_tagLabelCache[GAME_RENDER_HW_NUM_CARS];

/* Composes `label` into cache->tex if it differs from what's already cached
 * for this car. Returns false (cache left as-is) if the label is empty or
 * the atlas glyph data can't produce a non-empty bitmap -- caller should
 * skip drawing the tag text in that case. */
static bool ensure_tag_label_texture(SDL_GPUDevice *device, const char *label,
                                     GRHWTagLabelCache *cache)
{
    if (cache->tex && strncmp(cache->label, label, GAME_RENDER_HW_TAG_LABEL_MAX) == 0)
        return true;

    int totalW = 0, maxH = 0;
    for (const char *s = label; *s; s++) {
        uint8 ci = (uint8)ascii_conv3[(uint8)*s];
        if (ci == 255) { totalW += 4; continue; }
        const tBlockHeader *g = &rev_vga[0][ci];
        totalW += g->iWidth;
        if (g->iHeight > maxH) maxH = g->iHeight;
    }
    if (totalW <= 0 || maxH <= 0) return false;

    uint8 *rgba = calloc((size_t)totalW * (size_t)maxH, 4);
    if (!rgba) return false;

    int penX = 0;
    for (const char *s = label; *s; s++) {
        uint8 ci = (uint8)ascii_conv3[(uint8)*s];
        if (ci == 255) { penX += 4; continue; }
        const tBlockHeader *g = &rev_vga[0][ci];
        const uint8 *bitmap = (const uint8 *)rev_vga[0] + g->iDataOffset;
        for (int row = 0; row < g->iHeight; row++) {
            for (int col = 0; col < g->iWidth; col++) {
                uint8 idx = bitmap[row * g->iWidth + col];
                if (!idx) continue; /* transparent, matches SW's "0 = don't draw" */
                const tColor *c = &palette[idx];
                uint8 *px = &rgba[((size_t)row * totalW + (penX + col)) * 4];
                px[0] = (uint8)(c->byR * 255 / 63);
                px[1] = (uint8)(c->byG * 255 / 63);
                px[2] = (uint8)(c->byB * 255 / 63);
                px[3] = 255;
            }
        }
        penX += g->iWidth;
    }

    SDL_GPUTexture *newTex = scene_render_gpu_upload_rgba(device, rgba, totalW, maxH, false);
    free(rgba);
    if (!newTex) return false;

    if (cache->tex) SDL_ReleaseGPUTexture(device, cache->tex);
    cache->tex  = newTex;
    cache->texW = totalW;
    cache->texH = maxH;
    strncpy(cache->label, label, GAME_RENDER_HW_TAG_LABEL_MAX - 1);
    cache->label[GAME_RENDER_HW_TAG_LABEL_MAX - 1] = 0;
    return true;
}

/* ==========================================================================
 * Car texture atlas
 * ========================================================================== */
/* Shared by the race renderer and the car-select screen (menu_render_gpu.c).
 * Layout: the car's 64x64 tiles 4 per row, plus a white strip at the bottom
 * that flat-coloured polygons sample. *outAtlasH is that layout's height in
 * 64-pixel units (numRows * 64 + 1), which the UV maths uses.
 *
 * With HD Textures on and ../hd_textures/<car file>/<tile>.png present, the
 * sheet is built k = hdSize / 64 times larger in both directions: HD tiles
 * as they are, other tiles scaled up nearest-neighbour, and the white strip
 * k rows tall, so every UV fraction (and *outAtlasH) is unchanged. The HD
 * sheet gets mipmaps, since a 256-pixel tile on a car is usually drawn
 * smaller than that and would shimmer without them.
 *
 * A sheet made only of HD tiles doesn't depend on the palette, so it is
 * shared: a race has 16 cars but at most 8 designs, and each HD sheet costs
 * dozens of PNG decodes. Release with game_render_hw_release_car_atlas. */
#define GRHW_CAR_ATLAS_SHARE_SLOTS 32

typedef struct {
    SDL_GPUDevice  *dev;
    SDL_GPUTexture *tex;
    int             refs, numTiles, atlasH, scale;
} GRHWCarAtlasShare;

static GRHWCarAtlasShare s_carAtlasShare[GRHW_CAR_ATLAS_SHARE_SLOTS];   /* by texture bank */

SDL_GPUTexture *game_render_hw_build_car_atlas(SDL_GPUDevice *dev, int carIdx,
                                               const tColor *pal,
                                               int *outNumTiles, int *outAtlasH,
                                               int *outScale)
{
    if (outScale) *outScale = 1;
    int texSlot = car_texmap[carIdx];
    if (texSlot <= 0) return NULL;
    uint8 *texData = cartex_vga[texSlot - 1];
    if (!texData) return NULL;
    int nTiles = num_textures[texSlot - 1];
    if (nTiles <= 0) return NULL;

    int numRows = (nTiles + 3) / 4;

    GRHWCarAtlasShare *share = texSlot < GRHW_CAR_ATLAS_SHARE_SLOTS ? &s_carAtlasShare[texSlot] : NULL;
    if (share && share->refs > 0 && share->dev == dev && g_bHDTextures) {
        share->refs++;
        if (outNumTiles) *outNumTiles = share->numTiles;
        if (outAtlasH)   *outAtlasH  = share->atlasH;
        if (outScale)    *outScale   = share->scale;
        return share->tex;
    }

    int hdSize = 0, hdCount = 0;
    uint8 **hdTiles = scene_render_gpu_load_hd_bank_tiles(texSlot, nTiles, 64, &hdSize);
    for (int t = 0; hdTiles && t < nTiles; t++)
        if (hdTiles[t]) hdCount++;
    int k = hdCount > 0 ? hdSize / 64 : 1;
    int tilePx  = 64 * k;
    int atlasW  = 256 * k;
    int atlasH  = (numRows * 64 + 1) * k;   /* white strip: k rows */

    uint8 *rgba = calloc((size_t)atlasW * atlasH * 4, 1);
    if (!rgba) {
        for (int t = 0; hdTiles && t < nTiles; t++) free(hdTiles[t]);
        free(hdTiles);
        return NULL;
    }

    for (int t = 0; t < nTiles; t++) {
        int col = t % 4, row = t / 4;
        const uint8 *hd = hdTiles ? hdTiles[t] : NULL;
        for (int y = 0; y < tilePx; y++) {
            uint8 *dstRow = rgba + ((size_t)(row * tilePx + y) * atlasW + col * tilePx) * 4;
            if (hd) {
                memcpy(dstRow, hd + (size_t)y * tilePx * 4, (size_t)tilePx * 4);
                continue;
            }
            for (int x = 0; x < tilePx; x++) {
                int srcOff = (row * 64 + y / k) * 256 + col * 64 + x / k;
                uint8 palIdx = texData[srcOff];
                const tColor *c = &pal[palIdx];
                dstRow[x * 4 + 0] = (uint8)(c->byR * 255 / 63);
                dstRow[x * 4 + 1] = (uint8)(c->byG * 255 / 63);
                dstRow[x * 4 + 2] = (uint8)(c->byB * 255 / 63);
                dstRow[x * 4 + 3] = palIdx ? 255 : 0;
            }
        }
    }
    for (int y = atlasH - k; y < atlasH; y++) {
        for (int x = 0; x < atlasW; x++) {
            int off = (y * atlasW + x) * 4;
            rgba[off] = rgba[off+1] = rgba[off+2] = rgba[off+3] = 255;
        }
    }

    for (int t = 0; hdTiles && t < nTiles; t++) free(hdTiles[t]);
    free(hdTiles);

    SDL_GPUTexture *tex = scene_render_gpu_upload_rgba(dev, rgba, atlasW, atlasH, k > 1);
    free(rgba);
    if (tex && share && share->refs == 0 && hdCount == nTiles)
        *share = (GRHWCarAtlasShare){dev, tex, 1, nTiles, numRows * 64 + 1, k};
    if (outNumTiles) *outNumTiles = nTiles;
    if (outAtlasH)   *outAtlasH  = numRows * 64 + 1;
    if (outScale && tex) *outScale = k;
    return tex;
}

void game_render_hw_release_car_atlas(SDL_GPUDevice *dev, SDL_GPUTexture *tex)
{
    if (!tex) return;
    for (int i = 0; i < GRHW_CAR_ATLAS_SHARE_SLOTS; i++) {
        GRHWCarAtlasShare *share = &s_carAtlasShare[i];
        if (share->refs > 0 && share->tex == tex) {
            if (--share->refs == 0) {
                SDL_ReleaseGPUTexture(share->dev, tex);
                *share = (GRHWCarAtlasShare){0};
            }
            return;
        }
    }
    SDL_ReleaseGPUTexture(dev, tex);
}

void game_render_hw_car_corner_normals(int designIdx, float (*outSmooth)[4][3],
                                       float (*outBevel)[4][3])
{
    if (designIdx < 0 || designIdx > CAR_DESIGN_DEATH) return;
    tCarDesign *design = &CarDesigns[designIdx];
    int numPols = design->byNumPols, numCoords = design->byNumCoords;
    tVec3 *coords = design->pCoords;
    tPolygon *pols = design->pPols;
    if (!coords || !pols || numPols <= 0 || numCoords <= 0) return;
    memset(outSmooth, 0, (size_t)numPols * sizeof(*outSmooth));
    memset(outBevel, 0, (size_t)numPols * sizeof(*outBevel));

    /* Corners at the same position share a vertex, even under another index. */
    int canon[256];
    for (int i = 0; i < numCoords; i++) {
        canon[i] = i;
        for (int j = 0; j < i; j++) {
            if (fabsf(coords[i].fX - coords[j].fX) < 0.5f && fabsf(coords[i].fY - coords[j].fY) < 0.5f
                    && fabsf(coords[i].fZ - coords[j].fZ) < 0.5f) {
                canon[i] = canon[j];
                break;
            }
        }
    }
    float centre[3] = {0, 0, 0};
    for (int i = 0; i < numCoords; i++) {
        centre[0] += coords[i].fX; centre[1] += coords[i].fY; centre[2] += coords[i].fZ;
    }
    for (int c = 0; c < 3; c++) centre[c] /= (float)numCoords;

    /* Face normals (Newell's method, so repeated corners are fine), length =
     * twice the area, pointing away from the car's centre; the car models'
     * windings aren't consistent, but the cars are close enough to convex. */
    float (*fn)[3] = calloc((size_t)numPols, sizeof(*fn));
    float (*fu)[3] = calloc((size_t)numPols, sizeof(*fu));
    int   (*cv)[4] = calloc((size_t)numPols, sizeof(*cv));
    if (!fn || !fu || !cv) { free(fn); free(fu); free(cv); return; }
    for (int p = 0; p < numPols; p++) {
        float fc[3] = {0, 0, 0}, n[3] = {0, 0, 0};
        for (int k = 0; k < 4; k++) {
            int a = pols[p].verts[k], b = pols[p].verts[(k + 1) & 3];
            if (a >= numCoords) a = 0;
            if (b >= numCoords) b = 0;
            cv[p][k] = canon[a];
            const tVec3 *A = &coords[a], *B = &coords[b];
            n[0] += (A->fY - B->fY) * (A->fZ + B->fZ);
            n[1] += (A->fZ - B->fZ) * (A->fX + B->fX);
            n[2] += (A->fX - B->fX) * (A->fY + B->fY);
            fc[0] += A->fX * 0.25f; fc[1] += A->fY * 0.25f; fc[2] += A->fZ * 0.25f;
        }
        float out_dir = n[0] * (fc[0] - centre[0]) + n[1] * (fc[1] - centre[1]) + n[2] * (fc[2] - centre[2]);
        if (out_dir < 0.0f) { n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }
        float len = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (pols[p].uiTex & SURFACE_FLAG_SKIP_RENDER) len = 0.0f;   /* not drawn: no say */
        for (int c = 0; c < 3; c++) {
            fn[p][c] = len > 1e-6f ? n[c] : 0.0f;
            fu[p][c] = len > 1e-6f ? n[c] / len : 0.0f;
        }
    }

    /* Each corner: an area-weighted sum of the faces meeting there.
     * Smooth: each neighbour weighted by the cosine of its angle to this
     * face (a sliding scale: 30 deg 0.87, 60 deg 0.5, 80 deg 0.17, 90 deg
     * and beyond 0). Bevel: every neighbour up to 150 deg at full weight,
     * so even sharp edges get a rounded normal for the bevel strip. */
    float cosBevel = cosf(150.0f * 3.14159265f / 180.0f);
    for (int p = 0; p < numPols; p++) {
        for (int k = 0; k < 4; k++) {
            float sm[3] = {0, 0, 0}, bv[3] = {0, 0, 0};
            for (int q = 0; q < numPols; q++) {
                if (fu[q][0] == 0.0f && fu[q][1] == 0.0f && fu[q][2] == 0.0f) continue;
                if (cv[q][0] != cv[p][k] && cv[q][1] != cv[p][k]
                        && cv[q][2] != cv[p][k] && cv[q][3] != cv[p][k]) continue;
                float d = fu[p][0] * fu[q][0] + fu[p][1] * fu[q][1] + fu[p][2] * fu[q][2];
                float w = q == p ? 1.0f : (d > 0.0f ? d : 0.0f);
                for (int c = 0; c < 3; c++) {
                    sm[c] += fn[q][c] * w;
                    if (d >= cosBevel) bv[c] += fn[q][c];
                }
            }
            float ls = sqrtf(sm[0] * sm[0] + sm[1] * sm[1] + sm[2] * sm[2]);
            float lb = sqrtf(bv[0] * bv[0] + bv[1] * bv[1] + bv[2] * bv[2]);
            for (int c = 0; c < 3; c++) {
                outSmooth[p][k][c] = ls > 1e-6f ? sm[c] / ls : fu[p][c];
                outBevel[p][k][c]  = lb > 1e-6f ? bv[c] / lb : outSmooth[p][k][c];
            }
        }
    }
    free(fn); free(fu); free(cv);
}

void game_render_hw_write_car_quad(float *verts, uint32 *indices, uint32 baseVert,
                                   const float pos[4][3], const float uv[4][2],
                                   const float color[4], const float nrm[4][3],
                                   const float bev[4][3], bool reverseWinding)
{
    const int n = GAME_RENDER_HW_QUAD_GRID;
    for (int j = 0; j <= n; j++) {
        float t = (float)j / (float)n;
        for (int i = 0; i <= n; i++) {
            float s = (float)i / (float)n;
            float w[4] = { (1.0f - s) * (1.0f - t), s * (1.0f - t), s * t, (1.0f - s) * t };
            float *v = verts + (size_t)(j * (n + 1) + i) * GAME_RENDER_HW_CAR_VERTEX_FLOATS;
            for (int c = 0; c < 3; c++)
                v[c] = w[0] * pos[0][c] + w[1] * pos[1][c] + w[2] * pos[2][c] + w[3] * pos[3][c];
            for (int c = 0; c < 2; c++)
                v[3 + c] = w[0] * uv[0][c] + w[1] * uv[1][c] + w[2] * uv[2][c] + w[3] * uv[3][c];
            for (int c = 0; c < 4; c++)
                v[5 + c] = color[c];
            float nl = 0.0f;
            for (int c = 0; c < 3; c++) {
                v[9 + c] = nrm ? w[0] * nrm[0][c] + w[1] * nrm[1][c] + w[2] * nrm[2][c] + w[3] * nrm[3][c] : 0.0f;
                nl += v[9 + c] * v[9 + c];
            }
            if (nl > 1e-12f) {
                nl = 1.0f / sqrtf(nl);
                v[9] *= nl; v[10] *= nl; v[11] *= nl;
            }
            v[12] = s;
            v[13] = t;
            float bl = 0.0f;
            for (int c = 0; c < 3; c++) {
                v[14 + c] = bev ? w[0] * bev[0][c] + w[1] * bev[1][c] + w[2] * bev[2][c] + w[3] * bev[3][c]
                                : v[9 + c];
                bl += v[14 + c] * v[14 + c];
            }
            if (bl > 1e-12f) {
                bl = 1.0f / sqrtf(bl);
                v[14] *= bl; v[15] *= bl; v[16] *= bl;
            }
        }
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            uint32 a = baseVert + (uint32)(j * (n + 1) + i);
            uint32 b = a + 1, c = a + (uint32)n + 2, d = a + (uint32)n + 1;
            if (reverseWinding) { *indices++ = a; *indices++ = c; *indices++ = b;
                                  *indices++ = a; *indices++ = d; *indices++ = c; }
            else                { *indices++ = a; *indices++ = b; *indices++ = c;
                                  *indices++ = a; *indices++ = c; *indices++ = d; }
        }
    }
}

_Static_assert(sizeof(GRHWMeshVertex) == GAME_RENDER_HW_CAR_VERTEX_FLOATS * sizeof(float),
               "game_render_hw_write_car_quad writes GAME_RENDER_HW_CAR_VERTEX_FLOATS-float vertices");

/* Resolve animated polygon texture for a given animation frame.
 * Wheel-animation entries have animIdx 0-3 and advance with anim_frame.
 * Body-variant entries (animIdx >= 4) use bodyVariant = carIdx & 1, matching
 * SW's iAnimationOffset / 4 = carIdx & 1 scheme (car.c:1881). */
static uint32 resolve_car_tex(uint32 rawTex, tAnimation *pAnms, int animFrame, int bodyVariant)
{
    if ((rawTex & CAR_FLAG_ANMS_LOOKUP) && pAnms) {
        uint8 animIdx = (uint8)rawTex;
        tAnimation *anim = &pAnms[animIdx];
        int f = (animIdx < 4) ? animFrame : bodyVariant;
        if (anim->uiCount > 0 && f >= (int)anim->uiCount) f = 0;
        return anim->framesAy[f];
    }
    return rawTex;
}

/* Build vertex/index buffers for one car design. */
static bool build_car_mesh(SDL_GPUDevice *dev, int carIdx,
                            const tColor *pal, GRHWCarMesh *out)
{
    int numTiles = 0, atlasH = 0, atlasScale = 1;
    out->atlas = game_render_hw_build_car_atlas(dev, carIdx, pal, &numTiles, &atlasH, &atlasScale);
    bool hasAtlas = (out->atlas != NULL && numTiles > 0);
    float fAtlasH = hasAtlas ? (float)atlasH : 1.0f;
    /* HD sheets are sampled with filtering and mipmaps, which reach past a
     * tile's edge into its neighbour on the sheet (a square outline round
     * the see-through wheel corners). Pull each tile's UVs 2 HD texels in;
     * the original-size sheet keeps its exact mapping. */
    float insetU = atlasScale > 1 ? 2.0f / (256.0f * atlasScale) : 0.0f;
    float insetV = atlasScale > 1 ? 2.0f / (fAtlasH * atlasScale) : 0.0f;

    int designIdx = (carIdx >= 0 && carIdx < 16) ? (int)Car[carIdx].byCarDesignIdx : 0;
    if (designIdx < 0 || designIdx > CAR_DESIGN_DEATH) designIdx = 0;

    if (!out->atlas) {
        uint8 white[4] = {255, 255, 255, 255};
        out->atlas = scene_render_gpu_upload_rgba(dev, white, 1, 1, false);
    }
    float whiteU = hasAtlas ? 0.5f / 256.0f : 0.5f;
    float whiteV = hasAtlas ? (atlasH - 0.5f) / fAtlasH : 0.5f;

    tCarDesign     *design   = &CarDesigns[designIdx];
    int             numPols  = design->byNumPols;
    int             numCoords = design->byNumCoords;
    tVec3          *coords   = design->pCoords;
    tPolygon       *pols     = design->pPols;
    tAnimation     *pAnms    = design->pAnms;
    tCarColorRemap *remap    = &car_flat_remap[designIdx];

    if (!coords || !pols || numPols == 0) return false;

    /* *2 for front + back faces per polygon (pBacks[]); each face is a grid
     * (game_render_hw_write_car_quad), plus the 4-vertex shadow quad */
    GRHWMeshVertex *vertices = calloc((size_t)numPols * 2 * GAME_RENDER_HW_QUAD_VERTS + 4, sizeof(GRHWMeshVertex));
    uint32         *indices  = calloc((size_t)numPols * 2 * GAME_RENDER_HW_QUAD_INDICES + 6, sizeof(uint32));
    if (!vertices || !indices) {
        free(vertices); free(indices);
        if (out->atlas) { game_render_hw_release_car_atlas(dev, out->atlas); out->atlas = NULL; }
        return false;
    }

    /* Smoothed corner normals for car shine (shared by every animation frame). */
    float (*cornerN)[4][3] = calloc((size_t)numPols, sizeof(*cornerN));
    float (*cornerB)[4][3] = calloc((size_t)numPols, sizeof(*cornerB));
    if (cornerN && cornerB)
        game_render_hw_car_corner_normals(designIdx, cornerN, cornerB);
    else {
        free(cornerN); free(cornerB);
        cornerN = NULL; cornerB = NULL;
    }

    /* Build one vertex buffer per animation frame; share one index buffer (geometry is
     * identical across frames — only UV coordinates change for animated wheel polys). */
    for (int animFrame = 0; animFrame < GRHW_ANIM_FRAMES; animFrame++) {
    int vertCount = 0, idxCount = 0;

    for (int p = 0; p < numPols; p++) {
        uint32 tex = pols[p].uiTex;
        if (tex & SURFACE_FLAG_SKIP_RENDER) continue;
        tex = resolve_car_tex(tex, pAnms, animFrame, carIdx & 1);

        bool isTextured = hasAtlas && (tex & SURFACE_FLAG_APPLY_TEXTURE) &&
                          (uint8)tex < (uint8)numTiles;

        float u0, u1, v0, v1, cr, cg, cb, ca;

        if (isTextured) {
            uint8 tileIdx = (uint8)tex;
            int col = tileIdx % 4, row = tileIdx / 4;
            u0 = (col * 64.0f) / 256.0f;
            u1 = ((col + 1) * 64.0f) / 256.0f;
            v0 = (row * 64.0f) / fAtlasH;
            v1 = ((row + 1) * 64.0f) / fAtlasH;
            u0 += insetU; u1 -= insetU; v0 += insetV; v1 -= insetV;
            if (tex & SURFACE_FLAG_FLIP_HORIZ) { float t = u0; u0 = u1; u1 = t; }
            if (tex & SURFACE_FLAG_FLIP_VERT)  { float t = v0; v0 = v1; v1 = t; }
            cr = cg = cb = ca = 1.0f;
        } else {
            uint8 colorIdx = (uint8)tex;
            if ((textures_off & TEX_OFF_ADVANCED_CARS) != 0 &&
                !(tex & SURFACE_FLAG_APPLY_TEXTURE) &&
                remap->uiColorFrom != 0xFFFFFFFF &&
                colorIdx == (uint8)remap->uiColorFrom)
                colorIdx = (uint8)remap->uiColorTo;
            const tColor *c = &pal[colorIdx];
            cr = (c->byR * 255.0f / 63.0f) / 255.0f;
            cg = (c->byG * 255.0f / 63.0f) / 255.0f;
            cb = (c->byB * 255.0f / 63.0f) / 255.0f;
            ca = 1.0f;
            u0 = u1 = whiteU;
            v0 = v1 = whiteV;
        }

        int baseVert = vertCount;

        /* Swap u0/u1 to compensate for the scene renderer's horizontal mirror.
         * SURFACE_FLAG_FLIP_HORIZ pre-swaps, so together this produces the
         * intended horizontal flip. */
        float uvs[4][2] = {
            { u1, v0 }, { u0, v0 }, { u0, v1 }, { u1, v1 }
        };

        float pos[4][3], col[4] = { cr, cg, cb, ca };
        for (int v = 0; v < 4; v++) {
            uint8 vi = pols[p].verts[v];
            if (vi >= (uint8)numCoords) vi = 0;
            pos[v][0] = coords[vi].fX;
            pos[v][1] = coords[vi].fY;
            pos[v][2] = coords[vi].fZ;
        }
        game_render_hw_write_car_quad(vertices[baseVert].position, &indices[idxCount],
                                      (uint32)baseVert, pos, uvs, col,
                                      cornerN ? (const float (*)[3])cornerN[p] : NULL,
                                      cornerB ? (const float (*)[3])cornerB[p] : NULL, false);
        vertCount += GAME_RENDER_HW_QUAD_VERTS;
        idxCount  += GAME_RENDER_HW_QUAD_INDICES;
    }

    /* Back-face polygons (reversed winding), following SW renderer logic (car.c:1647):
       - FLIP_BACKFACE + BACK: front=uiTex, back=pBacks[p]  (explicit back texture)
       - FLIP_BACKFACE only:   both sides use the same resolved texture (double-sided)
       - BACK only or neither: front only — no reversed polygon needed (pBacks unused in SW) */
    for (int p = 0; p < numPols; p++) {
        uint32 rawFront = pols[p].uiTex;
        if (rawFront & SURFACE_FLAG_SKIP_RENDER) continue;
        if (!(rawFront & SURFACE_FLAG_FLIP_BACKFACE)) continue;

        uint32 tex;
        if (rawFront & SURFACE_FLAG_BACK) {
            if (!design->pBacks) continue;
            tex = design->pBacks[p];
        } else {
            tex = resolve_car_tex(rawFront, pAnms, animFrame, carIdx & 1);
        }
        if (tex & SURFACE_FLAG_SKIP_RENDER) continue;

        bool isTexturedBack = hasAtlas && (tex & SURFACE_FLAG_APPLY_TEXTURE) &&
                              (uint8)tex < (uint8)numTiles;

        float u0b, u1b, v0b, v1b, crb, cgb, cbb, cab;

        if (isTexturedBack) {
            uint8 tileIdx = (uint8)tex;
            int col = tileIdx % 4, row = tileIdx / 4;
            u0b = (col * 64.0f) / 256.0f;
            u1b = ((col + 1) * 64.0f) / 256.0f;
            v0b = (row * 64.0f) / fAtlasH;
            v1b = ((row + 1) * 64.0f) / fAtlasH;
            u0b += insetU; u1b -= insetU; v0b += insetV; v1b -= insetV;
            if (tex & SURFACE_FLAG_FLIP_HORIZ) { float t = u0b; u0b = u1b; u1b = t; }
            if (tex & SURFACE_FLAG_FLIP_VERT)  { float t = v0b; v0b = v1b; v1b = t; }
            crb = cgb = cbb = cab = 1.0f;
        } else {
            uint8 colorIdx = (uint8)tex;
            if ((textures_off & TEX_OFF_ADVANCED_CARS) != 0 &&
                !(tex & SURFACE_FLAG_APPLY_TEXTURE) &&
                remap->uiColorFrom != 0xFFFFFFFF &&
                colorIdx == (uint8)remap->uiColorFrom)
                colorIdx = (uint8)remap->uiColorTo;
            const tColor *c = &pal[colorIdx];
            crb = (c->byR * 255.0f / 63.0f) / 255.0f;
            cgb = (c->byG * 255.0f / 63.0f) / 255.0f;
            cbb = (c->byB * 255.0f / 63.0f) / 255.0f;
            cab = 1.0f;
            u0b = u1b = whiteU;
            v0b = v1b = whiteV;
        }

        int baseVert = vertCount;
        float uvsBack[4][2] = {
            { u1b, v0b }, { u0b, v0b }, { u0b, v1b }, { u1b, v1b }
        };
        float pos[4][3], col[4] = { crb, cgb, cbb, cab }, nrm[4][3] = {{0}}, bev[4][3] = {{0}};
        for (int v = 0; v < 4; v++) {
            uint8 vi = pols[p].verts[3 - v];  /* reversed vertex order */
            if (vi >= (uint8)numCoords) vi = 0;
            pos[v][0] = coords[vi].fX;
            pos[v][1] = coords[vi].fY;
            pos[v][2] = coords[vi].fZ;
            if (cornerN) memcpy(nrm[v], cornerN[p][3 - v], sizeof(nrm[v]));
            if (cornerB) memcpy(bev[v], cornerB[p][3 - v], sizeof(bev[v]));
        }
        game_render_hw_write_car_quad(vertices[baseVert].position, &indices[idxCount],
                                      (uint32)baseVert, pos, uvsBack, col, nrm, bev, false);
        vertCount += GAME_RENDER_HW_QUAD_VERTS;
        idxCount  += GAME_RENDER_HW_QUAD_INDICES;
    }

    if (animFrame == 0) out->bodyIndexCount = idxCount;

    /* Shadow quad: same X extents as the hitbox but narrower in Y to match
     * the SW renderer, which uses carpoint corners narrowed by 50 units on
     * each side (car.c:1149-1152, design index ≤ 7 only).
     * Use hitbox values directly — CarBaseX/Y are design-0-only globals and
     * may be 0 at first-draw time.  Hitbox is always valid (CalcCarSizes). */
    {
        const tVec3 *hb = CarBox.hitboxAy[designIdx];
        /* hb[0]=(front,right), hb[1]=(rear,right), hb[2]=(rear,left), hb[3]=(front,left) */
        float fYAdj = (designIdx <= 7) ? 50.0f : 0.0f;
        float fYLS  = hb[0].fY + fYAdj;   /* right edge, pull inward (+) */
        float fYHS  = hb[3].fY - fYAdj;   /* left edge, pull inward (−) */
        float fShadowZ = hb[0].fZ;
        float sx[4] = { hb[0].fX, hb[1].fX, hb[2].fX, hb[3].fX };
        float sy[4] = { fYLS,     fYLS,      fYHS,     fYHS     };
        int shadowBase = vertCount;
        for (int k = 0; k < 4; k++) {
            GRHWMeshVertex *mv = &vertices[vertCount++];
            mv->position[0] = sx[k];
            mv->position[1] = sy[k];
            mv->position[2] = fShadowZ;
            mv->uv[0] = whiteU; mv->uv[1] = whiteV;
            mv->color[0] = 0.0f; mv->color[1] = 0.0f;
            mv->color[2] = 0.0f; mv->color[3] = 0.5f;
        }
        indices[idxCount++] = (uint32)(shadowBase + 0);
        indices[idxCount++] = (uint32)(shadowBase + 1);
        indices[idxCount++] = (uint32)(shadowBase + 2);
        indices[idxCount++] = (uint32)(shadowBase + 0);
        indices[idxCount++] = (uint32)(shadowBase + 2);
        indices[idxCount++] = (uint32)(shadowBase + 3);
    }

    out->vertexBufs[animFrame] = scene_render_gpu_upload_buffer(dev, SDL_GPU_BUFFERUSAGE_VERTEX,
                                            vertices,
                                            (Uint32)(vertCount * (int)sizeof(GRHWMeshVertex)));
    if (animFrame == 0) {
        out->indexBuf   = scene_render_gpu_upload_buffer(dev, SDL_GPU_BUFFERUSAGE_INDEX,
                                            indices,
                                            (Uint32)(idxCount * (int)sizeof(uint32)));
        out->indexCount = idxCount;
    }
    } /* end animFrame loop */
    free(vertices);
    free(indices);
    free(cornerN);
    free(cornerB);

    if (!out->vertexBufs[0] || !out->indexBuf) {
        for (int f = 0; f < GRHW_ANIM_FRAMES; f++) {
            if (out->vertexBufs[f]) { SDL_ReleaseGPUBuffer(dev, out->vertexBufs[f]); out->vertexBufs[f] = NULL; }
        }
        if (out->indexBuf)  { SDL_ReleaseGPUBuffer(dev, out->indexBuf);  out->indexBuf  = NULL; }
        if (out->atlas)     { game_render_hw_release_car_atlas(dev, out->atlas); out->atlas = NULL; }
        return false;
    }
    return true;
}

/* ==========================================================================
 * Matrix helpers
 * ========================================================================== */

/* General 4x4 inverse (column-major in, column-major out); false if singular. */
static bool mat4_invert(float out[16], const float m[16])
{
    float inv[16];
    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];
    float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (fabsf(det) < 1e-20f) return false;
    for (int i = 0; i < 16; i++) out[i] = inv[i] / det;
    return true;
}

/* Car shine inputs for one car body: the camera position, world up (game
 * Z) and world X axis in the car's own coordinates, from its world matrix
 * and the view's view-projection. The camera centre is the point the
 * projection sends to clip (0, 0, z, 0), so it is inverse(vp) * (0, 0, 1, 0).
 * All zero (no shine) if a matrix can't be inverted. */
static void car_shine_inputs(float shine[12], const float vp[16], const float world[16])
{
    float ivp[16], iw[16];
    memset(shine, 0, 12 * sizeof(float));
    if (!mat4_invert(ivp, vp) || !mat4_invert(iw, world)) return;
    float cw = ivp[11];   /* column 2 of inverse(vp) is inverse(vp) * (0,0,1,0) */
    if (fabsf(cw) < 1e-20f) return;
    float cam[3] = { ivp[8] / cw, ivp[9] / cw, ivp[10] / cw };
    for (int r = 0; r < 3; r++) {
        shine[r]     = iw[r] * cam[0] + iw[4 + r] * cam[1] + iw[8 + r] * cam[2] + iw[12 + r];
        shine[4 + r] = iw[8 + r];   /* inverse(world) * (0, 0, 1, 0) */
        shine[8 + r] = iw[r];       /* inverse(world) * (1, 0, 0, 0) */
    }
}

/* 4×4 column-major matrix multiply: out = A * B */
static void mat4_mul(float out[16], const float A[16], const float B[16])
{
    for (int col = 0; col < 4; col++)
    for (int row = 0; row < 4; row++) {
        float sum = 0.0f;
        for (int k = 0; k < 4; k++)
            sum += A[k*4 + row] * B[col*4 + k];
        out[col*4 + row] = sum;
    }
}

/* Column-major model matrix with yaw, pitch, roll — matches the SW vertex transform
 * in car.c (worldX = carX*CY*CP + carY*(CY*SP*SR-SY*CR) + carZ*(-CY*SP*CR-SY*SR), etc.).
 * All angles in FATAL 14-bit units (16384 = 360°). */
static void car_model_matrix(float M[16], const GameRenderCarPose *pose)
{
    static const float kToRad = (float)(2.0 * 3.14159265358979) / 16384.0f;
    float CY = cosf((float)pose->yaw   * kToRad), SY = sinf((float)pose->yaw   * kToRad);
    float CP = cosf((float)pose->pitch * kToRad), SP = sinf((float)pose->pitch * kToRad);
    float CR = cosf((float)pose->roll  * kToRad), SR = sinf((float)pose->roll  * kToRad);

    M[ 0] =  CY*CP;             M[ 1] =  SY*CP;             M[ 2] =  SP;      M[ 3] = 0.0f;
    M[ 4] =  CY*SP*SR - SY*CR;  M[ 5] =  SY*SP*SR + CY*CR;  M[ 6] = -SR*CP;  M[ 7] = 0.0f;
    M[ 8] = -CY*SP*CR - SY*SR;  M[ 9] = -SY*SP*CR + CY*SR;  M[10] =  CP*CR;  M[11] = 0.0f;
    M[12] = pose->position.fX;
    M[13] = pose->position.fY;
    M[14] = pose->position.fZ;
    M[15] = 1.0f;
}

/* Column-major chunk transform from localdata[chunk].pointAy. */
static void chunk_model_matrix(float M[16], const tData *d)
{
    M[ 0]=d->pointAy[0].fX; M[ 1]=d->pointAy[1].fX; M[ 2]=d->pointAy[2].fX; M[ 3]=0.0f;
    M[ 4]=d->pointAy[0].fY; M[ 5]=d->pointAy[1].fY; M[ 6]=d->pointAy[2].fY; M[ 7]=0.0f;
    M[ 8]=d->pointAy[0].fZ; M[ 9]=d->pointAy[1].fZ; M[10]=d->pointAy[2].fZ; M[11]=0.0f;
    M[12]=-d->pointAy[3].fX;
    M[13]=-d->pointAy[3].fY;
    M[14]=-d->pointAy[3].fZ;
    M[15]=1.0f;
}

/* ==========================================================================
 * Lifecycle
 * ========================================================================== */

GameRendererHardware *game_render_hw_create(SDL_GPUDevice *device)
{
    GameRendererHardware *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->device = device;
    return r;
}

void game_render_hw_destroy(GameRendererHardware *r)
{
    if (!r) return;
    for (int i = 0; i < GAME_RENDER_HW_NUM_CARS; i++) {
        GRHWCarMesh *m = &r->meshes[i];
        if (!m->built) continue;
        for (int f = 0; f < GRHW_ANIM_FRAMES; f++)
            if (m->vertexBufs[f]) SDL_ReleaseGPUBuffer(r->device, m->vertexBufs[f]);
        if (m->indexBuf)  SDL_ReleaseGPUBuffer(r->device, m->indexBuf);
        if (m->atlas)     game_render_hw_release_car_atlas(r->device, m->atlas);
    }
    for (int i = 0; i < GAME_RENDER_HW_NUM_CARS; i++) {
        if (s_tagLabelCache[i].tex) SDL_ReleaseGPUTexture(r->device, s_tagLabelCache[i].tex);
        s_tagLabelCache[i] = (GRHWTagLabelCache){0};
    }
    free(r);
}

/* ==========================================================================
 * Public: draw one car this frame
 * ========================================================================== */

void game_render_hw_draw_car(GameRendererHardware       *r,
                              SceneRendererGPU           *scene,
                              int                         carIdx,
                              const GameRenderCarPose    *pose,
                              const GameRenderCarOptions *options)
{
    if (!r || !scene || !pose) return;
    if (carIdx < 0 || carIdx >= GAME_RENDER_HW_NUM_CARS) return;

    GRHWCarMesh *mesh = &r->meshes[carIdx];
    bool wantAdvanced = (textures_off & TEX_OFF_ADVANCED_CARS) != 0;
    int wantDesign = (int)Car[carIdx].byCarDesignIdx;
    if (mesh->built
        && (mesh->advancedCars != wantAdvanced || mesh->designIdx != wantDesign)) {
        for (int f = 0; f < GRHW_ANIM_FRAMES; f++) {
            if (mesh->vertexBufs[f]) { SDL_ReleaseGPUBuffer(r->device, mesh->vertexBufs[f]); mesh->vertexBufs[f] = NULL; }
        }
        if (mesh->indexBuf)  { SDL_ReleaseGPUBuffer(r->device, mesh->indexBuf);  mesh->indexBuf  = NULL; }
        if (mesh->atlas)     { game_render_hw_release_car_atlas(r->device, mesh->atlas); mesh->atlas = NULL; }
        mesh->built = false;
    }
    if (!mesh->built) {
        if (build_car_mesh(r->device, carIdx, palette, mesh))
            mesh->built = true;
        else
            return;
        mesh->advancedCars = wantAdvanced;
        mesh->designIdx = wantDesign;
    }
    if (mesh->indexCount <= 0) return;

    int animFrame = (options && options->anim_frame >= 0 && options->anim_frame < GRHW_ANIM_FRAMES)
                    ? options->anim_frame : 0;
    SDL_GPUBuffer *vertBuf = mesh->vertexBufs[animFrame];

    float vp[16], M[16], mvp[16];
    scene_render_gpu_build_vp(scene, vp);

    /* Apply the same visual offsets the SW renderer adds (car.c lines 1111-1113):
     * yaw += iYawMotion; pitch += CameraOffset + Motion + DynamicOffset; same for roll. */
    GameRenderCarPose adjPose = *pose;
    adjPose.yaw   = (pose->yaw
                     + (int)(int16)Car[carIdx].iYawMotion) & 0x3FFF;
    adjPose.pitch = (pose->pitch
                     + Car[carIdx].iPitchCameraOffset
                     + Car[carIdx].iPitchMotion
                     + Car[carIdx].iPitchDynamicOffset) & 0x3FFF;
    adjPose.roll  = (pose->roll
                     + Car[carIdx].iRollCameraOffset
                     + Car[carIdx].iRollMotion
                     + Car[carIdx].iRollDynamicOffset) & 0x3FFF;
    car_model_matrix(M, &adjPose);

    int chunk = Car[carIdx].nCurrChunk;
    bool airborne = (chunk < 0 || chunk >= MAX_TRACK_CHUNKS);

    int designIdx = (int)Car[carIdx].byCarDesignIdx;
    if (designIdx < 0 || designIdx > CAR_DESIGN_DEATH) designIdx = 0;
    float fZLow = CarBox.hitboxAy[designIdx][0].fZ;
    M[14] -= fZLow;  /* lift body so bottom vertex lands at physics pos */

    if (!airborne) {
        float Mc[16], Mcar_chunk[16];
        chunk_model_matrix(Mc, &localdata[chunk]);
        mat4_mul(Mcar_chunk, Mc, M);
        mat4_mul(mvp, vp, Mcar_chunk);
        float shine[12];
        car_shine_inputs(shine, vp, Mcar_chunk);
        scene_render_gpu_queue_car_draw(scene,
            vertBuf, mesh->indexBuf, mesh->atlas,
            0, mesh->bodyIndexCount, mvp, shine);

        /* Shadow: use physical pose (no camera lean offsets) so shadow corners
         * stay exactly on the road surface.  adjPose tilts the quad slightly
         * off-road, letting it pass depth-test through barriers.
         *
         * Rotation: rebuilt using YAW ONLY (pitch/roll dropped) so the quad's
         * footprint doesn't rotate/mirror with the car's own roll (corkscrew
         * sections, cars resting flipped) -- car_model_matrix's full rotation
         * would otherwise flip the quad's in-plane spread at roll=180.
         *
         * Translation: pose->position.fZ is not a world height -- it is
         * height above whichever point of the car is currently touching the
         * ground, referenced from the car's hitbox origin.  For an upright
         * car that's the hitbox BOTTOM (fZLow, index 0) and pose.fZ sits near
         * 0.  For a car resting on its roof, physics keeps the same
         * reference point but the touching surface is now the hitbox TOP
         * (fHitboxTop, index 4) -- pose.fZ then reads ~ the car's full
         * height (confirmed via logging: ~410 units while resting flipped,
         * ~0 while grounded normally), which used to leak straight into the
         * shadow's height and made it float up around wheel-level instead of
         * sitting on the ground.  Selecting the offset by orientation cancels
         * that out in both cases. */
        float M_phys[16];
        car_model_matrix(M_phys, pose);
        /* physInverted must come from Car[].iStunned (roll-only test, matches
         * control.c's own upside-down detection) rather than M_phys[10]
         * (cos(pitch)*cos(roll)): on a very steep decline pitch alone can
         * exceed 90 deg and flip that product negative even though the car
         * is right-side-up, which picked the wrong hitbox reference and,
         * since the chunk's local "up" axis is nearly horizontal on such a
         * steep slope, showed up as the shadow sliding far down-slope rather
         * than a small height error. */
        bool physInverted = (Car[carIdx].iStunned != 0);
        {
            static const float kToRad = (float)(2.0 * 3.14159265358979) / 16384.0f;
            float CY = cosf((float)pose->yaw * kToRad), SY = sinf((float)pose->yaw * kToRad);
            M_phys[0] =  CY; M_phys[1] = SY; M_phys[2] = 0.0f;
            M_phys[4] = -SY; M_phys[5] = CY; M_phys[6] = 0.0f;
            M_phys[8] = 0.0f; M_phys[9] = 0.0f; M_phys[10] = 1.0f;
        }
        float fHitboxTop = CarBox.hitboxAy[designIdx][4].fZ;
        M_phys[14] -= physInverted ? fHitboxTop : fZLow;
        float Mcar_chunk_phys[16], shadowMvp[16];
        mat4_mul(Mcar_chunk_phys, Mc, M_phys);
        mat4_mul(shadowMvp, vp, Mcar_chunk_phys);
        scene_render_gpu_queue_car_shadow_draw(scene,
            vertBuf, mesh->indexBuf, mesh->atlas,
            mesh->bodyIndexCount, 6, shadowMvp);

    } else {
        /* Body: no chunk transform (Car.pos is world-space when airborne) */
        mat4_mul(mvp, vp, M);
        float shine[12];
        car_shine_inputs(shine, vp, M);
        scene_render_gpu_queue_car_draw(scene,
            vertBuf, mesh->indexBuf, mesh->atlas,
            0, mesh->bodyIndexCount, mvp, shine);

        /* Shadow: replicate car.c:1339-1431's full branch choice for this
         * car's own airborne shadow (3 cases, matching SW exactly):
         *  1. GroundColour[][2] == -2: "underground check" -- direct height
         *     lookup (getbankz/getgroundz/getroadz), no plane fit.
         *  2. Otherwise, if the car is laterally within track+shoulder bounds
         *     and close to the road surface (or GroundColour[][2] == -1):
         *     direct road/ground height again -- this is what makes the
         *     shadow reappear right at touchdown instead of relying on a
         *     plane fit from the chunk the car launched from.
         *  3. Otherwise, the plane-equation fallback: separated (de-banked)
         *     frame if GroundColour[][2] >= 2 AND the car isn't AI-controlled
         *     (iControlType == 3), else the plain frame. The separated
         *     frame's chunk-pair-derived tangent is less smooth chunk-to-
         *     chunk than the regular frame's own orientation, so it must
         *     stay gated to this case only or ordinary jumps get visible
         *     wobble. */
        int shadowChunk = Car[carIdx].iLastValidChunk;
        if (shadowChunk >= 0 && shadowChunk < MAX_TRACK_CHUNKS) {
            float wx = M[12], wy = M[13], wz = M[14];
            const tData *sd0 = &localdata[shadowChunk];
            const tTrackInfo *ti0 = &TrackInfo[shadowChunk];
            float p3x0 = sd0->pointAy[3].fX, p3y0 = sd0->pointAy[3].fY, p3z0 = sd0->pointAy[3].fZ;
            float relX0 = wx + p3x0, relY0 = wy + p3y0, relZ0 = wz + p3z0;
            /* Full 3-term world->local transform (matches car.c's carlocal[]
             * exactly, unlike the abbreviated 2-term version the plain branch
             * below uses) -- needed for the bounds/tunnel tests that pick
             * which of SW's 3 branches applies. */
            float localX0 = sd0->pointAy[0].fX*relX0 + sd0->pointAy[1].fX*relY0 + sd0->pointAy[2].fX*relZ0;
            float localY0 = sd0->pointAy[0].fY*relX0 + sd0->pointAy[1].fY*relY0 + sd0->pointAy[2].fY*relZ0;
            float localZ0 = sd0->pointAy[0].fZ*relX0 + sd0->pointAy[1].fZ*relY0 + sd0->pointAy[2].fZ*relZ0;

            /* Tunnel check (car.c:1317-1329): true only under a roofed
             * section (surface type 5/6/9) below its roof height, on
             * whichever side of centerline the car sits. */
            bool tunnel = false;
            if (localY0 <= 0.0f) {
                if (-sd0->fTrackHalfWidth < localY0 &&
                    (ti0->iRightSurfaceType == 5 || ti0->iRightSurfaceType == 6 || ti0->iRightSurfaceType == 9) &&
                    localZ0 < ti0->fRoofHeight)
                    tunnel = true;
            } else if (localY0 < sd0->fTrackHalfWidth) {
                if ((ti0->iLeftSurfaceType == 5 || ti0->iLeftSurfaceType == 6 || ti0->iLeftSurfaceType == 9) &&
                    localZ0 < ti0->fRoofHeight)
                    tunnel = true;
            }

            int iGroundColorType = GroundColour[shadowChunk][2];
            bool havePointHeight = false;
            float pointHeight = 0.0f;

            if (iGroundColorType == -2) {
                /* Underground check (car.c:1339-1363): SW clamps each shadow
                 * corner independently against whichever height source
                 * applies at that corner, with no plane fit at all. This
                 * single-point stand-in evaluates the same branch choice
                 * once, at the car's own position, consistent with how the
                 * plain/separated branches below already treat the shadow as
                 * one flat plane rather than 4 independently-clamped corners. */
                bool outsideShoulder = (localY0 > sd0->fTrackHalfWidth + ti0->fLShoulderWidth) ||
                                       (localY0 < -sd0->fTrackHalfWidth - ti0->fRShoulderWidth);
                if (tunnel)
                    pointHeight = (float)getroadz(localX0, localY0, shadowChunk);
                else if (outsideShoulder)
                    pointHeight = (float)getbankz(localY0, shadowChunk, NULL);
                else
                    pointHeight = (float)getgroundz(localX0, localY0, shadowChunk);
                havePointHeight = true;
            } else {
                /* On-track check (car.c:1364-1385): true once GroundColour[][2]
                 * == -1, or once the car sits laterally within track+shoulder
                 * bounds AND close enough to the road surface (roadheight -
                 * 400 <= carZ) -- i.e. right around touchdown. Using the
                 * direct road/ground height here (no plane) is what actually
                 * restores the shadow just before landing: the plain/
                 * separated branches below fit a plane from the chunk the car
                 * launched from, which can be far off the true height by the
                 * time a long jump lands several chunks later. */
                bool withinBounds = (localY0 <= sd0->fTrackHalfWidth + ti0->fLShoulderWidth) &&
                                     (localY0 >= -sd0->fTrackHalfWidth - ti0->fRShoulderWidth);
                float roadHeight = tunnel ? (float)getroadz(localX0, localY0, shadowChunk)
                                          : (float)getgroundz(localX0, localY0, shadowChunk);
                bool onTrack = (iGroundColorType == -1);
                if (!onTrack && withinBounds &&
                    (TrakColour[shadowChunk][Car[carIdx].iLaneType] & SURFACE_FLAG_SKIP_RENDER) == 0 &&
                    roadHeight - 400.0f <= localZ0)
                    onTrack = true;
                if (onTrack) {
                    pointHeight = roadHeight;
                    havePointHeight = true;
                }
            }

            bool useSeparated = !havePointHeight && !(GroundColour[shadowChunk][2] < 2 || Car[carIdx].iControlType == 3);
            float groundZ, sxA, syA;

            if (havePointHeight) {
                /* Convert the local-frame height back to world Z the same
                 * way the separated branch does (dot with the local "up"
                 * axis's world-space representation), just sourced from the
                 * regular chunk frame instead of the debanked one. */
                groundZ = sd0->pointAy[2].fX*localX0 + sd0->pointAy[2].fY*localY0 + sd0->pointAy[2].fZ*pointHeight - p3z0;
                sxA = 0.0f; syA = 0.0f;
            } else if (useSeparated) {
                tData tempData;
                calculateseparatedcoordinatesystem(shadowChunk, &tempData);
                const tData *sd = &tempData;
                float p3x = sd->pointAy[3].fX, p3y = sd->pointAy[3].fY, p3z = sd->pointAy[3].fZ;
                float lx = sd->pointAy[0].fX*(wx+p3x) + sd->pointAy[1].fX*(wy+p3y) + sd->pointAy[2].fX*(wz+p3z);
                float ly = sd->pointAy[0].fY*(wx+p3x) + sd->pointAy[1].fY*(wy+p3y) + sd->pointAy[2].fY*(wz+p3z);
                float lz = (float)getbankz(ly, shadowChunk, &tempData);
                groundZ = sd->pointAy[2].fX*lx + sd->pointAy[2].fY*ly + sd->pointAy[2].fZ*lz - p3z;
                /* pointAy[2] is forced to (0,0,1) by calculateseparatedcoordinatesystem,
                 * so this plane has no XY tilt and the skew below is always zero. */
                sxA = 0.0f; syA = 0.0f;
            } else {
                /* Plain plane equation against the chunk's own regular frame --
                 * the formula that already worked for every normal jump before
                 * this session's changes. */
                const tData *sd = &localdata[shadowChunk];
                float p3x = sd->pointAy[3].fX, p3y = sd->pointAy[3].fY, p3z = sd->pointAy[3].fZ;
                float lx = sd->pointAy[0].fX*(wx+p3x) + sd->pointAy[1].fX*(wy+p3y);
                float ly = sd->pointAy[0].fY*(wx+p3x) + sd->pointAy[1].fY*(wy+p3y);
                groundZ = sd->pointAy[2].fX*lx + sd->pointAy[2].fY*ly - p3z;
                sxA = sd->pointAy[2].fX * sd->pointAy[0].fX + sd->pointAy[2].fY * sd->pointAy[0].fY;
                syA = sd->pointAy[2].fX * sd->pointAy[1].fX + sd->pointAy[2].fY * sd->pointAy[1].fY;
            }

            /* Direction from pose->yaw directly, NOT M[0]/M[1]: those equal
             * (cos(yaw)*cos(pitch), sin(yaw)*cos(pitch)), so their combined
             * length is |cos(pitch)| -- exactly zero when the car points
             * straight up/down. On a shallow ramp pitch never gets that
             * steep, but an extreme near-vertical launch collapses fx/fy to
             * ~0, degenerating the shadow quad's rotation basis to near-zero
             * size for as long as pitch stays near +-90 deg. Yaw alone is
             * always well-defined regardless of pitch. */
            static const float kToRad = (float)(2.0 * 3.14159265358979) / 16384.0f;
            float fx = cosf((float)pose->yaw * kToRad);
            float fy = sinf((float)pose->yaw * kToRad);
            float Mshadow[16] = {0};
            Mshadow[ 0]= fx;            Mshadow[ 1]= fy;
            Mshadow[ 2]= sxA*fx + syA*fy;
            Mshadow[ 4]=-fy;            Mshadow[ 5]= fx;
            Mshadow[ 6]= sxA*(-fy) + syA*fx;
            Mshadow[10]=1.0f;
            Mshadow[12]=wx;             Mshadow[13]=wy;
            Mshadow[14]=groundZ - fZLow;
            Mshadow[15]=1.0f;

            float shadowMvp[16];
            mat4_mul(shadowMvp, vp, Mshadow);
            scene_render_gpu_queue_car_shadow_draw(scene,
                vertBuf, mesh->indexBuf, mesh->atlas,
                mesh->bodyIndexCount, 6, shadowMvp);
        }
    }
}

/* ==========================================================================
 * Car name tag overlay (GPU mode)
 *
 * Mirrors the name-tag block from car.c's DisplayCarWithPose, using the same
 * globals and projection formula. Must be called AFTER game_render_hw_draw_car
 * and BEFORE NamesLeft is decremented for this car.
 * ========================================================================== */
void game_render_hw_draw_car_name_tag(int carIdx, const GameRenderCarPose *pose, int viewSlot)
{
    if (!names_on) return;
    if (NamesLeft >= 16 || NamesLeft < -2) return;
    if (replaytype == 2) return;
    if (winner_mode) return;
    if (intro) return;
    if (viewSlot < 0 || viewSlot >= GAME_RENDER_HW_MAX_VIEW_SLOTS) viewSlot = 0;

    tCar *pCar = &Car[carIdx];
    if (pCar->byStatusFlags & 2) { s_tagScrX[viewSlot][carIdx] = -1; s_tagScrY[viewSlot][carIdx] = -1; return; }
    if (!(names_on == 1 || (names_on == 2 && human_control[pCar->iDriverIdx])  || (names_on == 3 && !human_control[pCar->iDriverIdx]) )) return; // scf32
    
    int carDesignIndex = pCar->byCarDesignIdx;
    float fHitboxZ = CarBox.hitboxAy[carDesignIndex][4].fZ;

    int iCurrChunk = pCar->nCurrChunk;
    const float *p = (iCurrChunk >= 0 && iCurrChunk < MAX_TRACK_CHUNKS)
                     ? (const float *)localdata[iCurrChunk].pointAy : NULL;

    /* scrX: project bounding-box centre-top so the label stays centred over
     * the car regardless of yaw (no wobble as the car rotates).
     *
     * Matches car.c:2068-2072 (SW's own name-tag placement): offset from the
     * car's actual position by fHitboxZ along the car's own LOCAL Z axis,
     * rotated into the same frame by the car's full orientation matrix --
     * NOT a flat "add hitbox height" like the old `fHitboxZ - fZLowTag +
     * pose->position.fZ` formula used, which double-counted when the car
     * rests on its roof: pose->position.fZ is already referenced from
     * whichever hitbox point is touching down (see [[project_car_shadow]]),
     * so adding the full hitbox height again on top sent the tag way above
     * the car. Rotating by the car's own matrix instead means an inverted
     * car's local +Z point naturally maps close to world-down, landing the
     * tag right back near the visible (now-flipped) car -- exactly like SW. */
    /* Same camera-lean/motion offsets game_render_hw_draw_car applies to the
     * car body mesh (car.c:1111-1113) -- omitting them barely matters for a
     * car far away, but the resulting rotation error scales with proximity,
     * so a nearby car's tag visibly drifts from its actual position without
     * this (confirmed: only noticeable for cars close to the camera, in
     * either player's view in 2-player mode -- AI-only single-player rarely
     * puts another car's tag this close in frame). */
    GameRenderCarPose adjPose = *pose;
    adjPose.yaw   = (pose->yaw
                     + (int)(int16)Car[carIdx].iYawMotion) & 0x3FFF;
    adjPose.pitch = (pose->pitch
                     + Car[carIdx].iPitchCameraOffset
                     + Car[carIdx].iPitchMotion
                     + Car[carIdx].iPitchDynamicOffset) & 0x3FFF;
    adjPose.roll  = (pose->roll
                     + Car[carIdx].iRollCameraOffset
                     + Car[carIdx].iRollMotion
                     + Car[carIdx].iRollDynamicOffset) & 0x3FFF;

    float Mrot[16];
    car_model_matrix(Mrot, &adjPose);
    float fCX = pose->position.fX + fHitboxZ * Mrot[8];
    float fCY = pose->position.fY + fHitboxZ * Mrot[9];
    float fCZ_tag = pose->position.fZ + fHitboxZ * Mrot[10];
    float wX0, wY0, wZ0;
    if (!p) { wX0 = fCX; wY0 = fCY; wZ0 = fCZ_tag; }
    else {
        wX0 = p[0]*fCX + p[1]*fCY + p[2]*fCZ_tag - p[9];
        wY0 = p[3]*fCX + p[4]*fCY + p[5]*fCZ_tag - p[10];
        wZ0 = p[6]*fCX + p[7]*fCY + p[8]*fCZ_tag - p[11];
    }
    double dX0 = wX0-viewx, dY0 = wY0-viewy, dZ0 = wZ0-viewz;
    double fVX0 = dX0*vk1 + dY0*vk4 + dZ0*vk7;
    double fVY0 = dX0*vk2 + dY0*vk5 + dZ0*vk8;
    double fCamZ0 = dX0*vk3 + dY0*vk6 + dZ0*vk9;
    if (fCamZ0 <= 0.0) {
        /* Car behind camera -- not visible this frame. Reset the smoothing
         * state instead of leaving it frozen at its last value: otherwise,
         * when the car comes back into view, the EMA blends from that stale
         * position toward the real one over several frames instead of
         * popping in fresh, which reads as the tag "sliding in" from the
         * edge of the screen. */
        s_tagScrX[viewSlot][carIdx] = -1;
        s_tagScrY[viewSlot][carIdx] = -1;
        return;
    }
    double fClZ0 = fCamZ0 < 80.0 ? 80.0 : fCamZ0;
    int scrX = hw_d2i((double)VIEWDIST * fVX0 / fClZ0 + xbase) * scr_size >> 6;

    /* Smooth scrX with an EMA (K=4), same treatment as scrY below -- raw 1/z
     * projection noise shows up on X too (most visibly for distant cars, where
     * a small world-space wobble maps to a disproportionate screen-space swing),
     * and unlike scrY this had no smoothing at all before, so it jumped frame to
     * frame. Reset instantly on large deltas (car first appearing, large jump). */
    {
        int prev = s_tagScrX[viewSlot][carIdx];
        int delta = (prev < 0) ? (winw + 1) : (scrX - prev);
        if (delta < 0) delta = -delta;
        if (prev < 0 || delta > winw / 4)
            s_tagScrX[viewSlot][carIdx] = scrX;
        else
            s_tagScrX[viewSlot][carIdx] = (3 * prev + scrX + 2) / 4;
        scrX = s_tagScrX[viewSlot][carIdx];
    }

    /* scrY from same centre-top point as scrX.  The old approach scanned four
     * hitbox corners and took the highest-projecting one; as the car rotates
     * different corners win, producing visible Y jitter.  The centre point is
     * yaw-independent, but residual 1/z noise still needs the EMA below. */
    double sY0 = (double)VIEWDIST * fVY0 / fClZ0 + ybase;
    int scrY = (scr_size * (199 - hw_d2i(sY0))) >> 6;

    /* Smooth scrY with an EMA (K=4) to suppress residual 1/z noise.  Reset
     * instantly on large deltas (car first appearing, large jump). */
    {
        int prev = s_tagScrY[viewSlot][carIdx];
        int delta = (prev < 0) ? (winh + 1) : (scrY - prev);
        if (delta < 0) delta = -delta;
        if (prev < 0 || delta > winh / 4)
            s_tagScrY[viewSlot][carIdx] = scrY;
        else
            s_tagScrY[viewSlot][carIdx] = (3 * prev + scrY + 2) / 4;
        scrY = s_tagScrY[viewSlot][carIdx];
    }

    /* Real GPU billboard from here on: a screen-space, depth-tested quad (the
     * same mechanism smoke/particle quads already use -- scene_render_gpu_
     * screen_quad_textured/flat + set_particle_ndcz), NOT the old flat
     * scrbuf-HUD text draw. SW's own name tags have no depth/occlusion
     * concept at all (matches DisplayCarWithPose, car.c ~2061-2147), so real
     * occlusion behind walls is a GPU-only enhancement, not a SW-parity fix
     * -- but unlike an earlier CPU depth-readback attempt at the same thing,
     * this gets real hardware depth testing with no staleness/sync risk,
     * since it's just another queued draw in the same per-view pass cars/
     * signs/trees already use (so 2P/mirror "for free" too). */
    SceneRendererGPU *gpu = game_render_get_gpu(g_pGameRenderer);
    SDL_GPUDevice *device = game_render_get_device(g_pGameRenderer);
    if (!gpu || !device) return;

    /* Build label string: "NAME (POS)". */
    char tagLabel[32];
    if (pCar->byRacePosition < racers - 1 || racers == 1)
        sprintf(tagLabel, "%s (%s)", driver_names[carIdx],
                &language_buffer[64 * pCar->byRacePosition + 384]);
    else
        sprintf(tagLabel, "%s (%s)", driver_names[carIdx], &language_buffer[1344]);

    if (!ensure_tag_label_texture(device, tagLabel, &s_tagLabelCache[carIdx]))
        return;
    int texW = s_tagLabelCache[carIdx].texW;
    int texH = s_tagLabelCache[carIdx].texH;

    /* Text's TOP edge sits at a fixed scrY-16 offset (matches the old
     * iDisplayY = scrY - 16 convention exactly -- that -16 is a fixed margin
     * above the anchor, not "the text is 16px tall"), extending down by the
     * label's own real height; the triangle marker below occupies scrY-7 to
     * scrY-1, so this leaves the same gap between text and triangle the old
     * SW-drawn text had. Centred horizontally on scrX -- no mirror-direction
     * handling needed here (unlike the old mini_prt_string_rev asymmetric
     * left/right-edge anchor) since a centred quad doesn't care which way
     * text used to be drawn; mirror is instead handled below as a genuine
     * horizontal flip of the quad itself, so the text still reads correctly
     * in the rearview mirror's flipped camera. */
    float leftOff = -(float)(texW / 2), rightOff = leftOff + (float)texW;
    float topOff  = -16.0f,             bottomOff = topOff + (float)texH;
    if (scrX + rightOff <= 0 || scrX + leftOff >= winw
        || scrY + bottomOff <= 0 || scrY + topOff >= winh)
        return;

    float zF = SCENE_GPU_FAR / (SCENE_GPU_FAR - SCENE_GPU_NEAR);
    float zB = -(SCENE_GPU_FAR * SCENE_GPU_NEAR) / (SCENE_GPU_FAR - SCENE_GPU_NEAR);
    float tagDepth = zF + zB / (float)fCamZ0;

    /* Project each corner from an offset in the SAME camera-space coordinates
     * (fVX0/fVY0/fClZ0) the anchor itself was derived from, inverting exactly
     * the SW scrX/scrY pixel formulas above, then through scene_render_gpu_
     * project_to_ndc -- NOT a scrX/winw-based NDC conversion (see that
     * function's comment for why that breaks in "Render Scale (native)"
     * mode: the pixel-space intermediate bakes in the SW-implied FOV, which
     * only matches the GPU's actual FOV when the GPU viewport isn't
     * independently widened). pixToFv converts a FINAL screen-pixel offset
     * (texW/texH and the fixed triangle offsets below, all in real output-
     * pixel units) into a camera-space X/Y offset at this point's depth; Y is
     * negated since screen-down is -fV.y. The scrX/scrY formulas above have
     * an extra "* scr_size >> 6" beyond the raw VIEWDIST*fV/fClZ0 term this
     * inverts, so a FINAL-pixel delta needs an extra 64/scr_size factor to
     * convert to the same "raw" units first (that factor happens to cancel
     * back out against fovX/fovY's own scr_size dependence inside
     * project_to_ndc, so the net result is correct at every scr_size). */
    double pixToFv = (64.0 / (double)scr_size) * (fClZ0 / (double)VIEWDIST);
    float ndcX[4], ndcY[4], tNdcX[4], tNdcY[4];
    /* v0=top-right, v1=top-left, v2=bottom-left, v3=bottom-right normally;
     * mirror swaps v0<->v1 and v2<->v3, same flip trick used elsewhere in
     * this file's GPU code (game_render.c's picture-quad flipH). */
    float dx4[4], dy4[4];
    if (mirror) {
        dx4[0]=leftOff;  dy4[0]=topOff;    dx4[1]=rightOff; dy4[1]=topOff;
        dx4[2]=rightOff; dy4[2]=bottomOff; dx4[3]=leftOff;  dy4[3]=bottomOff;
    } else {
        dx4[0]=rightOff; dy4[0]=topOff;    dx4[1]=leftOff;  dy4[1]=topOff;
        dx4[2]=leftOff;  dy4[2]=bottomOff; dx4[3]=rightOff; dy4[3]=bottomOff;
    }
    for (int i = 0; i < 4; i++) {
        double cFVX = fVX0 + (double)dx4[i] * pixToFv;
        double cFVY = fVY0 - (double)dy4[i] * pixToFv;
        scene_render_gpu_project_to_ndc(gpu, cFVX, cFVY, fClZ0, &ndcX[i], &ndcY[i]);
    }
    scene_render_gpu_set_particle_ndcz(gpu, tagDepth);
    scene_render_gpu_screen_quad_textured(gpu, ndcX, ndcY, s_tagLabelCache[carIdx].tex,
                                          1.0f, 1.0f, 1.0f, 1.0f);

    /* Coloured downward-pointing triangle above car, same depth as the text. */
    float ttx4[4] = { 6.0f, -5.0f, 0.0f, 6.0f };
    float tty4[4] = { -7.0f, -7.0f, -1.0f, -7.0f };
    for (int i = 0; i < 4; i++) {
        double cFVX = fVX0 + (double)ttx4[i] * pixToFv;
        double cFVY = fVY0 - (double)tty4[i] * pixToFv;
        scene_render_gpu_project_to_ndc(gpu, cFVX, cFVY, fClZ0, &tNdcX[i], &tNdcY[i]);
    }
    /* Textured (not flat) so this survives 2-player mode -- see
     * scene_render_gpu_get_flat_color_texture's comment: flat-particle quads
     * draw as a whole batch before textured ones regardless of queue order,
     * and each player's composited view is itself an opaque textured quad
     * that would otherwise paint right over a flat triangle queued earlier. */
    SDL_GPUTexture *teamTex = scene_render_gpu_get_flat_color_texture(gpu, (uint8)team_col[carDesignIndex]);
    if (teamTex) {
        scene_render_gpu_set_particle_ndcz(gpu, tagDepth);
        scene_render_gpu_screen_quad_textured(gpu, tNdcX, tNdcY, teamTex, 1.0f, 1.0f, 1.0f, 1.0f);
    }
}

void game_render_hw_draw_fps_overlay(void)
{
    if (g_iFpsDisplay == 0 || !scrbuf) return;

    static Uint64 s_lastTick = 0;
    static float  s_avgMs    = 0.0f;

    Uint64 now  = SDL_GetPerformanceCounter();
    Uint64 freq = SDL_GetPerformanceFrequency();
    if (s_lastTick > 0 && freq > 0) {
        float ms = (float)((double)(now - s_lastTick) * 1000.0 / (double)freq);
        if (ms > 0.1f && ms < 2000.0f)
            s_avgMs = (s_avgMs > 0.0f) ? (0.9f * s_avgMs + 0.1f * ms) : ms;
    }
    s_lastTick = now;

    if (s_avgMs <= 0.0f) return;

    char buf[16];
    sprintf(buf, "%.0f FPS", 1000.0f / s_avgMs);

    int textW = 0;
    for (int i = 0; buf[i]; i++) textW += 5;
    const int textH = 7;
    const int margin = 4;

    int x, y;
    switch (g_iFpsDisplay) {
        default:
        case 1: x = margin;               y = margin;                    break;
        case 2: x = winw - textW - margin; y = margin;                    break;
        case 3: x = margin;               y = winh - textH - margin;     break;
        case 4: x = winw - textW - margin; y = winh - textH - margin;    break;
    }

    /* In 2-player split screen this runs after winw/winh are restored to the
     * FULL window size, so all four corners land inside one player's own
     * half at exactly the corner their own per-half HUD (lap list/current
     * lap top corners, minimap/speed-gear bottom corners) already occupies.
     * Every corner collides with something, so keep the requested top/bottom
     * choice but force horizontal centering, which every per-half HUD
     * element leaves clear. */
    if (player_type == 2)
        x = (winw - textW) / 2;

    if (x < 0 || y < 0 || x + textW > winw || y + textH > winh) return;

    uint8 *prevScreenPointer = screen_pointer;
    int    prevScrSize       = scr_size;
    screen_pointer = scrbuf;
    scr_size       = 64;
    mini_prt_string(rev_vga[0], buf, x, y);
    scr_size       = prevScrSize;
    screen_pointer = prevScreenPointer;
}
