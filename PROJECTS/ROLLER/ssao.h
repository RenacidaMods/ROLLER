#pragma once
#include <SDL3/SDL.h>
#include <stdbool.h>

/* Screen-space ambient occlusion for the GPU renderer: softly darkens the
 * scene where surfaces meet (walls and road, under barriers, around cars),
 * working only from a depth snapshot, so no existing shaders change. Off by
 * default; see scene_render_gpu_set_ambient_occlusion. */
typedef struct SSAOPass SSAOPass;

typedef struct {
    float fovX, fovY;      /* projection scale, as in build_mvp: ndc.x = fovX * x / z */
    float shiftY;          /* horizon shift added to ndc.y */
    float zNear, zFar;     /* SCENE_GPU_NEAR / SCENE_GPU_FAR */
    float radius;          /* sample radius in world units (a car is ~1300 long) */
    float strength;        /* 0 = no darkening, 1 = as computed, up to 2 */
    float fadeStart;       /* view distance where AO starts fading out */
    float fadeEnd;         /* view distance where AO has gone */
    float fogDensity;      /* scene fog, so AO fades with it */
    float fogStart;
    bool  showOnly;        /* debug: replace the scene with the AO term (greyscale) */
} SSAOParams;

SSAOPass *ssao_create(SDL_GPUDevice *device, SDL_GPUTextureFormat colorFormat);
void      ssao_destroy(SSAOPass *pass);

/* Darkens colorTex (w x h) by ambient occlusion computed from depthTex, a
 * sampleable D32 depth snapshot of the same size. Must be called with an
 * active command buffer and no render pass open. */
void ssao_apply(SSAOPass *pass, SDL_GPUCommandBuffer *cmd,
                SDL_GPUTexture *depthTex, SDL_GPUTexture *colorTex,
                Uint32 w, Uint32 h, const SSAOParams *params);
