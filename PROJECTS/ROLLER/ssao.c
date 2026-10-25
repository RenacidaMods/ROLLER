#include "ssao.h"
#include "ssao_shaders.h"
#include <stdlib.h>

/* Two full-screen passes:
 *   1. ssao_ao_pixel: half-resolution occlusion term into an R8 target,
 *      rebuilt from the depth snapshot (no normals or positions needed).
 *   2. ssao_composite_pixel: depth-aware 3x3 upsample, faded with distance
 *      and fog, multiplied into the scene colour (dst = src * dst).
 * Both reuse the full-screen triangle from crt_vertex.hlsl. */

/* Matches cbuffer SSAOUniforms in ssao_ao_pixel.hlsl */
typedef struct {
    float depthSizeX, depthSizeY;
    float fovX, fovY;
    float shiftY;
    float zF, zB;
    float radius;
    float fadeStart, fadeEnd;
    float zNear;
    float _pad;
} SSAOUniforms;

/* Matches cbuffer SSAOCompositeUniforms in ssao_composite_pixel.hlsl */
typedef struct {
    float aoSizeX, aoSizeY;
    float depthSizeX, depthSizeY;
    float zF, zB;
    float strength;
    float fogDensity;
    float fogStart;
    float _pad0, _pad1, _pad2;
} SSAOCompositeUniforms;

struct SSAOPass {
    SDL_GPUDevice           *device;
    SDL_GPUGraphicsPipeline *aoPipeline;
    SDL_GPUGraphicsPipeline *compositePipeline;
    SDL_GPUSampler          *nearestSampler;
    SDL_GPUSampler          *linearSampler;
    SDL_GPUTexture          *aoTex;
    Uint32                   aoW, aoH;
};

static SDL_GPUShader *load_ssao_shader(SDL_GPUDevice *device, SDL_GPUShaderStage stage,
                                       const unsigned char *spirv, unsigned int spirvSize,
                                       const unsigned char *msl,   unsigned int mslSize,
                                       const unsigned char *dxil,  unsigned int dxilSize,
                                       Uint32 numSamplers, Uint32 numUniforms)
{
    SDL_GPUShaderFormat fmts = SDL_GetGPUShaderFormats(device);
    SDL_GPUShaderCreateInfo info = {
        .stage               = stage,
        .num_samplers        = numSamplers,
        .num_uniform_buffers = numUniforms
    };
    if (fmts & SDL_GPU_SHADERFORMAT_SPIRV) {
        info.format = SDL_GPU_SHADERFORMAT_SPIRV; info.code = spirv; info.code_size = spirvSize;
        info.entrypoint = "main";
    } else if (fmts & SDL_GPU_SHADERFORMAT_MSL) {
        info.format = SDL_GPU_SHADERFORMAT_MSL;   info.code = msl;   info.code_size = mslSize;
        info.entrypoint = "main0";
    } else if (fmts & SDL_GPU_SHADERFORMAT_DXIL) {
        info.format = SDL_GPU_SHADERFORMAT_DXIL;  info.code = dxil;  info.code_size = dxilSize;
        info.entrypoint = "main";
    } else {
        return NULL;
    }
    return SDL_CreateGPUShader(device, &info);
}

#define LOAD_SSAO_SHADER(dev, stage, name, samplers, uniforms) \
    load_ssao_shader(dev, stage, name##_spirv, name##_spirv_size, name##_msl, name##_msl_size, \
                     name##_dxil, name##_dxil_size, samplers, uniforms)

static SDL_GPUGraphicsPipeline *make_pipeline(SDL_GPUDevice *device,
                                              SDL_GPUShader *vert, SDL_GPUShader *frag,
                                              SDL_GPUTextureFormat format, bool multiply)
{
    SDL_GPUColorTargetDescription ct = { .format = format };
    if (multiply) {
        ct.blend_state = (SDL_GPUColorTargetBlendState){
            .enable_blend          = true,
            .src_color_blendfactor = SDL_GPU_BLENDFACTOR_DST_COLOR,
            .dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ZERO,
            .color_blend_op        = SDL_GPU_BLENDOP_ADD,
            .src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ZERO,
            .dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE,
            .alpha_blend_op        = SDL_GPU_BLENDOP_ADD
        };
    }
    SDL_GPUGraphicsPipelineCreateInfo pi = {
        .vertex_shader   = vert,
        .fragment_shader = frag,
        .primitive_type  = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .target_info     = { .color_target_descriptions = &ct, .num_color_targets = 1 }
    };
    return SDL_CreateGPUGraphicsPipeline(device, &pi);
}

SSAOPass *ssao_create(SDL_GPUDevice *device, SDL_GPUTextureFormat colorFormat)
{
    if (!device) return NULL;
    SSAOPass *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->device = device;

    SDL_GPUShader *vert = LOAD_SSAO_SHADER(device, SDL_GPU_SHADERSTAGE_VERTEX,   ssao_fullscreen_vertex, 0, 0);
    SDL_GPUShader *ao   = LOAD_SSAO_SHADER(device, SDL_GPU_SHADERSTAGE_FRAGMENT, ssao_ao_pixel,          1, 1);
    SDL_GPUShader *comp = LOAD_SSAO_SHADER(device, SDL_GPU_SHADERSTAGE_FRAGMENT, ssao_composite_pixel,   2, 1);
    if (vert && ao && comp) {
        p->aoPipeline        = make_pipeline(device, vert, ao,   SDL_GPU_TEXTUREFORMAT_R8_UNORM, false);
        p->compositePipeline = make_pipeline(device, vert, comp, colorFormat, true);
    }
    if (vert) SDL_ReleaseGPUShader(device, vert);
    if (ao)   SDL_ReleaseGPUShader(device, ao);
    if (comp) SDL_ReleaseGPUShader(device, comp);

    SDL_GPUSamplerCreateInfo si = {
        .min_filter     = SDL_GPU_FILTER_NEAREST,
        .mag_filter     = SDL_GPU_FILTER_NEAREST,
        .mipmap_mode    = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE
    };
    p->nearestSampler = SDL_CreateGPUSampler(device, &si);
    si.min_filter = SDL_GPU_FILTER_LINEAR;
    si.mag_filter = SDL_GPU_FILTER_LINEAR;
    p->linearSampler = SDL_CreateGPUSampler(device, &si);

    if (!p->aoPipeline || !p->compositePipeline || !p->nearestSampler || !p->linearSampler) {
        SDL_Log("ssao: failed to create shaders/pipelines/samplers: %s", SDL_GetError());
        ssao_destroy(p);
        return NULL;
    }
    return p;
}

void ssao_destroy(SSAOPass *p)
{
    if (!p) return;
    if (p->aoTex)             SDL_ReleaseGPUTexture(p->device, p->aoTex);
    if (p->nearestSampler)    SDL_ReleaseGPUSampler(p->device, p->nearestSampler);
    if (p->linearSampler)     SDL_ReleaseGPUSampler(p->device, p->linearSampler);
    if (p->aoPipeline)        SDL_ReleaseGPUGraphicsPipeline(p->device, p->aoPipeline);
    if (p->compositePipeline) SDL_ReleaseGPUGraphicsPipeline(p->device, p->compositePipeline);
    free(p);
}

static bool ensure_ao_texture(SSAOPass *p, Uint32 w, Uint32 h)
{
    if (p->aoTex && p->aoW == w && p->aoH == h) return true;
    if (p->aoTex) SDL_ReleaseGPUTexture(p->device, p->aoTex);
    SDL_GPUTextureCreateInfo ti = {
        .type                 = SDL_GPU_TEXTURETYPE_2D,
        .format               = SDL_GPU_TEXTUREFORMAT_R8_UNORM,
        .usage                = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width                = w,
        .height               = h,
        .layer_count_or_depth = 1,
        .num_levels           = 1
    };
    p->aoTex = SDL_CreateGPUTexture(p->device, &ti);
    p->aoW = p->aoTex ? w : 0;
    p->aoH = p->aoTex ? h : 0;
    return p->aoTex != NULL;
}

void ssao_apply(SSAOPass *p, SDL_GPUCommandBuffer *cmd,
                SDL_GPUTexture *depthTex, SDL_GPUTexture *colorTex,
                Uint32 w, Uint32 h, const SSAOParams *params)
{
    if (!p || !cmd || !depthTex || !colorTex || !params || w == 0 || h == 0) return;
    Uint32 aoW = (w + 1) / 2, aoH = (h + 1) / 2;
    if (!ensure_ao_texture(p, aoW, aoH)) return;

    /* depth = zF + zB / viewZ, the same terms build_mvp puts in the matrix */
    float zF = params->zFar / (params->zFar - params->zNear);
    float zB = -(params->zFar * params->zNear) / (params->zFar - params->zNear);

    /* Pass 1: occlusion at half resolution */
    SDL_GPUColorTargetInfo aoTarget = {
        .texture  = p->aoTex,
        .load_op  = SDL_GPU_LOADOP_DONT_CARE,
        .store_op = SDL_GPU_STOREOP_STORE
    };
    SDL_GPURenderPass *rp = SDL_BeginGPURenderPass(cmd, &aoTarget, 1, NULL);
    if (!rp) return;
    SDL_BindGPUGraphicsPipeline(rp, p->aoPipeline);
    SDL_GPUTextureSamplerBinding depthBinding = { depthTex, p->nearestSampler };
    SDL_BindGPUFragmentSamplers(rp, 0, &depthBinding, 1);
    SSAOUniforms u = {
        .depthSizeX = (float)w,  .depthSizeY = (float)h,
        .fovX = params->fovX,    .fovY = params->fovY,
        .shiftY = params->shiftY,
        .zF = zF,                .zB = zB,
        .radius = params->radius,
        .fadeStart = params->fadeStart, .fadeEnd = params->fadeEnd,
        .zNear = params->zNear
    };
    SDL_PushGPUFragmentUniformData(cmd, 0, &u, sizeof(u));
    SDL_DrawGPUPrimitives(rp, 3, 1, 0, 0);
    SDL_EndGPURenderPass(rp);

    /* Pass 2: blur, fade and multiply into the scene */
    SDL_GPUColorTargetInfo colorTarget = {
        .texture  = colorTex,
        .load_op  = SDL_GPU_LOADOP_LOAD,
        .store_op = SDL_GPU_STOREOP_STORE
    };
    rp = SDL_BeginGPURenderPass(cmd, &colorTarget, 1, NULL);
    if (!rp) return;
    SDL_BindGPUGraphicsPipeline(rp, p->compositePipeline);
    SDL_GPUTextureSamplerBinding bindings[2] = {
        { p->aoTex, p->linearSampler },
        { depthTex, p->nearestSampler }
    };
    SDL_BindGPUFragmentSamplers(rp, 0, bindings, 2);
    SSAOCompositeUniforms cu = {
        .aoSizeX = (float)aoW,   .aoSizeY = (float)aoH,
        .depthSizeX = (float)w,  .depthSizeY = (float)h,
        .zF = zF,                .zB = zB,
        .strength = params->strength,
        .fogDensity = params->fogDensity,
        .fogStart = params->fogStart
    };
    SDL_PushGPUFragmentUniformData(cmd, 0, &cu, sizeof(cu));
    SDL_DrawGPUPrimitives(rp, 3, 1, 0, 0);
    SDL_EndGPURenderPass(rp);
}
