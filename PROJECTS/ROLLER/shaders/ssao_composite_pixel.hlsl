// Screen-space ambient occlusion, pass 2 of 2 (ssao.c): upsamples the
// half-resolution occlusion with a depth-aware 4x4 blur (matching the
// 4x4 rotation pattern in ssao_ao_pixel.hlsl, so its noise cancels) and
// outputs a grey level. The normal pipeline multiplies it into the scene
// colour (dst = src * dst), so 1.0 leaves a pixel untouched; the "AO only"
// debug pipeline writes it directly.

Texture2D<float> aoTex        : register(t0, space2);
SamplerState     aoSampler    : register(s0, space2);
Texture2D<float> depthTex     : register(t1, space2);
SamplerState     depthSampler : register(s1, space2);

cbuffer SSAOCompositeUniforms : register(b0, space3)
{
    float2 aoSize;      // half-resolution AO size in pixels
    float2 depthSize;   // full-resolution depth size in pixels
    float  zF;          // depth = zF + zB / viewZ (see ssao_ao_pixel.hlsl)
    float  zB;
    float  strength;    // 0 = off, 1 = as computed, 2 = doubled darkening
    float  fogDensity;  // same fog as game_scene_pixel.hlsl, so AO fades with it
    float  fogStart;
    float  _pad0, _pad1, _pad2;   // scalars, not float3, so every backend packs it the same
};

struct PixelInput
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

float viewZ(float d) { return zB / (d - zF); }

float loadDepth(float2 uv)
{
    int2 px = int2(clamp(uv * depthSize, float2(0.0, 0.0), depthSize - 1.0));
    return depthTex.Load(int3(px, 0));
}

float4 main(PixelInput input) : SV_Target
{
    float2 uv = input.uv;
    float d = loadDepth(uv);
    if (d >= 0.9999999)
        return float4(1.0, 1.0, 1.0, 1.0);   // sky: untouched

    float zc = viewZ(d);
    float2 aoTexel = 1.0 / aoSize;
    float sum = 0.0;
    float weightSum = 0.0;
    [unroll]
    for (int y = 0; y < 4; y++) {
        [unroll]
        for (int x = 0; x < 4; x++) {
            float2 tapUV = uv + (float2(x, y) - 1.5) * aoTexel;
            float tapZ = viewZ(loadDepth(tapUV));
            float w = exp(-abs(tapZ - zc) / max(zc * 0.03, 1.0));
            sum += aoTex.SampleLevel(aoSampler, tapUV, 0) * w;
            weightSum += w;
        }
    }
    float ao = sum / max(weightSum, 1e-4);

    float fogDepthAdj = max(0.0, zc - fogStart);
    float fogFactor = saturate(1.0 - exp(-fogDensity * fogDensity * fogDepthAdj * fogDepthAdj));
    ao = lerp(ao, 1.0, fogFactor);

    ao = saturate(1.0 - (1.0 - ao) * strength);
    return float4(ao, ao, ao, 1.0);
}
