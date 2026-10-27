Texture2D sceneTexture : register(t0, space2);
SamplerState sceneSampler : register(s0, space2);

/* Sky backdrop (the clipped sky polygon, drawn before the 3D scene). The
 * vertex uv carries the pixel's NDC position; the ray through it is rebuilt
 * with the inverse view-projection, and its height above the horizon picks
 * a colour between a pale horizon and a deeper zenith, so the gradient
 * follows the camera's pitch and roll. Equal colours give the original
 * flat sky. */
cbuffer SkyPixelUniforms : register(b0, space3)
{
    float4x4 invViewProj;
    float4   horizonColor;
    float4   zenithColor;
};

struct PixelInput
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
    float  fogDepth : TEXCOORD1;
};

float4 main(PixelInput input) : SV_Target
{
    float2 ndc  = input.uv;
    float4 nearP = mul(invViewProj, float4(ndc, 0.0, 1.0));
    float4 farP  = mul(invViewProj, float4(ndc, 0.5, 1.0));
    float3 dir   = normalize(farP.xyz / farP.w - nearP.xyz / nearP.w);
    /* World up is +Z. The tiny position and fogDepth terms change nothing;
     * reading every input keeps this shader's input signature lined up
     * with the scene vertex shader's outputs, which Direct3D 12 requires. */
    float  el    = saturate(dir.z + (input.position.x + input.fogDepth) * 1e-30);
    float  t     = sqrt(el);
    return float4(lerp(horizonColor.rgb, zenithColor.rgb, t), 1.0);
}
