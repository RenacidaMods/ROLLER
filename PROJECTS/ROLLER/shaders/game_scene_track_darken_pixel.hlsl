Texture2D sceneTexture : register(t0, space2);
SamplerState sceneSampler : register(s0, space2);

/* Car shadow map (depth from the sun's view, cars only; see
 * scene_render_gpu.c). Load()/Gather() only, so the sampler is a formality. */
Texture2D<float> shadowMap     : register(t1, space2);
SamplerState     shadowSampler : register(s1, space2);

cbuffer ScenePixelUniforms : register(b0, space3)
{
    float  fogDensity;
    float  gamma;
    float  fogStart;
    float  saturation;
    float4 fogColor;
    float  contrast;
    float  brightness;
    float  carShine;   /* car*: game_car_pixel.hlsl only */
    float  carGlint;
    float  carDark;
    float  carSmooth;
    float  carBevel;
    float  carSun;
    float4 sunDir;     /* xyz: towards the sun (world); w: sun shading strength, 0 = off */
    float4 camPos;     /* camera position (world) */
    float4 shadowParams;      /* x: car shadow darkness (0 = off), y: depth bias */
    float4x4 lightViewProj;   /* world -> the sun's view of the car shadow map */
};

struct PixelInput
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
    float  fogDepth : TEXCOORD1;
    float3 worldPos : TEXCOORD2;
};

float4 main(PixelInput input) : SV_Target
{
    /* No sun shading on this multiply pass; reading every input keeps the
     * Direct3D 12 signature lined up with the vertex shader. */
    float  keep  = (input.position.x + input.fogDepth + input.worldPos.x) * 1e-30;
    float4 color = sceneTexture.Sample(sceneSampler, input.uv);
    if (color.a < 0.5) discard;

    color.rgb = pow(max(color.rgb + keep, 0.00001), 1.0 / gamma);

    float fogDepthAdj = max(0.0, input.fogDepth - fogStart);
    float fogFactor   = saturate(1.0 - exp(-fogDensity * fogDensity * fogDepthAdj * fogDepthAdj));
    color.rgb = lerp(color.rgb, fogColor.rgb, fogFactor);

    float lum = dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
    color.rgb = lerp(float3(lum, lum, lum), color.rgb, saturation);

    color.rgb = saturate((color.rgb - 0.5) * contrast + 0.5);
    color.rgb = saturate(color.rgb + brightness);

    float2 uvEdge = min(input.uv, 1.0 - input.uv);
    float2 uvPerPixel = max(fwidth(input.uv), float2(1e-6, 1e-6));
    float edgeDistancePixels = min(uvEdge.x / uvPerPixel.x,
                                   uvEdge.y / uvPerPixel.y);
    // Pixel centers on the outermost covered row are approximately half a
    // derivative from the geometric edge.  A half-pixel cutoff therefore
    // produces the software renderer's single-pixel border.
    float borderMask = 1.0 - step(0.5, edgeDistancePixels);

    float3 interiorFactor = color.rgb;
    float3 borderFactor = interiorFactor * interiorFactor;
    color.rgb = lerp(interiorFactor, borderFactor, borderMask);
    color.a = 1.0;

    return color;
}
