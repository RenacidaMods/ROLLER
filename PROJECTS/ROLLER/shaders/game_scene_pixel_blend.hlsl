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

/* Sun shading: the flat face normal from how the world position changes
 * across the screen, turned towards the camera, lit half-Lambert from the
 * sun. Faces towards the sun keep their colour; faces turned away darken
 * by up to 45% at full strength, so the painted shading isn't washed out.
 * Runs before any discard (it uses screen-space derivatives). */
float sunShade(float3 worldPos)
{
    if (sunDir.w <= 0.0)
        return 1.0;
    float3 N = cross(ddx(worldPos), ddy(worldPos));
    float  len = length(N);
    if (len < 1e-12)
        return 1.0;
    N /= len;
    if (dot(N, camPos.xyz - worldPos) < 0.0)
        N = -N;
    float wrap = saturate(dot(N, sunDir.xyz) * 0.5 + 0.5);
    return 1.0 - sunDir.w * 0.45 * (1.0 - wrap);
}

/* Car shadows: project the point into the sun's view and compare its depth
 * with the nearest car there. A 3x3 grid of 2x2 gathers, each blended by
 * the sub-texel position, gives soft edges about three texels wide. Fades
 * out towards the edge of the map so there's no hard boundary. Returns how
 * lit the point is, 0..1. */
float carShadowLit(float3 worldPos)
{
    if (shadowParams.x <= 0.0)
        return 1.0;
    float4 lp = mul(lightViewProj, float4(worldPos, 1.0));
    float2 uv = float2(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
    if (any(uv <= 0.0) || any(uv >= 1.0) || lp.z <= 0.0 || lp.z >= 1.0)
        return 1.0;
    float w, h;
    shadowMap.GetDimensions(w, h);
    float2 size  = float2(w, h);
    float2 texel = uv * size - 0.5;
    float2 f     = frac(texel);
    float2 base  = (floor(texel) + 1.0) / size;   /* gather centre for texel (i, j)..(i+1, j+1) */
    float  z     = lp.z - shadowParams.y;
    float  lit   = 0.0;
    [unroll]
    for (int y = -1; y <= 1; y++) {
        [unroll]
        for (int x = -1; x <= 1; x++) {
            /* Gather order: (0,1) (1,1) (1,0) (0,0) */
            float4 d = shadowMap.GatherRed(shadowSampler, base + float2(x, y) / size);
            float4 l = step(z, d);
            lit += lerp(lerp(l.w, l.z, f.x), lerp(l.x, l.y, f.x), f.y);
        }
    }
    lit /= 9.0;
    float edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    lit = lerp(1.0, lit, smoothstep(0.0, 0.08, edge));
    return 1.0 - shadowParams.x * (1.0 - lit);
}


float4 main(PixelInput input) : SV_Target
{
    float  shade = sunShade(input.worldPos) + (input.position.x + input.fogDepth + input.worldPos.x) * 1e-30;
    float4 color = sceneTexture.Sample(sceneSampler, input.uv);

    color.rgb *= shade * carShadowLit(input.worldPos);
    color.rgb = pow(max(color.rgb, 0.00001), 1.0 / gamma);

    float fogDepthAdj = max(0.0, input.fogDepth - fogStart);
    float fogFactor   = saturate(1.0 - exp(-fogDensity * fogDensity * fogDepthAdj * fogDepthAdj));
    color.rgb = lerp(color.rgb, fogColor.rgb, fogFactor);

    float lum = dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
    color.rgb = lerp(float3(lum, lum, lum), color.rgb, saturation);

    color.rgb = saturate((color.rgb - 0.5) * contrast + 0.5);
    color.rgb = saturate(color.rgb + brightness);

    return color;
}
