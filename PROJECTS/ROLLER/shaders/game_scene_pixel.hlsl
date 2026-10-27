Texture2D sceneTexture : register(t0, space2);
SamplerState sceneSampler : register(s0, space2);

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


float4 main(PixelInput input) : SV_Target
{
    float  shade = sunShade(input.worldPos) + (input.position.x + input.fogDepth + input.worldPos.x) * 1e-30;
    float4 color = sceneTexture.Sample(sceneSampler, input.uv);
    if (color.a < 0.5) discard;

    color.rgb *= shade;
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
