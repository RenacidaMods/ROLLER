Texture2D meshTexture : register(t0, space2);
SamplerState meshSampler : register(s0, space2);

cbuffer ScenePixelUniforms : register(b0, space3)
{
    float  fogDensity;
    float  gamma;
    float  fogStart;
    float  saturation;
    float4 fogColor;
    float  contrast;
    float  brightness;
    float  carShine;   /* 0 = off (and always 0 for car shadows) */
    float  carGlint;   /* sun glint strength, 1 = default */
    float  carDark;    /* how dark the reflected ground and skyline are, 1 = default */
    float  _pad0;
    float  _pad1;
    float  _pad2;
};

struct PixelInput
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
    float4 color    : TEXCOORD1;
    float  fogDepth : TEXCOORD2;
    float3 objPos   : TEXCOORD3;
    float3 toCam    : TEXCOORD4;
    float3 up       : TEXCOORD5;
    float3 xAxis    : TEXCOORD6;
};

/* The surroundings the paint reflects, by direction (unit, world-aligned
 * frame): sky fading from a pale horizon to blue overhead, a dark skyline of
 * block "buildings" along the horizon, darker ground below. The sharp
 * horizon and skyline edges are what slide across the panels as the car
 * turns, the classic env-mapped car look (Screamer, Need for Speed II). */
float3 car_environment(float el, float az, float elWidth, float dark)
{
    float3 horizon = float3(0.86, 0.89, 0.93);
    float3 zenith  = float3(0.22, 0.42, 0.82);
    float3 sky     = lerp(horizon, zenith, sqrt(saturate(el)));

    float cell     = floor((az / 6.2831853 + 0.5) * 28.0);
    float building = 0.025 + 0.11 * frac(sin(cell * 12.9898) * 43758.5453);
    float inSky    = saturate((el - building) / elWidth + 0.5);
    float3 skyline = lerp(horizon * 0.9, float3(0.16, 0.17, 0.20), dark);
    float3 above   = lerp(skyline, sky, inSky);

    float3 ground  = lerp(float3(0.30, 0.29, 0.27), float3(0.10, 0.10, 0.11), saturate(-el * 3.0));
    ground = lerp(horizon * 0.85, ground, dark);
    float aboveHorizon = saturate(el / elWidth + 0.5);
    return lerp(ground, above, aboveHorizon);
}

float4 main(PixelInput input) : SV_Target
{
    /* Car shine: the paint reflects car_environment() along the mirror
     * direction, more at grazing angles (Fresnel), plus a sun glint. The
     * reflection replaces part of the paint colour rather than adding to
     * it, so ground-facing panels darken and the contrast reads as gloss.
     * Car panels are flat, so the face normal comes from how the position
     * changes across the screen. Everything that uses screen-space
     * derivatives runs before the discard below. */
    float3 env   = float3(0.0, 0.0, 0.0);
    float  k     = 0.0;
    float  glint = 0.0;
    float3 faceN = cross(ddx(input.objPos), ddy(input.objPos));
    float  nLen  = length(faceN);
    if (carShine > 0.0 && nLen > 1e-12)
    {
        float3 V   = normalize(input.toCam);
        float3 N   = faceN / nLen;
        float  ndv = dot(N, V);
        if (ndv < 0.0) { N = -N; ndv = -ndv; }
        float3 R   = reflect(-V, N);

        float3 up  = normalize(input.up);
        float3 ex  = normalize(input.xAxis - up * dot(input.xAxis, up));
        float3 ey  = cross(up, ex);
        float  el  = dot(R, up);
        float  az  = atan2(dot(R, ey), dot(R, ex));
        float  elWidth = max(fwidth(el), 0.002);

        env = car_environment(el, az, elWidth, saturate(carDark));
        float fresnel = 0.25 + 0.75 * pow(1.0 - saturate(ndv), 4.0);
        k = saturate(carShine * 0.6 * fresnel);

        float3 sunDir = normalize(ex * 0.60 + ey * 0.45 + up * 0.66);
        float  sd     = saturate(dot(R, sunDir));
        glint = (pow(sd, 400.0) * 3.0 + pow(sd, 24.0) * 0.25) * saturate(carShine) * carGlint;
    }

    float4 texColor   = meshTexture.Sample(meshSampler, input.uv);
    float4 finalColor = texColor * input.color;

    if (finalColor.a < 0.01)
        discard;

    finalColor.rgb = saturate(lerp(finalColor.rgb, env, k) + glint);

    finalColor.rgb = pow(max(finalColor.rgb, 0.00001), 1.0 / gamma);

    float fogDepthAdj = max(0.0, input.fogDepth - fogStart);
    float fogFactor   = saturate(1.0 - exp(-fogDensity * fogDensity * fogDepthAdj * fogDepthAdj));
    finalColor.rgb = lerp(finalColor.rgb, fogColor.rgb, fogFactor);

    float lum = dot(finalColor.rgb, float3(0.2126, 0.7152, 0.0722));
    finalColor.rgb = lerp(float3(lum, lum, lum), finalColor.rgb, saturation);

    finalColor.rgb = saturate((finalColor.rgb - 0.5) * contrast + 0.5);
    finalColor.rgb = saturate(finalColor.rgb + brightness);

    return finalColor;
}
