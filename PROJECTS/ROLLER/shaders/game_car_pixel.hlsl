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
    float  _pad;
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
};

float4 main(PixelInput input) : SV_Target
{
    /* Car panels are flat, so the face normal comes from how the position
     * changes across the screen. Taken before any discard. */
    float3 faceN = cross(ddx(input.objPos), ddy(input.objPos));

    float4 texColor   = meshTexture.Sample(meshSampler, input.uv);
    float4 finalColor = texColor * input.color;

    if (finalColor.a < 0.01)
        discard;

    /* Car shine: a rim of light where a panel turns away from the camera
     * (Fresnel) plus a soft sky gloss on upward-facing panels (bonnet,
     * roof), screen-blended so colours can't blow out. No sharp sun
     * highlight: on big flat panels it would flash a whole panel at once. */
    float nLen = length(faceN);
    if (carShine > 0.0 && nLen > 1e-12)
    {
        float3 N   = faceN / nLen;
        float3 V   = normalize(input.toCam);
        float  ndv = dot(N, V);
        if (ndv < 0.0) { N = -N; ndv = -ndv; }
        float rim    = pow(1.0 - saturate(ndv), 3.0);
        float facing = saturate(dot(N, normalize(input.up)));
        float amount = saturate(carShine * (0.55 * rim + 0.30 * facing * facing));
        float3 tint  = lerp(float3(1.0, 1.0, 1.0), fogColor.rgb, 0.35);
        finalColor.rgb = 1.0 - (1.0 - finalColor.rgb) * (1.0 - amount * tint);
    }

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
