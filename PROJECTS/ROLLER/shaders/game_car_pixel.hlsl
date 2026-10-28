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
    float  carSmooth;  /* flat panels (0) to smoothed corner normals (1, default) */
    float  carBevel;   /* width of the rounded strip along panel edges, 0-1 */
    float  carSun;     /* sun shading strength (same sun as the world), 0 = off */
    float4 sunDir;     /* world shaders only */
    float4 camPos;     /* world shaders only */
    float4 shadowParams;  /* z: this car's darkening (tunnels), 0 = none */
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
    float3 smoothN  : TEXCOORD7;
    float2 edge     : TEXCOORD8;
    float3 bevelN   : TEXCOORD9;
    float3 sunObj   : TEXCOORD10;
};

/* The sun in the car's coordinates: passed in for race cars (the same sun
 * as the world shading and shadows); menu cars pass none and get the old
 * fixed direction built from up and the X axis. */
float3 carSunDir(PixelInput input, float3 up, float3 ex, float3 ey)
{
    if (dot(input.sunObj, input.sunObj) > 0.25)
        return normalize(input.sunObj);
    return normalize(ex * 0.60 + ey * 0.45 + up * 0.66);
}

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
    float  groundW = 0.0;   /* how much of env is reflected ground */
    float3 faceN = cross(ddx(input.objPos), ddy(input.objPos));
    float  nLen  = length(faceN);
    if (carShine > 0.0 && nLen > 1e-12)
    {
        float3 V   = normalize(input.toCam);
        float3 N   = faceN / nLen;
        if (dot(N, V) < 0.0) N = -N;
        /* Underside: panels that really face the ground (sills, the
         * underside) sit in the car's own shade, so the shine fades out as
         * the flat panel tips downward: full down to about 9 degrees, gone
         * by about 30. Uses the panel's own facing, not the smoothed one
         * below, so upright doors keep all their shine. */
        float3 up      = normalize(input.up);
        float  lowFade = smoothstep(-0.5, -0.15, dot(N, up));
        /* Smoothing: blend towards the corner normals averaged over
         * neighbouring panels (game_render_hw_car_corner_normals), so
         * reflections flow round edges instead of jumping at them. */
        float  sLen = length(input.smoothN);
        if (carSmooth > 0.0 && sLen > 1e-6)
        {
            float3 S = input.smoothN / sLen;
            if (dot(S, N) < 0.0) S = -S;
            N = normalize(lerp(N, S, carSmooth));
        }
        /* Bevel: in a thin strip along the panel's edges (up to 15% of the
         * panel in from each side at full slider) the normal leans towards
         * the fully blended corner normal, so even sharp edges catch a
         * rounded highlight while the panel's middle stays flat. */
        float  bLen = length(input.bevelN);
        if (carBevel > 0.0 && bLen > 1e-6)
        {
            float2 e  = input.edge;
            float  d  = min(min(e.x, 1.0 - e.x), min(e.y, 1.0 - e.y));
            float  bw = 1.0 - smoothstep(0.0, 0.15 * carBevel, d);
            float3 Bn = input.bevelN / bLen;
            if (dot(Bn, N) < 0.0) Bn = -Bn;
            N = normalize(lerp(N, Bn, bw));
        }
        float  ndv = saturate(dot(N, V));
        float3 R   = reflect(-V, N);

        float3 ex  = normalize(input.xAxis - up * dot(input.xAxis, up));
        float3 ey  = cross(up, ex);
        float  el  = dot(R, up);
        float  az  = atan2(dot(R, ey), dot(R, ex));
        float  elWidth = max(fwidth(el), 0.002);

        env = car_environment(el, az, elWidth, saturate(carDark));
        groundW = saturate(0.5 - el / elWidth);
        float fresnel = 0.25 + 0.75 * pow(1.0 - saturate(ndv), 4.0);
        float3 sunDir = carSunDir(input, up, ex, ey);
        float  ndl    = dot(N, sunDir);
        /* Reflections are dimmer on the side turned away from the sun, so
         * the shaded side doesn't glow. */
        k = saturate(carShine * 0.6 * fresnel) * lowFade * lerp(0.45, 1.0, smoothstep(-0.25, 0.5, ndl));

        /* Sun highlight on the panels facing the sun. A true mirror glint
         * only shows when looking towards the sun, which put the shine on
         * the shaded side and none on the sunlit side when the sun was
         * behind the camera; it read as back to front. The half vector
         * leans towards the sun, so the highlight sits on the sunlit
         * panels and only slides a little with the view. */
        float3 H  = normalize(sunDir + V * 0.35);
        float  nh = saturate(dot(N, H));
        glint = (pow(nh, 300.0) * 2.0 + pow(nh, 16.0) * 0.22)
              * saturate(ndl * 3.0) * saturate(carShine) * carGlint * lowFade;
    }

    /* Sun shading: half-Lambert from the same sun as the world shading
     * (and the glint above), on the smoothed normal so it rolls round the
     * body; the paint darkens on the side away from the sun. */
    float sunK = 1.0;
    if (carSun > 0.0 && nLen > 1e-12)
    {
        float3 Vs  = normalize(input.toCam);
        float3 Ns  = faceN / nLen;
        if (dot(Ns, Vs) < 0.0) Ns = -Ns;
        float  sLn = length(input.smoothN);
        if (carSmooth > 0.0 && sLn > 1e-6)
        {
            float3 S = input.smoothN / sLn;
            if (dot(S, Ns) < 0.0) S = -S;
            Ns = normalize(lerp(Ns, S, carSmooth));
        }
        float3 upS = normalize(input.up);
        float3 exS = normalize(input.xAxis - upS * dot(input.xAxis, upS));
        float3 eyS = cross(upS, exS);
        float3 sunS = carSunDir(input, upS, exS, eyS);
        float  k      = carSun * 0.75;
        float  wrap   = saturate(dot(Ns, sunS) * 0.5 + 0.5);
        float  wrapUp = saturate(dot(upS, sunS) * 0.5 + 0.5);   /* like the world: roof = flat ground */
        sunK = (1.0 - k * (1.0 - wrap)) / (1.0 - k * (1.0 - wrapUp));
    }

    float4 texColor   = meshTexture.Sample(meshSampler, input.uv);
    float4 finalColor = texColor * input.color;
    finalColor.rgb *= sunK;

    if (finalColor.a < 0.01)
        discard;

    /* Reflected ground only ever darkens the paint: a grey ground lifted
     * dark paint on the lower panels, where the car should look darkest. */
    float3 target = lerp(env, min(finalColor.rgb, env), groundW);
    finalColor.rgb = saturate(lerp(finalColor.rgb, target, k) + glint);
    /* In a tunnel the whole car dims, reflection and glint included. */
    finalColor.rgb *= 1.0 - saturate(shadowParams.z);

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
