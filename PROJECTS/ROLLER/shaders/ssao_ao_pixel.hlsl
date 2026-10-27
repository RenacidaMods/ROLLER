// Screen-space ambient occlusion, pass 1 of 2 (ssao.c): renders a
// half-resolution occlusion term (1 = open, 0 = fully occluded) from the
// scene's depth snapshot. View-space positions are rebuilt from depth using
// the same projection as build_mvp in scene_render_gpu.c:
//   ndc.x = fovX * x / z,  ndc.y = fovY * y / z + shiftY,  depth = zF + zB / z

Texture2D<float> depthTex     : register(t0, space2);
SamplerState     depthSampler : register(s0, space2);

cbuffer SSAOUniforms : register(b0, space3)
{
    float2 depthSize;   // full-resolution depth size in pixels
    float2 fov;         // fovX, fovY
    float  shiftY;      // horizon shift added to ndc.y
    float  zF;          // depth = zF + zB / viewZ
    float  zB;
    float  radius;      // sample radius in world units
    float  fadeStart;   // view distance where AO starts fading out
    float  fadeEnd;     // view distance where AO has gone
    float  zNear;
    float  _pad;
};

struct PixelInput
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

#define SAMPLE_COUNT 16

// Hemisphere kernel (z = along the surface normal), denser near the centre.
static const float3 kKernel[SAMPLE_COUNT] = {
    float3( 0.53, 0.14, 0.38), float3(-0.27, 0.48, 0.21), float3(-0.11,-0.36, 0.56),
    float3( 0.24,-0.18, 0.22), float3( 0.06, 0.27, 0.12), float3(-0.46,-0.05, 0.35),
    float3( 0.38, 0.52, 0.47), float3(-0.62, 0.31, 0.58), float3( 0.71,-0.33, 0.29),
    float3(-0.21,-0.74, 0.43), float3( 0.12, 0.81, 0.48), float3(-0.85,-0.38, 0.27),
    float3( 0.17, 0.09, 0.81), float3(-0.34,-0.21, 0.72), float3( 0.58,-0.61, 0.41),
    float3(-0.04, 0.37, 0.88)
};

// Per-pixel kernel rotation from interleaved gradient noise (Jimenez 2014).
// A repeating 4x4 rotation tile left a faint 4-texel grid after the blur;
// with the scene drawn larger than the window (render scale) and scaled
// down, that grid beat into broad bands and a plaid pattern fixed to the
// screen. This noise doesn't repeat, so what the blur leaves is fine grain.
float rotationNoise(float2 pixel)
{
    return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

float viewZ(float d) { return zB / (d - zF); }

// Depth work is done on whole depth texels, with positions taken at texel
// centres. This pass runs at half size, so each pixel's uv lands exactly on
// the border between two depth texels; rounding uv picked the left or right
// one differently from column to column (and row to row), so a neighbour
// could come back as the same texel. That made a receding wall or roof look
// flat-on to the camera in those columns and rows, and they darkened: crisp
// stripes fixed to the screen that changed with the window size.
int2 clampTexel(int2 px)
{
    return clamp(px, int2(0, 0), int2(depthSize) - 1);
}

int2 uvTexel(float2 uv)
{
    return clampTexel(int2(floor(uv * depthSize)));
}

float loadDepthTexel(int2 px)
{
    return depthTex.Load(int3(px, 0));
}

float3 viewPos(float2 uv, float d)
{
    float z = viewZ(d);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * z / fov.x, (ndc.y - shiftY) * z / fov.y, z);
}

// View-space position of a depth texel's centre (px must be in range).
float3 texelPos(int2 px)
{
    return viewPos((float2(px) + 0.5) / depthSize, loadDepthTexel(px));
}

float2 projectUV(float3 p)
{
    float2 ndc = float2(fov.x * p.x / p.z, fov.y * p.y / p.z + shiftY);
    return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

// One tangent step along off (one texel), for the face normal. Takes the
// side with the smaller depth step so silhouettes don't produce bogus
// normals (or the side that's on screen, at the edges), then, where the
// surface carries on smoothly, measures over three texels instead: far away
// the depth buffer only resolves z to a few units, so a longer baseline
// keeps the normal steady.
float3 tangentStep(int2 px, int2 off, float3 P)
{
    int2 maxPx = int2(depthSize) - 1;
    bool hasA = all(px + off <= maxPx) && all(px + off >= 0);
    bool hasB = all(px - off <= maxPx) && all(px - off >= 0);
    float3 a1 = texelPos(clampTexel(px + off));
    float3 b1 = texelPos(clampTexel(px - off));
    bool useA = hasA && (!hasB || abs(a1.z - P.z) < abs(P.z - b1.z));
    float3 one = useA ? (a1 - P) : (P - b1);
    int2 px3 = clampTexel(useA ? px + off * 3 : px - off * 3);
    int steps = abs(px3.x - px.x) + abs(px3.y - px.y);
    if (steps < 2)
        return one;
    float3 p3 = texelPos(px3);
    float3 longStep = useA ? (p3 - P) : (P - p3);
    float k = (float)steps;
    bool sameSurface = abs(longStep.z - k * one.z) < max(abs(one.z) * 1.5, P.z * 0.002);
    return sameSurface ? longStep / k : one;
}

float4 main(PixelInput input) : SV_Target
{
    // This half-size pixel covers depth texels 2x..2x+1; use the first.
    int2 px = clampTexel(int2(floor(input.position.xy)) * 2);
    float d = loadDepthTexel(px);
    if (d >= 0.9999999)
        return float4(1.0, 1.0, 1.0, 1.0);   // sky

    float3 P = texelPos(px);

    // Face normal from neighbouring depths (see tangentStep).
    float3 dx = tangentStep(px, int2(1, 0), P);
    float3 dy = tangentStep(px, int2(0, 1), P);
    float3 N = normalize(cross(dx, dy));
    if (dot(N, -P) < 0.0)
        N = -N;

    // Per-pixel rotation of the kernel (see rotationNoise).
    float angle = 6.2831853 * rotationNoise(floor(input.position.xy));
    float3 rv = float3(cos(angle), sin(angle), 0.0);
    float3 T = rv - N * dot(rv, N);
    T = (dot(T, T) < 1e-4) ? normalize(cross(N, float3(0.0, 0.0, 1.0)) + 1e-3) : normalize(T);
    float3 B = cross(N, T);

    // Close to the camera a fixed world radius covers a large part of the
    // screen, so many samples landed off-screen; on a wall beside the camera
    // that drew a hard-edged region and a grid of lines. Cap the radius at
    // 6% of the screen width (projected), and average over the samples that
    // actually landed rather than counting skipped ones as open.
    float pxPerUnit = fov.x * 0.5 * depthSize.x / P.z;
    float r = min(radius, (0.06 * depthSize.x) / max(pxPerUnit, 1e-6));
    float bias = max(r * 0.03, P.z * 0.0015);
    float occlusion = 0.0;
    float landed = 0.0;
    [unroll]
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float3 k = kKernel[i];
        float3 s = P + (T * k.x + B * k.y + N * k.z) * r;
        if (s.z <= zNear)
            continue;
        float2 suv = projectUV(s);
        if (any(suv < 0.0) || any(suv > 1.0))
            continue;
        int2 spx = uvTexel(suv);
        float3 scene = texelPos(spx);
        float sceneZ = scene.z;
        float inRange = smoothstep(0.0, 1.0, r / max(abs(P.z - sceneZ), 1e-3));
        landed += 1.0;
        // Angle bias: only surfaces rising well above this point's own plane
        // occlude. Without it, the shallow folds where road sections meet on
        // a hill (a few degrees each) showed as dark lines across the road;
        // real corners (wall and road, under cars) rise much more steeply.
        float3 v = scene - P;
        float height = dot(v, N);
        float elevation = height / max(length(v), 1e-3);
        float angleTerm = saturate((elevation - 0.3) / 0.4);
        // Height test: the occluder must also stand a real distance above
        // this surface. Neighbouring track sections often overlap or step by
        // a few units; those lips are steep, so the angle test alone let
        // each one darken a radius-wide band beside the join (stripes along
        // wall and tunnel-roof sections).
        float minHeight = max(r * 0.08, P.z * 0.002);
        float heightTerm = saturate((height - minHeight) / minHeight);
        occlusion += (sceneZ <= s.z - bias ? 1.0 : 0.0) * inRange * angleTerm * heightTerm;
    }

    float ao = landed >= 4.0 ? 1.0 - occlusion / landed : 1.0;
    ao = lerp(ao, 1.0, saturate((P.z - fadeStart) / max(fadeEnd - fadeStart, 1.0)));
    return float4(ao, ao, ao, 1.0);
}
