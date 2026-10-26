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

// 4x4 Bayer order: the kernel rotation repeats every 4 pixels, so the
// composite pass's 4x4 blur averages each pixel over all 16 rotations and
// the sampling noise cancels out instead of showing as grain.
static const float kBayer4[16] = {
     0.0,  8.0,  2.0, 10.0,
    12.0,  4.0, 14.0,  6.0,
     3.0, 11.0,  1.0,  9.0,
    15.0,  7.0, 13.0,  5.0
};

float viewZ(float d) { return zB / (d - zF); }

float loadDepth(float2 uv)
{
    int2 px = int2(clamp(uv * depthSize, float2(0.0, 0.0), depthSize - 1.0));
    return depthTex.Load(int3(px, 0));
}

float3 viewPos(float2 uv, float d)
{
    float z = viewZ(d);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * z / fov.x, (ndc.y - shiftY) * z / fov.y, z);
}

float2 projectUV(float3 p)
{
    float2 ndc = float2(fov.x * p.x / p.z, fov.y * p.y / p.z + shiftY);
    return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

// One tangent step along off (one texel), for the face normal. Takes the
// side with the smaller depth step so silhouettes don't produce bogus
// normals, then, where the surface carries on smoothly, measures over three
// texels instead: far away the depth buffer only resolves z to a few units,
// and on a wall receding sideways those steps showed as vertical AO stripes
// that followed the camera. Over three texels they matter a third as much.
float3 tangentStep(float2 uv, float2 off, float3 P)
{
    float3 a1 = viewPos(uv + off, loadDepth(uv + off));
    float3 b1 = viewPos(uv - off, loadDepth(uv - off));
    bool useA = abs(a1.z - P.z) < abs(P.z - b1.z);
    float3 one = useA ? (a1 - P) : (P - b1);
    float2 off3 = useA ? off * 3.0 : -off * 3.0;
    float3 p3 = viewPos(uv + off3, loadDepth(uv + off3));
    float3 three = useA ? (p3 - P) : (P - p3);
    bool sameSurface = abs(three.z - 3.0 * one.z) < max(abs(one.z) * 1.5, P.z * 0.002);
    return sameSurface ? three / 3.0 : one;
}

float4 main(PixelInput input) : SV_Target
{
    float2 uv = input.uv;
    float d = loadDepth(uv);
    if (d >= 0.9999999)
        return float4(1.0, 1.0, 1.0, 1.0);   // sky

    float3 P = viewPos(uv, d);

    // Face normal from neighbouring depths (see tangentStep).
    float2 texel = 1.0 / depthSize;
    float3 dx = tangentStep(uv, float2(texel.x, 0.0), P);
    float3 dy = tangentStep(uv, float2(0.0, texel.y), P);
    float3 N = normalize(cross(dx, dy));
    if (dot(N, -P) < 0.0)
        N = -N;

    // Per-pixel rotation of the kernel from the 4x4 Bayer pattern.
    int2 cell = int2(input.position.xy) & 3;
    float angle = 6.2831853 * (kBayer4[cell.y * 4 + cell.x] + 0.5) / 16.0;
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
        float sd = loadDepth(suv);
        float sceneZ = viewZ(sd);
        float inRange = smoothstep(0.0, 1.0, r / max(abs(P.z - sceneZ), 1e-3));
        landed += 1.0;
        // Angle bias: only surfaces rising well above this point's own plane
        // occlude. Without it, the shallow folds where road sections meet on
        // a hill (a few degrees each) showed as dark lines across the road;
        // real corners (wall and road, under cars) rise much more steeply.
        float3 v = viewPos(suv, sd) - P;
        float elevation = dot(v, N) / max(length(v), 1e-3);
        float angleTerm = saturate((elevation - 0.3) / 0.4);
        occlusion += (sceneZ <= s.z - bias ? 1.0 : 0.0) * inRange * angleTerm;
    }

    float ao = landed >= 4.0 ? 1.0 - occlusion / landed : 1.0;
    ao = lerp(ao, 1.0, saturate((P.z - fadeStart) / max(fadeEnd - fadeStart, 1.0)));
    return float4(ao, ao, ao, 1.0);
}
