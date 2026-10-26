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

float4 main(PixelInput input) : SV_Target
{
    float2 uv = input.uv;
    float d = loadDepth(uv);
    if (d >= 0.9999999)
        return float4(1.0, 1.0, 1.0, 1.0);   // sky

    float3 P = viewPos(uv, d);

    // Face normal from neighbouring depths, taking the side with the smaller
    // depth step on each axis so silhouettes don't produce bogus normals.
    float2 texel = 1.0 / depthSize;
    float3 Pr = viewPos(uv + float2(texel.x, 0.0), loadDepth(uv + float2(texel.x, 0.0)));
    float3 Pl = viewPos(uv - float2(texel.x, 0.0), loadDepth(uv - float2(texel.x, 0.0)));
    float3 Pd = viewPos(uv + float2(0.0, texel.y), loadDepth(uv + float2(0.0, texel.y)));
    float3 Pu = viewPos(uv - float2(0.0, texel.y), loadDepth(uv - float2(0.0, texel.y)));
    float3 dx = (abs(Pr.z - P.z) < abs(P.z - Pl.z)) ? (Pr - P) : (P - Pl);
    float3 dy = (abs(Pd.z - P.z) < abs(P.z - Pu.z)) ? (Pd - P) : (P - Pu);
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

    float bias = max(radius * 0.03, P.z * 0.0015);
    float occlusion = 0.0;
    [unroll]
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float3 k = kKernel[i];
        float3 s = P + (T * k.x + B * k.y + N * k.z) * radius;
        if (s.z <= zNear)
            continue;
        float2 suv = projectUV(s);
        if (any(suv < 0.0) || any(suv > 1.0))
            continue;
        float sd = loadDepth(suv);
        float sceneZ = viewZ(sd);
        float inRange = smoothstep(0.0, 1.0, radius / max(abs(P.z - sceneZ), 1e-3));
        // Angle bias: only surfaces rising well above this point's own plane
        // occlude. Without it, the shallow folds where road sections meet on
        // a hill (a few degrees each) showed as dark lines across the road;
        // real corners (wall and road, under cars) rise much more steeply.
        float3 v = viewPos(suv, sd) - P;
        float elevation = dot(v, N) / max(length(v), 1e-3);
        float angleTerm = saturate((elevation - 0.3) / 0.4);
        occlusion += (sceneZ <= s.z - bias ? 1.0 : 0.0) * inRange * angleTerm;
    }

    float ao = 1.0 - occlusion / SAMPLE_COUNT;
    ao = lerp(ao, 1.0, saturate((P.z - fadeStart) / max(fadeEnd - fadeStart, 1.0)));
    return float4(ao, ao, ao, 1.0);
}
