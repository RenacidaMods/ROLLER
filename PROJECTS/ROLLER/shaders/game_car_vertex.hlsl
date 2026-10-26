cbuffer MeshVertexUniforms : register(b0, space1)
{
    float4x4 mvpMatrix;
    /* Car shine, all in the car's own coordinates: */
    float4   camPosObj;   /* camera position */
    float4   upObj;       /* world "up" */
    float4   xAxisObj;    /* a fixed world horizontal axis (orients the environment) */
};

struct VertexInput
{
    float3 position : TEXCOORD0;
    float2 uv       : TEXCOORD1;
    float4 color    : TEXCOORD2;
};

struct VertexOutput
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

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.position = mul(mvpMatrix, float4(input.position, 1.0));
    output.uv       = input.uv;
    output.color    = input.color;
    output.fogDepth = output.position.w;
    output.objPos   = input.position;
    output.toCam    = camPosObj.xyz - input.position;
    output.up       = upObj.xyz;
    output.xAxis    = xAxisObj.xyz;
    return output;
}
