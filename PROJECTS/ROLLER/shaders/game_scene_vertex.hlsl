cbuffer SceneVertexUniforms : register(b0, space1)
{
    float4x4 mvpMatrix;
};

struct VertexInput
{
    float3 position : TEXCOORD0;
    float2 uv       : TEXCOORD1;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
    float  fogDepth : TEXCOORD1;
    float3 worldPos : TEXCOORD2;   /* world quads: VP only, so input is world space */
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.position = mul(mvpMatrix, float4(input.position, 1.0));
    output.uv       = input.uv;
    output.fogDepth = output.position.w;
    output.worldPos = input.position;
    return output;
}
