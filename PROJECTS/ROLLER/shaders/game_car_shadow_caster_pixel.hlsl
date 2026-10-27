/* Car shadow map caster (scene_render_gpu.c): car bodies are drawn from the
 * sun's view into a depth-only target, so this writes nothing. It takes no
 * inputs, which keeps its Direct3D 12 signature compatible with the car
 * vertex shader's outputs. */
void main()
{
}
