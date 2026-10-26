#ifndef GAME_RENDER_HARDWARE_H
#define GAME_RENDER_HARDWARE_H

#include <SDL3/SDL.h>
#include "scene_render_gpu.h"

/* Forward declarations (defined in game_render.h) */
typedef struct GameRenderCarPose    GameRenderCarPose;
typedef struct GameRenderCarOptions GameRenderCarOptions;

typedef struct GameRendererHardware GameRendererHardware;

GameRendererHardware *game_render_hw_create(SDL_GPUDevice *device);
void                  game_render_hw_destroy(GameRendererHardware *r);

void game_render_hw_draw_car(GameRendererHardware       *r,
                              SceneRendererGPU           *scene,
                              int                         carIdx,
                              const GameRenderCarPose    *pose,
                              const GameRenderCarOptions *options);

/* Draw the 2D name-tag overlay for one car in GPU mode.
 * Must be called after game_render_hw_draw_car, before NamesLeft is decremented.
 * viewSlot: which player's view is currently being queued (0=P1, 1=P2 in
 * 2-player mode; 0 otherwise) -- keys the per-car scrY smoothing state so one
 * player's view doesn't inherit the other's camera-relative computation. */
void game_render_hw_draw_car_name_tag(int carIdx, const GameRenderCarPose *pose, int viewSlot);

/* Reset all name-tag EMA smoothing state (both scrX and scrY, every car/view
 * slot) to uninitialised. Call on any SW<->GPU render-mode switch: while GPU
 * mode is inactive this smoothing state goes stale (frozen at wherever it was
 * last time GPU mode drew a frame) but keeps existing, so resuming GPU mode
 * later would otherwise blend from that stale position toward the current
 * real one over several frames -- visible as tags "flying in" right after a
 * mode switch, the same class of bug as the camera-visibility case handled
 * inside game_render_hw_draw_car_name_tag itself. */
void game_render_hw_reset_name_tags(void);

/* Draw FPS counter into scrbuf at the corner specified by g_iFpsDisplay (1-4; 0=off). */
void game_render_hw_draw_fps_overlay(void);

/* RGBA texture sheet for car carIdx's 3D model (tiles 4 per row plus a white
 * strip), using HD tiles from ../hd_textures when HD Textures is on. Shared
 * by the race renderer and the car-select screen. *outAtlasH is the layout
 * height in 64-pixel units (numRows * 64 + 1) for UV maths, whatever the
 * texture's real size. *outScale (optional) is the sheet's size relative to
 * the original: 1, or 4 for 256-pixel HD tiles, which also get mipmaps.
 * NULL if the car has no texture data. */
SDL_GPUTexture *game_render_hw_build_car_atlas(SDL_GPUDevice *dev, int carIdx,
                                               const tColor *pal,
                                               int *outNumTiles, int *outAtlasH,
                                               int *outScale);

/* Car panels are quads with one texture tile stretched corner to corner.
 * Drawn as two triangles, a tapered panel (the bonnet, say) bends its texture
 * along the shared diagonal. This writes one quad as a
 * GAME_RENDER_HW_QUAD_GRID x GAME_RENDER_HW_QUAD_GRID grid instead, positions
 * and UVs blended bilinearly from the corners, which shrinks the bend below
 * what's visible. Every panel gets the same grid, so neighbouring panels'
 * shared edges get identical vertices (no T-junction cracks).
 *
 * pos/uv corners are in the usual order (triangles 0,1,2 and 0,2,3). verts
 * receives GAME_RENDER_HW_QUAD_VERTS vertices of GAME_RENDER_HW_CAR_VERTEX_FLOATS
 * floats each -- position[3],
 * uv[2], color[4], normal[3] (nrm corners blended and normalised; NULL =
 * zero), edge[2] (the grid's s, t across the panel), bevel[3] (bev corners
 * blended and normalised; NULL = the normal), the layout of both
 * GRHWMeshVertex and the menu's
 * MeshVertex -- and indices GAME_RENDER_HW_QUAD_INDICES entries offset by
 * baseVert. reverseWinding flips every triangle (a back face). */
#define GAME_RENDER_HW_QUAD_GRID    4
#define GAME_RENDER_HW_CAR_VERTEX_FLOATS 17
#define GAME_RENDER_HW_QUAD_VERTS   ((GAME_RENDER_HW_QUAD_GRID + 1) * (GAME_RENDER_HW_QUAD_GRID + 1))
#define GAME_RENDER_HW_QUAD_INDICES (GAME_RENDER_HW_QUAD_GRID * GAME_RENDER_HW_QUAD_GRID * 6)
void game_render_hw_write_car_quad(float *verts, uint32 *indices, uint32 baseVert,
                                   const float pos[4][3], const float uv[4][2],
                                   const float color[4], const float nrm[4][3],
                                   const float bev[4][3], bool reverseWinding);

/* Car shine smoothing: for each polygon of car design designIdx and each of
 * its 4 corners, in the design's game coordinates, all pointing away from
 * the car's centre:
 * - outSmooth: area-weighted average of the polygons meeting at that corner,
 *   each neighbour weighted by the cosine of its angle to this polygon (a
 *   sliding scale that fades to nothing at 90 degrees);
 * - outBevel: every polygon there up to 150 degrees, equally, used only in
 *   a thin strip along panel edges so sharp edges catch a rounded highlight.
 * Both arrays have byNumPols entries. */
void game_render_hw_car_corner_normals(int designIdx, float (*outSmooth)[4][3],
                                       float (*outBevel)[4][3]);

/* Release a texture from game_render_hw_build_car_atlas (HD sheets are shared
 * between cars of the same design). Any other texture is simply released. */
void game_render_hw_release_car_atlas(SDL_GPUDevice *dev, SDL_GPUTexture *tex);

#endif /* GAME_RENDER_HARDWARE_H */
