/*
 * gpu_mesh.cu — GPU-accelerated particle-to-mesh surface reconstruction
 *
 * Replaces SplashSurf for converting fluid particle positions (.ply)
 * into triangle meshes via density splatting + marching cubes, all on GPU.
 *
 * Build:  nvcc -O3 -arch=sm_80 -o gpu_mesh gpu_mesh.cu
 * Usage:  ./gpu_mesh --input particles.ply --output meshes/ --resolution 384
 *
 * (c) 2026  —  Fluid Simulation Project, Phase 11
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Marching Cubes lookup tables  (Paul Bourke / Lorensen & Cline)
// ---------------------------------------------------------------------------

// Edge table: for each of 256 cube configurations, a 12-bit mask indicating
// which edges are intersected by the isosurface.
static const int MC_EDGE_TABLE[256] = {
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

// Triangle table: for each of 256 cube configurations, up to 5 triangles
// (15 edge indices terminated by -1).
static const int MC_TRI_TABLE[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1},
    { 3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1},
    { 3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    { 3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1},
    { 9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1},
    { 2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1, -1},
    { 8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1},
    { 4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1, -1},
    { 3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1, -1},
    { 1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1, -1},
    { 4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1, -1},
    { 4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1},
    { 5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1, -1},
    { 2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1, -1},
    { 9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1},
    { 0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1},
    { 2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1, -1},
    {10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1, -1},
    { 5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1, -1},
    { 5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1},
    { 9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1},
    { 1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1, -1},
    {10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1, -1},
    { 8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1, -1},
    { 2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1},
    { 7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1, -1},
    { 2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1},
    {11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1, -1},
    { 5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0, -1},
    {11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0, -1},
    {11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1, -1},
    { 9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1, -1},
    { 2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1, -1},
    { 6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1},
    { 3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1},
    { 6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1},
    {10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1, -1},
    { 6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1, -1},
    { 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1, -1},
    { 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1},
    { 3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1, -1},
    { 0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1},
    { 9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6, -1},
    { 8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1, -1},
    { 5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11, -1},
    { 0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7, -1},
    { 6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1},
    {10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1},
    {10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1},
    { 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1},
    { 1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1, -1},
    { 0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1},
    {10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1, -1},
    { 3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1, -1},
    { 6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1, -1},
    { 9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1, -1},
    { 8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1, -1},
    { 3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1, -1},
    { 6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1},
    { 0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1, -1},
    {10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1, -1},
    {10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1, -1},
    { 2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9, -1},
    { 7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1},
    { 7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1, -1},
    { 2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1},
    { 1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11, -1},
    {11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1, -1},
    { 8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6, -1},
    { 0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1, -1},
    { 7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    {10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1},
    { 2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1},
    { 6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1, -1},
    { 7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1, -1},
    { 2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1, -1},
    {10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1, -1},
    {10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1, -1},
    { 0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1, -1},
    { 7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1, -1},
    { 6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1, -1},
    { 8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1, -1},
    { 6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1, -1},
    { 4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1, -1},
    {10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3, -1},
    { 8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1},
    { 0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1, -1},
    { 1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1, -1},
    { 8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1, -1},
    {10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1, -1},
    { 4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3, -1},
    {10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1, -1},
    { 9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1},
    { 6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1, -1},
    { 7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1, -1},
    { 3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6, -1},
    { 7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1, -1},
    { 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1, -1},
    { 6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8, -1},
    { 9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1, -1},
    { 1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4, -1},
    { 4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10, -1},
    { 7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1, -1},
    { 6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1, -1},
    { 3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1, -1},
    { 0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1, -1},
    { 6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1, -1},
    { 0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10, -1},
    {11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5, -1},
    { 6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1, -1},
    { 5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1, -1},
    { 9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1, -1},
    { 1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8, -1},
    { 1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6, -1},
    {10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1, -1},
    { 0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1, -1},
    { 5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1, -1},
    {11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1, -1},
    { 9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1, -1},
    { 7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2, -1},
    { 2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1},
    { 8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1, -1},
    { 9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1, -1},
    { 9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2, -1},
    { 1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1, -1},
    { 9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1},
    { 5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1, -1},
    { 0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1, -1},
    {10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4, -1},
    { 2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1, -1},
    { 0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11, -1},
    { 0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5, -1},
    { 9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1, -1},
    { 5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1, -1},
    { 3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9, -1},
    { 5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1, -1},
    { 8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1, -1},
    { 9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1, -1},
    { 1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1, -1},
    { 3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4, -1},
    { 4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1, -1},
    { 9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3, -1},
    {11,  7,  4, 11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1},
    {11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1, -1},
    { 2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1, -1},
    { 9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7, -1},
    { 3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10, -1},
    { 1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1, -1},
    { 4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    { 3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1, -1},
    { 0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1, -1},
    { 1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

// ---------------------------------------------------------------------------
// Device-side constant memory for lookup tables
// ---------------------------------------------------------------------------
__constant__ int d_edgeTable[256];
__constant__ int d_triTable[256 * 16];

// ---------------------------------------------------------------------------
// PLY I/O helpers
// ---------------------------------------------------------------------------
struct ParticleData {
    std::vector<float> pos;   // x,y,z interleaved
    int n;
};

static bool read_ply(const char* path, ParticleData& out) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return false; }

    // Parse ASCII header
    char line[512];
    int num_verts = 0;
    bool got_format = false;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "element vertex", 14) == 0) {
            sscanf(line, "element vertex %d", &num_verts);
        }
        if (strncmp(line, "format binary_little_endian", 26) == 0) {
            got_format = true;
        }
        if (strncmp(line, "end_header", 10) == 0) {
            break;
        }
    }
    if (num_verts <= 0 || !got_format) {
        fprintf(stderr, "Invalid PLY header in %s (verts=%d, format_ok=%d)\n",
                path, num_verts, (int)got_format);
        fclose(fp);
        return false;
    }

    out.n = num_verts;
    out.pos.resize(num_verts * 3);
    size_t read = fread(out.pos.data(), sizeof(float), num_verts * 3, fp);
    fclose(fp);

    if ((int)read != num_verts * 3) {
        fprintf(stderr, "Truncated PLY data in %s: expected %d floats, got %d\n",
                path, num_verts * 3, (int)read);
        return false;
    }
    return true;
}

static bool write_ply(const char* path,
                      const float* verts, int nv,
                      const unsigned int* tris, int nt) {
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", path); return false; }

    // Write ASCII header
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n", nv);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "element face %d\n", nt);
    fprintf(fp, "property list uchar uint vertex_indices\n");
    fprintf(fp, "end_header\n");

    // Vertex data
    fwrite(verts, sizeof(float), nv * 3, fp);

    // Face data: each face = 1 byte (count=3) + 3 x uint32 indices
    for (int i = 0; i < nt; i++) {
        unsigned char three = 3;
        fwrite(&three, 1, 1, fp);
        fwrite(&tris[i * 3], sizeof(unsigned int), 3, fp);
    }

    fclose(fp);
    return true;
}

// ---------------------------------------------------------------------------
// CUDA Kernel 1: Gaussian density splatting
// ---------------------------------------------------------------------------
__global__ void splat_kernel(const float* __restrict__ particles, int N,
                             float* __restrict__ grid,
                             int res, float radius, float inv_sigma2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = particles[idx * 3 + 0];
    float py = particles[idx * 3 + 1];
    float pz = particles[idx * 3 + 2];

    float dx = 1.0f / (float)res;
    float inv_dx = (float)res;

    // Grid cell of particle center
    int ci = (int)(px * inv_dx);
    int cj = (int)(py * inv_dx);
    int ck = (int)(pz * inv_dx);

    // Splatting radius in grid cells
    int r = (int)ceilf(radius * inv_dx) + 1;

    int imin = max(ci - r, 0);
    int imax = min(ci + r, res - 1);
    int jmin = max(cj - r, 0);
    int jmax = min(cj + r, res - 1);
    int kmin = max(ck - r, 0);
    int kmax = min(ck + r, res - 1);

    float cutoff2 = radius * radius;

    for (int i = imin; i <= imax; i++) {
        float gx = ((float)i + 0.5f) * dx;
        float ddx = gx - px;
        float ddx2 = ddx * ddx;
        if (ddx2 > cutoff2) continue;

        for (int j = jmin; j <= jmax; j++) {
            float gy = ((float)j + 0.5f) * dx;
            float ddy = gy - py;
            float ddy2 = ddy * ddy;
            if (ddx2 + ddy2 > cutoff2) continue;

            for (int k = kmin; k <= kmax; k++) {
                float gz = ((float)k + 0.5f) * dx;
                float ddz = gz - pz;
                float dist2 = ddx2 + ddy2 + ddz * ddz;

                if (dist2 < cutoff2) {
                    float w = expf(-dist2 * inv_sigma2);
                    int gidx = i * res * res + j * res + k;
                    atomicAdd(&grid[gidx], w);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA Kernel 2: Marching cubes — count triangles per voxel
// ---------------------------------------------------------------------------
__global__ void mc_count_kernel(const float* __restrict__ grid, int res,
                                float iso, int* __restrict__ tri_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (res - 1) * (res - 1) * (res - 1);
    if (idx >= total) return;

    int r1 = res - 1;
    int i = idx / (r1 * r1);
    int j = (idx / r1) % r1;
    int k = idx % r1;

    // 8 corner values
    float v[8];
    v[0] = grid[(i    ) * res * res + (j    ) * res + (k    )];
    v[1] = grid[(i + 1) * res * res + (j    ) * res + (k    )];
    v[2] = grid[(i + 1) * res * res + (j + 1) * res + (k    )];
    v[3] = grid[(i    ) * res * res + (j + 1) * res + (k    )];
    v[4] = grid[(i    ) * res * res + (j    ) * res + (k + 1)];
    v[5] = grid[(i + 1) * res * res + (j    ) * res + (k + 1)];
    v[6] = grid[(i + 1) * res * res + (j + 1) * res + (k + 1)];
    v[7] = grid[(i    ) * res * res + (j + 1) * res + (k + 1)];

    int cube_index = 0;
    if (v[0] >= iso) cube_index |= 1;
    if (v[1] >= iso) cube_index |= 2;
    if (v[2] >= iso) cube_index |= 4;
    if (v[3] >= iso) cube_index |= 8;
    if (v[4] >= iso) cube_index |= 16;
    if (v[5] >= iso) cube_index |= 32;
    if (v[6] >= iso) cube_index |= 64;
    if (v[7] >= iso) cube_index |= 128;

    if (d_edgeTable[cube_index] == 0) {
        tri_counts[idx] = 0;
        return;
    }

    int count = 0;
    for (int t = 0; t < 15; t += 3) {
        if (d_triTable[cube_index * 16 + t] == -1) break;
        count++;
    }
    tri_counts[idx] = count;
}

// ---------------------------------------------------------------------------
// Device helper: interpolate vertex along an edge
// ---------------------------------------------------------------------------
__device__ void interp_vertex(float iso,
                              float x0, float y0, float z0, float v0,
                              float x1, float y1, float z1, float v1,
                              float& ox, float& oy, float& oz) {
    if (fabsf(v0 - v1) < 1e-10f) {
        ox = x0; oy = y0; oz = z0;
        return;
    }
    float t = (iso - v0) / (v1 - v0);
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    ox = x0 + t * (x1 - x0);
    oy = y0 + t * (y1 - y0);
    oz = z0 + t * (z1 - z0);
}

// ---------------------------------------------------------------------------
// CUDA Kernel 3: Marching cubes — emit triangles
// ---------------------------------------------------------------------------
__global__ void mc_emit_kernel(const float* __restrict__ grid, int res,
                               float iso,
                               const int* __restrict__ tri_offsets,
                               float* __restrict__ out_verts,
                               unsigned int* __restrict__ out_tris) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (res - 1) * (res - 1) * (res - 1);
    if (idx >= total) return;

    int r1 = res - 1;
    int i = idx / (r1 * r1);
    int j = (idx / r1) % r1;
    int k = idx % r1;

    float dx = 1.0f / (float)res;

    // Corner positions
    float cx[8], cy[8], cz[8];
    cx[0] = (i    ) * dx; cy[0] = (j    ) * dx; cz[0] = (k    ) * dx;
    cx[1] = (i + 1) * dx; cy[1] = (j    ) * dx; cz[1] = (k    ) * dx;
    cx[2] = (i + 1) * dx; cy[2] = (j + 1) * dx; cz[2] = (k    ) * dx;
    cx[3] = (i    ) * dx; cy[3] = (j + 1) * dx; cz[3] = (k    ) * dx;
    cx[4] = (i    ) * dx; cy[4] = (j    ) * dx; cz[4] = (k + 1) * dx;
    cx[5] = (i + 1) * dx; cy[5] = (j    ) * dx; cz[5] = (k + 1) * dx;
    cx[6] = (i + 1) * dx; cy[6] = (j + 1) * dx; cz[6] = (k + 1) * dx;
    cx[7] = (i    ) * dx; cy[7] = (j + 1) * dx; cz[7] = (k + 1) * dx;

    // Corner values
    float v[8];
    v[0] = grid[(i    ) * res * res + (j    ) * res + (k    )];
    v[1] = grid[(i + 1) * res * res + (j    ) * res + (k    )];
    v[2] = grid[(i + 1) * res * res + (j + 1) * res + (k    )];
    v[3] = grid[(i    ) * res * res + (j + 1) * res + (k    )];
    v[4] = grid[(i    ) * res * res + (j    ) * res + (k + 1)];
    v[5] = grid[(i + 1) * res * res + (j    ) * res + (k + 1)];
    v[6] = grid[(i + 1) * res * res + (j + 1) * res + (k + 1)];
    v[7] = grid[(i    ) * res * res + (j + 1) * res + (k + 1)];

    int cube_index = 0;
    if (v[0] >= iso) cube_index |= 1;
    if (v[1] >= iso) cube_index |= 2;
    if (v[2] >= iso) cube_index |= 4;
    if (v[3] >= iso) cube_index |= 8;
    if (v[4] >= iso) cube_index |= 16;
    if (v[5] >= iso) cube_index |= 32;
    if (v[6] >= iso) cube_index |= 64;
    if (v[7] >= iso) cube_index |= 128;

    if (d_edgeTable[cube_index] == 0) return;

    // Interpolate edge vertices
    // Edge connectivity: edge i connects corner edge_conn[i][0] to edge_conn[i][1]
    // Edges: 0:(0,1) 1:(1,2) 2:(2,3) 3:(3,0) 4:(4,5) 5:(5,6) 6:(6,7) 7:(7,4)
    //        8:(0,4) 9:(1,5) 10:(2,6) 11:(3,7)
    float evx[12], evy[12], evz[12];
    int edges = d_edgeTable[cube_index];

    if (edges & 1)
        interp_vertex(iso, cx[0],cy[0],cz[0],v[0], cx[1],cy[1],cz[1],v[1], evx[0],evy[0],evz[0]);
    if (edges & 2)
        interp_vertex(iso, cx[1],cy[1],cz[1],v[1], cx[2],cy[2],cz[2],v[2], evx[1],evy[1],evz[1]);
    if (edges & 4)
        interp_vertex(iso, cx[2],cy[2],cz[2],v[2], cx[3],cy[3],cz[3],v[3], evx[2],evy[2],evz[2]);
    if (edges & 8)
        interp_vertex(iso, cx[3],cy[3],cz[3],v[3], cx[0],cy[0],cz[0],v[0], evx[3],evy[3],evz[3]);
    if (edges & 16)
        interp_vertex(iso, cx[4],cy[4],cz[4],v[4], cx[5],cy[5],cz[5],v[5], evx[4],evy[4],evz[4]);
    if (edges & 32)
        interp_vertex(iso, cx[5],cy[5],cz[5],v[5], cx[6],cy[6],cz[6],v[6], evx[5],evy[5],evz[5]);
    if (edges & 64)
        interp_vertex(iso, cx[6],cy[6],cz[6],v[6], cx[7],cy[7],cz[7],v[7], evx[6],evy[6],evz[6]);
    if (edges & 128)
        interp_vertex(iso, cx[7],cy[7],cz[7],v[7], cx[4],cy[4],cz[4],v[4], evx[7],evy[7],evz[7]);
    if (edges & 256)
        interp_vertex(iso, cx[0],cy[0],cz[0],v[0], cx[4],cy[4],cz[4],v[4], evx[8],evy[8],evz[8]);
    if (edges & 512)
        interp_vertex(iso, cx[1],cy[1],cz[1],v[1], cx[5],cy[5],cz[5],v[5], evx[9],evy[9],evz[9]);
    if (edges & 1024)
        interp_vertex(iso, cx[2],cy[2],cz[2],v[2], cx[6],cy[6],cz[6],v[6], evx[10],evy[10],evz[10]);
    if (edges & 2048)
        interp_vertex(iso, cx[3],cy[3],cz[3],v[3], cx[7],cy[7],cz[7],v[7], evx[11],evy[11],evz[11]);

    // Emit triangles
    int tri_base = tri_offsets[idx];

    for (int t = 0; t < 15; t += 3) {
        int e0 = d_triTable[cube_index * 16 + t];
        if (e0 == -1) break;
        int e1 = d_triTable[cube_index * 16 + t + 1];
        int e2 = d_triTable[cube_index * 16 + t + 2];

        // Each triangle gets 3 unique vertices (no dedup — simple approach)
        unsigned int vi = (unsigned int)(tri_base * 3 + (t / 3) * 3);

        out_verts[vi * 3 + 0] = evx[e0];
        out_verts[vi * 3 + 1] = evy[e0];
        out_verts[vi * 3 + 2] = evz[e0];

        out_verts[(vi + 1) * 3 + 0] = evx[e1];
        out_verts[(vi + 1) * 3 + 1] = evy[e1];
        out_verts[(vi + 1) * 3 + 2] = evz[e1];

        out_verts[(vi + 2) * 3 + 0] = evx[e2];
        out_verts[(vi + 2) * 3 + 1] = evy[e2];
        out_verts[(vi + 2) * 3 + 2] = evz[e2];

        out_tris[(tri_base + t / 3) * 3 + 0] = vi;
        out_tris[(tri_base + t / 3) * 3 + 1] = vi + 1;
        out_tris[(tri_base + t / 3) * 3 + 2] = vi + 2;
    }
}

// ---------------------------------------------------------------------------
// Exclusive prefix sum on host (for small-ish arrays we do it on CPU
// to keep the code simple; for very large grids, use thrust::exclusive_scan)
// ---------------------------------------------------------------------------
static void exclusive_scan(const int* in, int* out, int n, int& total) {
    total = 0;
    for (int i = 0; i < n; i++) {
        out[i] = total;
        total += in[i];
    }
}

// ---------------------------------------------------------------------------
// Main meshing pipeline
// ---------------------------------------------------------------------------
static bool process_file(const char* inpath, const char* outpath,
                         int resolution, float radius, float iso) {
    // 1. Read particles
    ParticleData pd;
    if (!read_ply(inpath, pd)) return false;
    printf("  Loaded %d particles from %s\n", pd.n, inpath);

    int res = resolution;
    long long grid_size = (long long)res * res * res;

    // Sanity check
    if (grid_size > 2LL * 1024 * 1024 * 1024) {
        fprintf(stderr, "  Grid too large: %lld voxels (max ~2G)\n", grid_size);
        return false;
    }

    // 2. Upload particles to GPU
    float* d_particles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_particles, pd.n * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_particles, pd.pos.data(),
                          pd.n * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // 3. Allocate and clear density grid on GPU
    float* d_grid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grid, 0, grid_size * sizeof(float)));

    // 4. Splat particles
    float sigma = radius / 3.0f;   // Gaussian sigma: radius ≈ 3 sigma
    float inv_sigma2 = 1.0f / (2.0f * sigma * sigma);

    {
        int block = 256;
        int grid_dim = (pd.n + block - 1) / block;
        printf("  Splatting %d particles (radius=%.4f, res=%d)...\n",
               pd.n, radius, res);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);

        splat_kernel<<<grid_dim, block>>>(d_particles, pd.n, d_grid,
                                          res, radius, inv_sigma2);
        CUDA_CHECK(cudaGetLastError());

        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("  Splat done in %.1f ms\n", ms);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    cudaFree(d_particles);

    // 5. Marching cubes — count triangles
    int num_voxels = (res - 1) * (res - 1) * (res - 1);
    int* d_tri_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tri_counts, num_voxels * sizeof(int)));

    {
        int block = 256;
        int grid_dim = (num_voxels + block - 1) / block;
        mc_count_kernel<<<grid_dim, block>>>(d_grid, res, iso, d_tri_counts);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 6. Download counts, prefix sum on CPU
    std::vector<int> h_counts(num_voxels);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_tri_counts,
                          num_voxels * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_tri_counts);

    std::vector<int> h_offsets(num_voxels);
    int total_tris = 0;
    exclusive_scan(h_counts.data(), h_offsets.data(), num_voxels, total_tris);

    int total_verts = total_tris * 3;
    printf("  Marching cubes: %d triangles, %d vertices\n", total_tris, total_verts);

    if (total_tris == 0) {
        printf("  No surface found — skipping output.\n");
        cudaFree(d_grid);
        return true;
    }

    // 7. Upload offsets, allocate output buffers
    int* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets, num_voxels * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(),
                          num_voxels * sizeof(int), cudaMemcpyHostToDevice));

    float* d_out_verts = nullptr;
    unsigned int* d_out_tris = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_verts, (long long)total_verts * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_tris, (long long)total_tris * 3 * sizeof(unsigned int)));

    // 8. Emit triangles
    {
        int block = 256;
        int grid_dim = (num_voxels + block - 1) / block;

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);

        mc_emit_kernel<<<grid_dim, block>>>(d_grid, res, iso,
                                             d_offsets, d_out_verts, d_out_tris);
        CUDA_CHECK(cudaGetLastError());

        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("  MC emit done in %.1f ms\n", ms);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    cudaFree(d_grid);
    cudaFree(d_offsets);

    // 9. Download results
    std::vector<float> h_verts(total_verts * 3);
    std::vector<unsigned int> h_tris(total_tris * 3);

    CUDA_CHECK(cudaMemcpy(h_verts.data(), d_out_verts,
                          (long long)total_verts * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tris.data(), d_out_tris,
                          (long long)total_tris * 3 * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_out_verts);
    cudaFree(d_out_tris);

    // 10. Write output PLY
    if (!write_ply(outpath, h_verts.data(), total_verts,
                   h_tris.data(), total_tris)) {
        return false;
    }
    printf("  Wrote mesh: %d verts, %d faces -> %s\n",
           total_verts, total_tris, outpath);
    return true;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --input <ply_or_dir> --output <dir> [options]\n"
        "Options:\n"
        "  --resolution <int>    Grid resolution (default: 384)\n"
        "  --radius <float>      Splatting radius in [0,1] coords (default: 0.008)\n"
        "  --iso <float>         Isosurface threshold (default: 10.0)\n"
        "  --help                Show this message\n", prog);
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool is_directory(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

static std::vector<std::string> list_ply_files(const char* dir) {
    std::vector<std::string> files;
    DIR* d = opendir(dir);
    if (!d) return files;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name(ent->d_name);
        if (ends_with(name, ".ply")) {
            files.push_back(std::string(dir) + "/" + name);
        }
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}

static std::string basename_noext(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.rfind('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return name;
}

int main(int argc, char** argv) {
    // Defaults
    std::string input_path;
    std::string output_dir;
    int resolution = 384;
    float radius = 0.008f;
    float iso = 10.0f;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (strcmp(argv[i], "--resolution") == 0 && i + 1 < argc) {
            resolution = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--radius") == 0 && i + 1 < argc) {
            radius = atof(argv[++i]);
        } else if (strcmp(argv[i], "--iso") == 0 && i + 1 < argc) {
            iso = atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (input_path.empty() || output_dir.empty()) {
        usage(argv[0]);
        return 1;
    }

    // Print GPU info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d, %.1f GB)\n", prop.name,
           prop.major, prop.minor, prop.totalGlobalMem / 1e9);
    printf("Config: resolution=%d, radius=%.4f, iso=%.1f\n",
           resolution, radius, iso);

    // Upload lookup tables to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_edgeTable, MC_EDGE_TABLE,
                                   256 * sizeof(int)));
    // Flatten tri table
    int triTableFlat[256 * 16];
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 16; j++)
            triTableFlat[i * 16 + j] = MC_TRI_TABLE[i][j];
    CUDA_CHECK(cudaMemcpyToSymbol(d_triTable, triTableFlat,
                                   256 * 16 * sizeof(int)));

    // Create output directory
    mkdir(output_dir.c_str(), 0755);

    // Collect input files
    std::vector<std::string> input_files;
    if (is_directory(input_path.c_str())) {
        input_files = list_ply_files(input_path.c_str());
        if (input_files.empty()) {
            fprintf(stderr, "No .ply files found in %s\n", input_path.c_str());
            return 1;
        }
        printf("Found %d PLY files in %s\n",
               (int)input_files.size(), input_path.c_str());
    } else {
        input_files.push_back(input_path);
    }

    // Process each file
    int success = 0, fail = 0;
    for (size_t fi = 0; fi < input_files.size(); fi++) {
        const std::string& infile = input_files[fi];
        std::string outname = basename_noext(infile) + "_mesh.ply";
        std::string outfile = output_dir + "/" + outname;

        printf("[%d/%d] %s\n", (int)(fi + 1), (int)input_files.size(),
               infile.c_str());

        if (process_file(infile.c_str(), outfile.c_str(),
                         resolution, radius, iso)) {
            success++;
        } else {
            fail++;
        }
    }

    printf("\nDone: %d succeeded, %d failed\n", success, fail);
    return fail > 0 ? 1 : 0;
}
