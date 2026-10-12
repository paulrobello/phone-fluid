#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

#define MAX_TOUCHES 4

/// One MLS-MPM particle. Positions are in grid cells, velocities in cells per sim time unit.
typedef struct {
    simd_float3 position;
    simd_float3 velocity;
    simd_float3 c0;  // affine velocity matrix C, column 0
    simd_float3 c1;
    simd_float3 c2;
} Particle;

typedef struct {
    simd_float4 positionRadius;  // xy = position (cells), z = radius (cells), w = strength 0..1
    simd_float4 velocity;        // xy = drag velocity (cells / sim time), z = splash impulse
} TouchInput;

typedef struct {
    simd_uint3 gridSize;
    uint32_t particleCount;
    simd_float3 gravity;  // cells / sim time^2
    float dt;
    float restDensity;
    float stiffness;
    float eosPower;
    float viscosity;
    float cohesion;  // allowed negative pressure as a fraction of stiffness
    float wallMargin;
    float maxSpeed;
    float wallStiffness;  // soft wall push strength in g2p
    float affineDamping;  // scales the APIC affine matrix C each step (1 = none)
    float drag;           // fraction of grid velocity removed per step
    float foamSpeed;      // particle speed where whitewater starts to form
    float foamDecay;      // per-step foam retention
    uint32_t touchCount;
    TouchInput touches[MAX_TOUCHES];
} SimParams;

/// Reduced once per frame on the GPU, read back by the CPU one frame later.
typedef struct {
    float kineticEnergy;
    float impactEnergy;
    float maxSpeed;
    float particleCount;
} FrameStats;

typedef struct {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 inverseProjection;
    simd_float4 eyeWorld;      // xyz = eye position (mm, world)
    simd_float4 boxMin;        // world mm
    simd_float4 boxMax;
    simd_float4 lightDirView;  // xyz = direction toward the light, view space
    simd_float4 lightDirWorld;
    simd_float4 lightColor;    // rgb * intensity
    simd_float4 ambientColor;
    simd_float4 absorption;    // rgb absorption per mm, w = refraction strength
    simd_float4 tint;          // rgb tint, w = opacity floor
    simd_float4 emission;      // rgb emission, w = emission strength
    simd_float4 material;      // x = reflectivity F0, y = metallic, z = roughness, w = foam amount
    simd_float2 viewportSize;  // pixels
    float cellMM;
    float particleRadiusMM;
    float time;
    uint32_t backdrop;  // 0 checker, 1 glass, 2 photo, 3 void
    float foamSpeed;
    float causticStrength;
    simd_float4 simOrigin;  // world mm position of sim cell (0,0,0)
} RenderUniforms;

#endif /* ShaderTypes_h */
