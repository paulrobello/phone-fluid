#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

// MLS-MPM fluid (APIC transfer, quadratic B-spline weights, Tait equation of state).
// Grid nodes hold 4 floats: mass, momentum xyz. Positions are in cell units (dx = 1).

constant float kAPIC = 4.0f;  // 1 / (dx^2 * 1/4) for quadratic kernels with dx = 1

inline uint gridIndex(int3 c, uint3 size) {
    return uint(c.x) + uint(c.y) * size.x + uint(c.z) * size.x * size.y;
}

inline void bsplineWeights(float3 d, thread float3 w[3]) {
    w[0] = 0.5f * (0.5f - d) * (0.5f - d);
    w[1] = 0.75f - d * d;
    w[2] = 0.5f * (0.5f + d) * (0.5f + d);
}

kernel void clearGrid(device float4 *grid [[buffer(0)]],
                      constant SimParams &p [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    uint count = p.gridSize.x * p.gridSize.y * p.gridSize.z;
    if (id >= count) return;
    grid[id] = float4(0.0f);
}

kernel void p2g1(device const Particle *particles [[buffer(0)]],
                 device atomic<float> *grid [[buffer(1)]],
                 constant SimParams &p [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    if (id >= p.particleCount) return;
    Particle pt = particles[id];
    float3 cellIdx = floor(pt.position);
    float3 d = pt.position - cellIdx - 0.5f;
    float3 w[3];
    bsplineWeights(d, w);
    float3x3 C = float3x3(pt.c0, pt.c1, pt.c2);
    int3 base = int3(cellIdx);

    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                float weight = w[gx].x * w[gy].y * w[gz].z;
                int3 cell = base + int3(gx - 1, gy - 1, gz - 1);
                float3 dist = float3(cell) - pt.position + 0.5f;
                float3 momentum = weight * (pt.velocity + C * dist);
                uint gi = gridIndex(cell, p.gridSize) * 4;
                atomic_fetch_add_explicit(&grid[gi + 0], weight, memory_order_relaxed);
                atomic_fetch_add_explicit(&grid[gi + 1], momentum.x, memory_order_relaxed);
                atomic_fetch_add_explicit(&grid[gi + 2], momentum.y, memory_order_relaxed);
                atomic_fetch_add_explicit(&grid[gi + 3], momentum.z, memory_order_relaxed);
            }
        }
    }
}

kernel void p2g2(device const Particle *particles [[buffer(0)]],
                 device atomic<float> *grid [[buffer(1)]],
                 constant SimParams &p [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    if (id >= p.particleCount) return;
    Particle pt = particles[id];
    float3 cellIdx = floor(pt.position);
    float3 d = pt.position - cellIdx - 0.5f;
    float3 w[3];
    bsplineWeights(d, w);
    int3 base = int3(cellIdx);

    float density = 0.0f;
    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                float weight = w[gx].x * w[gy].y * w[gz].z;
                int3 cell = base + int3(gx - 1, gy - 1, gz - 1);
                uint gi = gridIndex(cell, p.gridSize) * 4;
                density += weight * atomic_load_explicit(&grid[gi], memory_order_relaxed);
            }
        }
    }
    density = max(density, 1e-4f);
    float volume = 1.0f / density;
    float pressure = p.stiffness * (pow(density / p.restDensity, p.eosPower) - 1.0f);
    pressure = max(pressure, -p.cohesion * p.stiffness);

    float3x3 C = float3x3(pt.c0, pt.c1, pt.c2);
    float3x3 strain = C + transpose(C);
    float3x3 stress = p.viscosity * strain;
    stress[0][0] -= pressure;
    stress[1][1] -= pressure;
    stress[2][2] -= pressure;
    float3x3 term = (-volume * kAPIC * p.dt) * stress;

    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                float weight = w[gx].x * w[gy].y * w[gz].z;
                int3 cell = base + int3(gx - 1, gy - 1, gz - 1);
                float3 dist = float3(cell) - pt.position + 0.5f;
                float3 momentum = weight * (term * dist);
                uint gi = gridIndex(cell, p.gridSize) * 4;
                atomic_fetch_add_explicit(&grid[gi + 1], momentum.x, memory_order_relaxed);
                atomic_fetch_add_explicit(&grid[gi + 2], momentum.y, memory_order_relaxed);
                atomic_fetch_add_explicit(&grid[gi + 3], momentum.z, memory_order_relaxed);
            }
        }
    }
}

kernel void updateGrid(device const float4 *grid [[buffer(0)]],
                       device float4 *gridVel [[buffer(1)]],
                       constant SimParams &p [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    uint3 size = p.gridSize;
    uint count = size.x * size.y * size.z;
    if (id >= count) return;
    float4 node = grid[id];
    if (node.x <= 0.0f) {
        gridVel[id] = float4(0.0f);
        return;
    }
    float3 v = node.yzw / node.x;
    v += p.gravity * p.dt;

    int3 c = int3(id % size.x, (id / size.x) % size.y, id / (size.x * size.y));
    float3 nodePos = float3(c) + 0.5f;

    for (uint i = 0; i < p.touchCount; i++) {
        TouchInput t = p.touches[i];
        float2 delta = nodePos.xy - t.positionRadius.xy;
        float dist = length(delta);
        float radius = t.positionRadius.z;
        if (dist < radius) {
            float falloff = 1.0f - dist / radius;
            falloff *= falloff;
            float k = clamp(t.positionRadius.w * falloff, 0.0f, 1.0f);
            v.xy = mix(v.xy, t.velocity.xy, k);
            float splash = t.velocity.z * falloff;
            if (splash > 0.0f) {
                float2 radial = dist > 1e-3f ? delta / dist : float2(0.0f, 1.0f);
                v += float3(radial * splash, splash * 0.6f);
            }
        }
    }

    float margin = p.wallMargin;
    float3 lo = float3(margin);
    float3 hi = float3(size) - margin - 1.0f;
    float3 cf = float3(c);
    if (cf.x < lo.x && v.x < 0.0f) v.x = 0.0f;
    if (cf.x > hi.x && v.x > 0.0f) v.x = 0.0f;
    if (cf.y < lo.y && v.y < 0.0f) v.y = 0.0f;
    if (cf.y > hi.y && v.y > 0.0f) v.y = 0.0f;
    if (cf.z < lo.z && v.z < 0.0f) v.z = 0.0f;
    if (cf.z > hi.z && v.z > 0.0f) v.z = 0.0f;

    float speed = length(v);
    if (speed > p.maxSpeed) v *= p.maxSpeed / speed;
    gridVel[id] = float4(v, node.x);
}

kernel void g2p(device Particle *particles [[buffer(0)]],
                device const float4 *gridVel [[buffer(1)]],
                constant SimParams &p [[buffer(2)]],
                device atomic<float> *stats [[buffer(3)]],
                device atomic<uint> *statsBits [[buffer(4)]],
                uint id [[thread_position_in_grid]]) {
    if (id >= p.particleCount) return;
    Particle pt = particles[id];
    float3 cellIdx = floor(pt.position);
    float3 d = pt.position - cellIdx - 0.5f;
    float3 w[3];
    bsplineWeights(d, w);
    int3 base = int3(cellIdx);

    float3 v = float3(0.0f);
    float3x3 B = float3x3(0.0f);
    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                float weight = w[gx].x * w[gy].y * w[gz].z;
                int3 cell = base + int3(gx - 1, gy - 1, gz - 1);
                float3 dist = float3(cell) - pt.position + 0.5f;
                float3 wv = gridVel[gridIndex(cell, p.gridSize)].xyz * weight;
                B[0] += wv * dist.x;
                B[1] += wv * dist.y;
                B[2] += wv * dist.z;
                v += wv;
            }
        }
    }
    float3x3 C = kAPIC * B;

    float3 size = float3(p.gridSize);
    float3 pos = pt.position + v * p.dt;
    pos = clamp(pos, float3(1.0f), size - 2.0f);

    // Soft wall push keeps particles from piling up inside the boundary layer.
    float3 wallMin = float3(p.wallMargin + 0.5f);
    float3 wallMax = size - p.wallMargin - 0.5f;
    float3 predicted = pos + v * p.dt * 3.0f;
    float3 push = max(wallMin - predicted, 0.0f) - max(predicted - wallMax, 0.0f);
    v += 0.3f * push;

    float speed = length(v);
    if (speed > p.maxSpeed) {
        v *= p.maxSpeed / speed;
        speed = p.maxSpeed;
    }

    pt.position = pos;
    pt.velocity = v;
    pt.c0 = C[0];
    pt.c1 = C[1];
    pt.c2 = C[2];
    particles[id] = pt;

    float kinetic = simd_sum(speed * speed);
    float impact = simd_sum(dot(push, push));
    float maxSpeed = simd_max(speed);
    if (simd_is_first()) {
        atomic_fetch_add_explicit(&stats[0], kinetic, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[1], impact, memory_order_relaxed);
        atomic_fetch_max_explicit(&statsBits[2], as_type<uint>(maxSpeed), memory_order_relaxed);
    }
}
