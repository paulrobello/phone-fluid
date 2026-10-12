#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

// MLS-MPM fluid (APIC transfer, quadratic B-spline weights, Tait equation of state).
// Grid nodes hold 4 floats: mass, momentum xyz. Positions are in cell units (dx = 1).

constant float kAPIC = 4.0f;  // 1 / (dx^2 * 1/4) for quadratic kernels with dx = 1

inline uint gridIndex(int3 c, uint3 size) {
    return uint(c.x) + uint(c.y) * size.x + uint(c.z) * size.x * size.y;
}

inline uint cellOf(float3 pos, uint3 size) {
    int3 c = int3(floor(pos));
    return gridIndex(c, size);
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
                 device float *densities [[buffer(3)]],
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
    densities[id] = density;
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

/// Grid velocity update shared by both P2G paths: momentum → velocity, gravity, touch forces,
/// free-slip walls and a speed clamp.
inline float3 gridNodeVelocity(float4 node, uint id, constant SimParams &p) {
    uint3 size = p.gridSize;
    float3 v = node.yzw / node.x * (1.0f - p.drag);
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

    // Viscous liquids cling to the glass: damp motion in the boundary layer.
    if (any(cf < lo + 1.0f) || any(cf > hi - 1.0f)) v *= 1.0f - p.wallFriction;

    float speed = length(v);
    if (speed > p.maxSpeed) v *= p.maxSpeed / speed;
    return v;
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
    float3 v = gridNodeVelocity(node, id, p);
    gridVel[id] = float4(v, node.x);
}

kernel void g2p(device Particle *particles [[buffer(0)]],
                device const float4 *gridVel [[buffer(1)]],
                constant SimParams &p [[buffer(2)]],
                device atomic<float> *stats [[buffer(3)]],
                device atomic<uint> *statsBits [[buffer(4)]],
                device const float *densities [[buffer(5)]],
                device float *foam [[buffer(6)]],
                device atomic_uint *cellCounts [[buffer(7)]],
                device uint *rank [[buffer(8)]],
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
    float3x3 C = (kAPIC * p.affineDamping) * B;

    float3 size = float3(p.gridSize);
    float3 pos = pt.position + v * p.dt;
    pos = clamp(pos, float3(1.0f), size - 2.0f);

    // Soft wall push keeps particles from piling up inside the boundary layer.
    float3 wallMin = float3(p.wallMargin + 0.5f);
    float3 wallMax = size - p.wallMargin - 0.5f;
    float3 predicted = pos + v * p.dt * 3.0f;
    float3 push = max(wallMin - predicted, 0.0f) - max(predicted - wallMax, 0.0f);
    v += p.wallStiffness * push;

    float speed = length(v);
    if (speed > p.maxSpeed) {
        v *= p.maxSpeed / speed;
        speed = p.maxSpeed;
    }

    // Whitewater: fast flow near the free surface (below rest density), wall impacts, and
    // isolated spray droplets generate foam that then fades over about a second.
    float relDensity = densities[id] / p.restDensity;
    float surface = smoothstep(0.97f, 0.7f, relDensity);
    float spray = smoothstep(0.45f, 0.2f, relDensity);
    float fast = smoothstep(p.foamSpeed, p.foamSpeed * 2.0f, speed);
    float hit = saturate(length(push) * 1.5f) * smoothstep(p.foamSpeed * 0.5f, p.foamSpeed * 1.5f, speed);
    float generated = saturate(fast * (0.05f + surface) + spray * fast + hit);
    // Generation per step scales with (1 - decay) so foam builds at the same real-time rate.
    foam[id] = saturate(foam[id] * p.foamDecay + generated * (1.0f - p.foamDecay) * 40.0f);

    if (p.countCells != 0) {
        rank[id] = atomic_fetch_add_explicit(&cellCounts[cellOf(pos, p.gridSize)], 1u,
                                             memory_order_relaxed);
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
    if (p.sumPositions != 0) {
        float3 sum = float3(simd_sum(pos.x), simd_sum(pos.y), simd_sum(pos.z));
        if (simd_is_first()) {
            atomic_fetch_add_explicit(&stats[4], sum.x, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[5], sum.y, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[6], sum.z, memory_order_relaxed);
        }
    }
}

// MARK: - Atomic-free gather path
//
// Particles are binned by base cell (floor(position)) with a counting sort every substep. Each grid
// node then gathers from the particles in the 27 cells whose stencils reach it, so P2G needs no
// float atomics (heavily contended on tiled GPUs when neighbouring threads share nodes).

/// Quadratic B-spline as a function of node–particle distance (cells).
inline float bspline(float r) {
    r = abs(r);
    if (r < 0.5f) return 0.75f - r * r;
    if (r < 1.5f) return 0.5f * (1.5f - r) * (1.5f - r);
    return 0.0f;
}

inline float bweight(float d, int g) {
    if (g == 0) return 0.5f * (0.5f - d) * (0.5f - d);
    if (g == 1) return 0.75f - d * d;
    return 0.5f * (0.5f + d) * (0.5f + d);
}


kernel void clearCounts(device uint *counts [[buffer(0)]],
                        constant SimParams &p [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= p.gridSize.x * p.gridSize.y * p.gridSize.z) return;
    counts[id] = 0;
}

kernel void countParticles(device const Particle *particles [[buffer(0)]],
                           device atomic_uint *counts [[buffer(1)]],
                           device uint *rank [[buffer(2)]],
                           constant SimParams &p [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= p.particleCount) return;
    uint c = cellOf(particles[id].position, p.gridSize);
    rank[id] = atomic_fetch_add_explicit(&counts[c], 1u, memory_order_relaxed);
}

/// Exclusive prefix sum of the cell counts. Dispatched as a single threadgroup (up to 1024
/// threads, a multiple of 32), each thread scanning a contiguous chunk.
kernel void scanCounts(device uint *counts [[buffer(0)]],
                       device uint *starts [[buffer(1)]],
                       constant SimParams &p [[buffer(2)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint threads [[threads_per_threadgroup]],
                       uint simdLane [[thread_index_in_simdgroup]],
                       uint simdId [[simdgroup_index_in_threadgroup]]) {
    uint n = p.gridSize.x * p.gridSize.y * p.gridSize.z;
    uint chunk = (n + threads - 1) / threads;
    uint begin = min(tid * chunk, n);
    uint end = min(begin + chunk, n);
    uint local = 0;
    for (uint i = begin; i < end; i++) local += counts[i];

    threadgroup uint simdTotals[32];
    if (tid < 32) simdTotals[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint inclusive = simd_prefix_inclusive_sum(local);
    if (simdLane == 31) simdTotals[simdId] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdId == 0) {
        uint t = simdTotals[simdLane];
        simdTotals[simdLane] = simd_prefix_exclusive_sum(t);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint offset = simdTotals[simdId] + inclusive - local;
    for (uint i = begin; i < end; i++) {
        starts[i] = offset;
        offset += counts[i];
        counts[i] = 0;  // ready for the next count in g2p
    }
    if (end == n && begin < end) starts[n] = offset;
    if (n == 0 && tid == 0) starts[0] = 0;
}

/// Physically reorders particles (and their foam) by cell so the gathers read contiguous memory.
kernel void reorderParticles(device const Particle *src [[buffer(0)]],
                             device const uint *rank [[buffer(1)]],
                             device const uint *starts [[buffer(2)]],
                             device Particle *dst [[buffer(3)]],
                             constant SimParams &p [[buffer(4)]],
                             device const float *srcFoam [[buffer(5)]],
                             device float *dstFoam [[buffer(6)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= p.particleCount) return;
    Particle pt = src[id];
    uint slot = starts[cellOf(pt.position, p.gridSize)] + rank[id];
    dst[slot] = pt;
    dstFoam[slot] = srcFoam[id];
}

kernel void p2gGatherMass(device const Particle *particles [[buffer(0)]],
                          device const uint *starts [[buffer(1)]],
                          device float4 *grid [[buffer(4)]],
                          constant SimParams &p [[buffer(5)]],
                          uint id [[thread_position_in_grid]]) {
    uint3 size = p.gridSize;
    if (id >= size.x * size.y * size.z) return;
    int3 n = int3(id % size.x, (id / size.x) % size.y, id / (size.x * size.y));
    float3 nodePos = float3(n) + 0.5f;
    float4 acc = float4(0.0f);
    // Cells along x are contiguous after the reorder, so each (y, z) row of three cells is one
    // contiguous run of particles.
    int x0 = max(n.x - 1, 0);
    int x1 = min(n.x + 1, int(size.x) - 1);
    for (int dz = -1; dz <= 1; dz++) {
        int bz = n.z + dz;
        if (bz < 0 || bz >= int(size.z)) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int by = n.y + dy;
            if (by < 0 || by >= int(size.y)) continue;
            uint s = starts[gridIndex(int3(x0, by, bz), size)];
            uint e = starts[gridIndex(int3(x1, by, bz), size) + 1];
            for (uint k = s; k < e; k++) {
                Particle pt = particles[k];
                float3 dist = nodePos - pt.position;
                float w = bspline(dist.x) * bspline(dist.y) * bspline(dist.z);
                float3x3 C = float3x3(pt.c0, pt.c1, pt.c2);
                acc += float4(w, w * (pt.velocity + C * dist));
            }
        }
    }
    grid[id] = acc;
}

kernel void particleStress(device const Particle *particles [[buffer(0)]],
                           device const float4 *grid [[buffer(1)]],
                           device float3x3 *terms [[buffer(2)]],
                           constant SimParams &p [[buffer(3)]],
                           device float *densities [[buffer(4)]],
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
                density += weight * grid[gridIndex(cell, p.gridSize)].x;
            }
        }
    }
    density = max(density, 1e-4f);
    densities[id] = density;
    float volume = 1.0f / density;
    float pressure = p.stiffness * (pow(density / p.restDensity, p.eosPower) - 1.0f);
    pressure = max(pressure, -p.cohesion * p.stiffness);
    float3x3 C = float3x3(pt.c0, pt.c1, pt.c2);
    float3x3 stress = p.viscosity * (C + transpose(C));
    stress[0][0] -= pressure;
    stress[1][1] -= pressure;
    stress[2][2] -= pressure;
    terms[id] = (-volume * kAPIC * p.dt) * stress;
}

/// Adds the stress momentum to each node and applies the grid update in the same pass.
kernel void p2gGatherStress(device const Particle *particles [[buffer(0)]],
                            device const uint *starts [[buffer(1)]],
                            device float4 *gridVel [[buffer(2)]],
                              device const float4 *grid [[buffer(4)]],
                            constant SimParams &p [[buffer(5)]],
                            device const float3x3 *terms [[buffer(6)]],
                            uint id [[thread_position_in_grid]]) {
    uint3 size = p.gridSize;
    if (id >= size.x * size.y * size.z) return;
    float4 node = grid[id];
    if (node.x <= 0.0f) {
        gridVel[id] = float4(0.0f);
        return;
    }
    int3 n = int3(id % size.x, (id / size.x) % size.y, id / (size.x * size.y));
    float3 nodePos = float3(n) + 0.5f;
    float3 acc = float3(0.0f);
    int x0 = max(n.x - 1, 0);
    int x1 = min(n.x + 1, int(size.x) - 1);
    for (int dz = -1; dz <= 1; dz++) {
        int bz = n.z + dz;
        if (bz < 0 || bz >= int(size.z)) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int by = n.y + dy;
            if (by < 0 || by >= int(size.y)) continue;
            uint s = starts[gridIndex(int3(x0, by, bz), size)];
            uint e = starts[gridIndex(int3(x1, by, bz), size) + 1];
            for (uint k = s; k < e; k++) {
                float3 dist = nodePos - particles[k].position;
                float w = bspline(dist.x) * bspline(dist.y) * bspline(dist.z);
                acc += w * (terms[k] * dist);
            }
        }
    }
    node.yzw += acc;
    gridVel[id] = float4(gridNodeVelocity(node, id, p), node.x);
}
