#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

// Screen-space fluid rendering: particle depth + thickness sprites, bilateral depth smoothing,
// and a composite that refracts the tank interior through the reconstructed surface.

// MARK: - Shared helpers

inline float hash13(float3 p) {
    p = fract(p * 0.1031f);
    p += dot(p, p.zyx + 31.32f);
    return fract((p.x + p.y) * p.z);
}

inline float valueNoise(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    float n000 = hash13(i);
    float n100 = hash13(i + float3(1, 0, 0));
    float n010 = hash13(i + float3(0, 1, 0));
    float n110 = hash13(i + float3(1, 1, 0));
    float n001 = hash13(i + float3(0, 0, 1));
    float n101 = hash13(i + float3(1, 0, 1));
    float n011 = hash13(i + float3(0, 1, 1));
    float n111 = hash13(i + float3(1, 1, 1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

/// Interference of drifting sine sheets; bright filaments like light focused by ripples.
inline float causticPattern(float2 p, float t) {
    float v = 0.0f;
    float2 q = p;
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        q += 0.55f * float2(sin(q.y * 1.7f + t * 0.9f + fi * 1.3f),
                            cos(q.x * 1.4f - t * 0.75f + fi * 2.1f));
        v += 1.0f - abs(sin(q.x * 0.9f + q.y * 0.6f + fi));
    }
    v /= 3.0f;
    return pow(v, 7.0f);
}

inline float3 environment(float3 dir, constant RenderUniforms &u) {
    float up = dir.y * 0.5f + 0.5f;
    float3 amb = u.ambientColor.rgb;
    float3 sky = mix(amb * 0.35f, amb * 1.3f + 0.05f, smoothstep(0.0f, 1.0f, up));
    // A soft rectangular "window" in the light direction gives glossy liquids something to show.
    float3 L = normalize(u.lightDirWorld.xyz);
    float d = max(dot(dir, L), 0.0f);
    float sun = pow(d, 220.0f) * 8.0f + pow(d, 18.0f) * 0.6f;
    float band = smoothstep(0.75f, 0.95f, dir.y) * 0.25f;
    return sky + u.lightColor.rgb * (sun + band);
}

/// Walls: 0 back, 1 left, 2 right, 3 bottom, 4 top.
inline float3 wallNormal(int wall) {
    switch (wall) {
        case 0: return float3(0, 0, 1);
        case 1: return float3(1, 0, 0);
        case 2: return float3(-1, 0, 0);
        case 3: return float3(0, 1, 0);
        default: return float3(0, -1, 0);
    }
}

inline float2 wallCoords(float3 p, int wall) {
    switch (wall) {
        case 0: return p.xy;
        case 1:
        case 2: return float2(p.z, p.y);
        default: return float2(p.x, p.z);
    }
}

inline float3 backdropAlbedo(float3 p, int wall, constant RenderUniforms &u,
                             texture2d<float> photo, sampler s) {
    float2 c = wallCoords(p, wall);
    float3 bmin = u.boxMin.xyz;
    float3 bmax = u.boxMax.xyz;
    switch (u.backdrop) {
        case 1: {  // tinted glass tank
            float g = 0.5f + 0.5f * (p.y - bmin.y) / (bmax.y - bmin.y);
            float3 base = mix(float3(0.62f, 0.78f, 0.80f), float3(0.86f, 0.94f, 0.95f), g);
            float frost = valueNoise(float3(c * 0.9f, 1.7f)) * 0.06f;
            return base - frost;
        }
        case 2: {  // user photo on the back wall, edge pixels stretched along the sides
            float2 uv = float2((p.x - bmin.x) / (bmax.x - bmin.x), 1.0f - (p.y - bmin.y) / (bmax.y - bmin.y));
            float3 col = photo.sample(s, clamp(uv, 0.001f, 0.999f)).rgb;
            return wall == 0 ? col : col * 0.7f;
        }
        case 3: {  // dark void
            float g = (p.y - bmin.y) / (bmax.y - bmin.y);
            return mix(float3(0.01f, 0.012f, 0.02f), float3(0.035f, 0.04f, 0.06f), g);
        }
        default: {  // transparency checkerboard
            float2 cell = floor(c / 5.0f);
            float parity = fmod(abs(cell.x + cell.y), 2.0f);
            return parity < 0.5f ? float3(0.86f) : float3(0.62f);
        }
    }
}

inline float3 shadeWall(float3 p, int wall, constant RenderUniforms &u,
                        texture2d<float> photo, sampler s) {
    float3 albedo = backdropAlbedo(p, wall, u, photo, s);
    float3 n = wallNormal(wall);
    float3 L = normalize(u.lightDirWorld.xyz);
    float diffuse = max(dot(n, L), 0.0f) * 0.55f + 0.45f;

    // Contact darkening where walls meet; the open glass side (z = 0) has no wall.
    float3 bmin = u.boxMin.xyz;
    float3 bmax = u.boxMax.xyz;
    float d = 1e6f;
    if (wall != 1) d = min(d, p.x - bmin.x);
    if (wall != 2) d = min(d, bmax.x - p.x);
    if (wall != 3) d = min(d, p.y - bmin.y);
    if (wall != 4) d = min(d, bmax.y - p.y);
    if (wall != 0) d = min(d, p.z - bmin.z);
    float ao = mix(0.55f, 1.0f, smoothstep(0.0f, 9.0f, d));
    // Side walls recede into shadow toward the back.
    float recess = wall == 0 ? 0.88f : mix(0.7f, 1.0f, saturate((p.z - bmin.z) / (bmax.z - bmin.z)));

    float3 light = u.ambientColor.rgb * 0.55f + u.lightColor.rgb * diffuse * 0.65f;
    return albedo * light * ao * recess;
}

/// Ray from inside the tank to the first wall hit. Returns t; `wall` receives the wall id
/// (-1 when the ray exits through the glass).
inline float traceBox(float3 o, float3 dir, constant RenderUniforms &u, thread int &wall) {
    float3 bmin = u.boxMin.xyz;
    float3 bmax = u.boxMax.xyz;
    float3 inv = 1.0f / select(dir, float3(1e-5f), abs(dir) < 1e-5f);
    float3 tMax = (select(bmin, bmax, dir > 0.0f) - o) * inv;
    float t = 1e9f;
    wall = -1;
    if (tMax.x > 0.0f && tMax.x < t) { t = tMax.x; wall = dir.x > 0.0f ? 2 : 1; }
    if (tMax.y > 0.0f && tMax.y < t) { t = tMax.y; wall = dir.y > 0.0f ? 4 : 3; }
    if (tMax.z > 0.0f && tMax.z < t) { t = tMax.z; wall = dir.z > 0.0f ? -1 : 0; }
    return t;
}

// MARK: - Tank interior

struct BoxOut {
    float4 position [[position]];
    float3 world;
    int wall [[flat]];
};

vertex BoxOut boxVertex(uint vid [[vertex_id]], constant RenderUniforms &u [[buffer(0)]]) {
    float3 a = u.boxMin.xyz;
    float3 b = u.boxMax.xyz;
    float3 corners[5][4] = {
        {float3(a.x, a.y, a.z), float3(b.x, a.y, a.z), float3(a.x, b.y, a.z), float3(b.x, b.y, a.z)},
        {float3(a.x, a.y, b.z), float3(a.x, a.y, a.z), float3(a.x, b.y, b.z), float3(a.x, b.y, a.z)},
        {float3(b.x, a.y, a.z), float3(b.x, a.y, b.z), float3(b.x, b.y, a.z), float3(b.x, b.y, b.z)},
        {float3(a.x, a.y, b.z), float3(b.x, a.y, b.z), float3(a.x, a.y, a.z), float3(b.x, a.y, a.z)},
        {float3(a.x, b.y, a.z), float3(b.x, b.y, a.z), float3(a.x, b.y, b.z), float3(b.x, b.y, b.z)},
    };
    const uint order[6] = {0, 1, 2, 2, 1, 3};
    uint wall = vid / 6;
    float3 p = corners[wall][order[vid % 6]];
    BoxOut out;
    out.world = p;
    out.wall = int(wall);
    out.position = u.projectionMatrix * (u.viewMatrix * float4(p, 1.0f));
    return out;
}

fragment float4 boxFragment(BoxOut in [[stage_in]], constant RenderUniforms &u [[buffer(0)]],
                            texture2d<float> photo [[texture(0)]], sampler s [[sampler(0)]]) {
    return float4(shadeWall(in.world, in.wall, u, photo, s), 1.0f);
}

// MARK: - Particle sprites

struct SpriteOut {
    float4 position [[position]];
    float2 uv;
    float3 centerView;
    float radius;
    float speed;
};

vertex SpriteOut particleSprite(uint vid [[vertex_id]], uint iid [[instance_id]],
                                device const Particle *particles [[buffer(0)]],
                                constant RenderUniforms &u [[buffer(1)]],
                                constant float &radiusScale [[buffer(2)]]) {
    const float2 corners[4] = {float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)};
    float2 c = corners[vid];
    Particle p = particles[iid];
    float3 world = u.simOrigin.xyz + p.position * u.cellMM;
    float3 centerView = (u.viewMatrix * float4(world, 1.0f)).xyz;
    float r = u.particleRadiusMM * radiusScale;
    SpriteOut out;
    out.position = u.projectionMatrix * float4(centerView + float3(c * r, 0.0f), 1.0f);
    out.uv = c;
    out.centerView = centerView;
    out.radius = r;
    out.speed = length(p.velocity);
    return out;
}

struct DepthOut {
    float linearDepth [[color(0)]];
    float depth [[depth(less)]];
};

fragment DepthOut particleDepth(SpriteOut in [[stage_in]], constant RenderUniforms &u [[buffer(0)]]) {
    float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0f) discard_fragment();
    float3 vp = in.centerView + float3(in.uv * in.radius, sqrt(1.0f - r2) * in.radius);
    float4 clip = u.projectionMatrix * float4(vp, 1.0f);
    DepthOut out;
    out.linearDepth = -vp.z;
    out.depth = clip.z / clip.w;
    return out;
}

fragment float4 particleThickness(SpriteOut in [[stage_in]], constant RenderUniforms &u [[buffer(0)]]) {
    float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0f) discard_fragment();
    float falloff = exp(-r2 * 2.5f);
    // Normalised so a full column of resting liquid sums to roughly its depth in mm.
    float thickness = in.radius * 0.105f * falloff;
    float foam = smoothstep(u.foamSpeed, u.foamSpeed * 1.9f, in.speed) * falloff;
    return float4(thickness, foam, 0.0f, 0.0f);
}

// MARK: - Depth smoothing

struct SmoothParams {
    int2 direction;
    int radius;
    float depthFalloff;  // mm
};

kernel void smoothDepth(texture2d<float, access::read> src [[texture(0)]],
                        texture2d<float, access::write> dst [[texture(1)]],
                        constant SmoothParams &sp [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint2 size = uint2(src.get_width(), src.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;
    float d0 = src.read(gid).r;
    if (d0 <= 0.0f) {
        dst.write(float4(0.0f), gid);
        return;
    }
    float sigma = float(sp.radius) * 0.5f;
    float sum = 0.0f;
    float wsum = 0.0f;
    for (int i = -sp.radius; i <= sp.radius; i++) {
        int2 q = int2(gid) + sp.direction * i;
        if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y)) continue;
        float d = src.read(uint2(q)).r;
        if (d <= 0.0f) continue;
        // Narrow-range clamp: pull far samples toward the centre instead of dropping them.
        d = clamp(d, d0 - sp.depthFalloff, d0 + sp.depthFalloff);
        float w = exp(-float(i * i) / (2.0f * sigma * sigma));
        float dz = (d - d0) / sp.depthFalloff;
        w *= exp(-dz * dz);
        sum += d * w;
        wsum += w;
    }
    dst.write(float4(sum / max(wsum, 1e-5f)), gid);
}

// MARK: - Composite

struct FullscreenOut {
    float4 position [[position]];
    float2 uv;
};

vertex FullscreenOut fullscreenVertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    FullscreenOut out;
    out.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    out.uv = float2(p.x, 1.0f - p.y);
    return out;
}

inline float3 viewFromDepth(float2 uv, float depth, constant RenderUniforms &u) {
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float4x4 P = u.projectionMatrix;
    float x = depth * (ndc.x + P[2][0]) / P[0][0];
    float y = depth * (ndc.y + P[2][1]) / P[1][1];
    return float3(x, y, -depth);
}

fragment float4 compositeFragment(FullscreenOut in [[stage_in]],
                                  constant RenderUniforms &u [[buffer(0)]],
                                  texture2d<float> scene [[texture(0)]],
                                  texture2d<float> depthTex [[texture(1)]],
                                  texture2d<float> thickTex [[texture(2)]],
                                  texture2d<float> photo [[texture(3)]],
                                  sampler linearSampler [[sampler(0)]],
                                  sampler depthSampler [[sampler(1)]]) {
    float2 uv = in.uv;
    float3 sceneColor = scene.sample(linearSampler, uv).rgb;
    float2 thick = thickTex.sample(linearSampler, uv).rg;
    float depth = depthTex.sample(depthSampler, uv).r;

    // Front-glass sheen that slides with the viewpoint.
    float sheenCoord = uv.x * 0.7f + uv.y * 0.45f - u.eyeWorld.x * 0.004f + u.eyeWorld.y * 0.002f;
    float sheen = exp(-pow((fract(sheenCoord * 0.8f) - 0.5f) * 7.0f, 2.0f)) * 0.035f;
    float3 glass = u.lightColor.rgb * sheen;

    if (depth <= 0.0f || thick.r < 0.02f) {
        return float4(sceneColor + glass, 1.0f);
    }

    float2 texel = 1.0f / float2(depthTex.get_width(), depthTex.get_height());
    float3 P = viewFromDepth(uv, depth, u);
    float dR = depthTex.sample(depthSampler, uv + float2(texel.x, 0)).r;
    float dL = depthTex.sample(depthSampler, uv - float2(texel.x, 0)).r;
    float dU = depthTex.sample(depthSampler, uv - float2(0, texel.y)).r;
    float dD = depthTex.sample(depthSampler, uv + float2(0, texel.y)).r;
    float3 ddx = viewFromDepth(uv + float2(texel.x, 0), dR > 0 ? dR : depth, u) - P;
    float3 ddx2 = P - viewFromDepth(uv - float2(texel.x, 0), dL > 0 ? dL : depth, u);
    if (abs(ddx2.z) < abs(ddx.z)) ddx = ddx2;
    float3 ddy = viewFromDepth(uv - float2(0, texel.y), dU > 0 ? dU : depth, u) - P;
    float3 ddy2 = P - viewFromDepth(uv + float2(0, texel.y), dD > 0 ? dD : depth, u);
    if (abs(ddy2.z) < abs(ddy.z)) ddy = ddy2;
    float3 N = normalize(cross(ddx, ddy));

    // View space and world space share orientation (the view matrix is a translation).
    float3 worldPos = P + u.eyeWorld.xyz;
    float3 V = normalize(-P);
    float3 L = normalize(u.lightDirWorld.xyz);
    float ndv = saturate(dot(N, V));
    float F0 = u.material.x;
    float fresnel = F0 + (1.0f - F0) * pow(1.0f - ndv, 5.0f);

    float thickness = thick.r;
    float foam = saturate(thick.g * u.material.w * 1.4f);

    // Transmission: refract into the liquid and shade the wall the ray lands on.
    float3 refr = refract(-V, N, 1.0f / 1.33f);
    float bend = u.absorption.w;
    float3 tdir = normalize(mix(-V, refr, bend));
    int wall;
    float t = traceBox(worldPos, tdir, u, wall);
    float3 behind;
    if (wall >= 0) {
        float3 hit = worldPos + tdir * t;
        behind = shadeWall(hit, wall, u, photo, linearSampler);
        float2 wc = wallCoords(hit, wall) * 0.32f + N.xy * 2.0f;
        float c = causticPattern(wc, u.time * 1.4f) * u.causticStrength;
        behind += behind * c * u.lightColor.rgb * 1.6f;
    } else {
        behind = sceneColor;
    }
    float3 absorb = exp(-u.absorption.rgb * thickness);
    float3 transmitted = behind * absorb * u.tint.rgb;
    // In-scattering: thick liquid glows in the colour it absorbs least.
    float3 scatterTint = exp(-u.absorption.rgb * 12.0f);
    float scatterAmount = (1.0f - exp(-0.12f * thickness)) * (1.0f - u.tint.w);
    transmitted += scatterTint * scatterAmount * (u.ambientColor.rgb * 0.22f + u.lightColor.rgb * 0.18f);
    // Opaque liquids (milk, lava) show a diffuse body instead of what is behind them.
    float diffuse = saturate(dot(N, L)) * 0.6f + 0.4f;
    float3 body = u.tint.rgb * (u.ambientColor.rgb * 0.6f + u.lightColor.rgb * diffuse * 0.7f);
    transmitted = mix(transmitted, body, u.tint.w);

    float3 R = reflect(-V, N);
    float3 reflected = environment(R, u);
    float3 color;
    if (u.material.y > 0.5f) {
        // Metallic: reflections tinted by the metal, plus a hint of the tank interior.
        int rwall;
        float rt = traceBox(worldPos, R, u, rwall);
        float3 interior = rwall >= 0 ? shadeWall(worldPos + R * rt, rwall, u, photo, linearSampler) : reflected;
        float3 env = mix(interior, reflected, smoothstep(-0.2f, 0.6f, R.z));
        color = env * u.tint.rgb * mix(u.material.x, 1.0f, pow(1.0f - ndv, 3.0f));
    } else {
        color = mix(transmitted, reflected, fresnel);
    }

    float3 H = normalize(L + V);
    float shininess = mix(900.0f, 40.0f, u.material.z);
    float spec = pow(saturate(dot(N, H)), shininess) * mix(4.0f, 1.0f, u.material.z);
    color += u.lightColor.rgb * spec;

    // Up-facing free surface catches the light from above, like the bright band where a wave
    // meets the air in a real tank.
    float surfaceBand = smoothstep(0.55f, 0.95f, N.y) * (1.0f - u.material.y);
    color += (u.lightColor.rgb * 0.3f + u.ambientColor.rgb * 0.15f) * surfaceBand * mix(0.5f, 1.0f, u.material.w);

    // Lava: glowing core under a cooling crust.
    if (u.emission.w > 0.0f) {
        float n = valueNoise(worldPos * 0.35f + float3(0, 0, u.time * 0.35f));
        float n2 = valueNoise(worldPos * 1.1f - float3(u.time * 0.2f));
        float crust = smoothstep(0.45f, 0.7f, n * 0.7f + n2 * 0.3f);
        float core = saturate(thickness * 0.15f);
        color += u.emission.rgb * u.emission.w * (1.0f - crust * 0.85f) * (0.35f + core);
    }

    float3 foamColor = (u.ambientColor.rgb * 0.7f + u.lightColor.rgb * 0.8f) * 0.95f;
    color = mix(color, foamColor, foam);

    float alpha = smoothstep(0.02f, 0.6f, thickness);
    alpha = max(alpha, foam);
    return float4(mix(sceneColor, color, alpha) + glass, 1.0f);
}
