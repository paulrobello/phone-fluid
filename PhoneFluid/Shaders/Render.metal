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
        case 2: {  // user photo on the back wall (side walls are glass, see shadeWall)
            float2 uv = float2((p.x - bmin.x) / (bmax.x - bmin.x), 1.0f - (p.y - bmin.y) / (bmax.y - bmin.y));
            return photo.sample(s, clamp(uv, 0.001f, 0.999f)).rgb;
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

/// The room seen through a glass tank: an out-of-focus interior (floor, wall, ceiling bands with
/// soft warm light blobs). It shifts with the viewing direction, so it parallaxes behind the tank.
inline float3 roomBehindGlass(float3 dir, constant RenderUniforms &u) {
    float up = dir.y;
    float3 floorColor = float3(0.16f, 0.12f, 0.09f);
    float3 wallColor = float3(0.34f, 0.31f, 0.27f);
    float3 ceilingColor = float3(0.62f, 0.64f, 0.66f);
    float3 room = mix(floorColor, wallColor, smoothstep(-0.55f, 0.0f, up));
    room = mix(room, ceilingColor, smoothstep(0.15f, 0.75f, up));
    // Defocused lamps and windows: soft blobs at a few scales.
    float2 q = dir.xy / max(abs(dir.z), 0.25f) * 14.0f;
    float warm = smoothstep(0.6f, 0.92f, valueNoise(float3(q, 3.1f)));
    float warmSmall = smoothstep(0.72f, 0.95f, valueNoise(float3(q * 2.3f, 8.7f)));
    float cool = smoothstep(0.66f, 0.95f, valueNoise(float3(q * 0.7f + 7.0f, 1.3f)));
    room += float3(1.0f, 0.78f, 0.5f) * (warm * 0.45f + warmSmall * 0.35f);
    room += float3(0.55f, 0.75f, 1.0f) * cool * 0.3f;
    return room * (u.ambientColor.rgb * 0.8f + 0.2f);
}

/// Clear glass walls: see-through tint that deepens at grazing angles, Fresnel reflection, bright
/// edges where the thick panes meet, and faint streaks that slide with the viewpoint.
inline float3 shadeGlassWall(float3 p, int wall, constant RenderUniforms &u) {
    float3 dir = normalize(p - u.eyeWorld.xyz);
    float3 n = wallNormal(wall);
    float cosT = abs(dot(dir, n));
    float3 tint = mix(float3(0.92f, 0.99f, 0.97f), float3(0.55f, 0.80f, 0.74f), (1.0f - cosT) * 0.7f);
    float3 color = roomBehindGlass(dir, u) * tint;

    float fresnel = 0.04f + 0.96f * pow(1.0f - cosT, 5.0f);
    color = mix(color, environment(reflect(dir, n), u) * 0.85f, fresnel * 0.55f);

    float3 bmin = u.boxMin.xyz;
    float3 bmax = u.boxMax.xyz;
    float d = 1e6f;
    if (wall != 1) d = min(d, p.x - bmin.x);
    if (wall != 2) d = min(d, bmax.x - p.x);
    if (wall != 3) d = min(d, p.y - bmin.y);
    if (wall != 4) d = min(d, bmax.y - p.y);
    if (wall != 0) d = min(d, p.z - bmin.z);
    float edge = 1.0f - smoothstep(0.0f, 1.8f, d);
    color += float3(0.45f, 0.85f, 0.75f) * edge * (u.lightColor.rgb * 0.7f + 0.15f);

    float2 c = wallCoords(p, wall);
    float streak = sin(c.x * 0.09f + c.y * 0.05f - u.eyeWorld.x * 0.01f) * 0.5f + 0.5f;
    color += u.lightColor.rgb * pow(streak, 40.0f) * 0.06f;
    return color;
}

inline float3 shadeWall(float3 p, int wall, constant RenderUniforms &u,
                        texture2d<float> photo, sampler s) {
    // Glass tank, and the side walls of the photo backdrop (stretching the photo's edge pixels
    // along the depth streaked them up the walls).
    if (u.backdrop == 1 || (u.backdrop == 2 && wall != 0)) return shadeGlassWall(p, wall, u);
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
    float foam;
    float thicknessScale;
};

vertex SpriteOut particleSprite(uint vid [[vertex_id]], uint iid [[instance_id]],
                                device const Particle *particles [[buffer(0)]],
                                constant RenderUniforms &u [[buffer(1)]],
                                constant float &radiusScale [[buffer(2)]],
                                device const float *foam [[buffer(3)]],
                                device const float *density [[buffer(4)]]) {
    const float2 corners[4] = {float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)};
    float2 c = corners[vid];
    Particle p = particles[iid];
    float3 world = u.simOrigin.xyz + p.position * u.cellMM;
    float3 centerView = (u.viewMatrix * float4(world, 1.0f)).xyz;
    // Sparse particles (thin films, drips) grow so neighbours overlap into a continuous sheet
    // instead of separate beads; the bulk (at rest density) keeps its size.
    float sparse = clamp(pow(u.restDensity / max(density[iid], 0.05f), 1.0f / 3.0f), 1.0f, 2.0f);
    float r = u.particleRadiusMM * radiusScale * sparse;
    SpriteOut out;
    out.position = u.projectionMatrix * float4(centerView + float3(c * r, 0.0f), 1.0f);
    out.uv = c;
    out.centerView = centerView;
    out.radius = r;
    out.foam = foam[iid];
    out.thicknessScale = 1.0f / sparse;
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
    float thickness = in.radius * 0.105f * falloff * in.thicknessScale;
    // b accumulates weight so the composite can use the foam *fraction* of the column.
    return float4(thickness, in.foam * falloff, falloff, 0.0f);
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
        // Bridge gaps between sparse particles (thin films): fill a hole only when liquid lies
        // on both sides within the kernel, so silhouettes don't grow outward.
        int reach = max(sp.radius / 2, 2);
        float before = 0.0f, after = 0.0f;
        for (int i = 1; i <= reach; i++) {
            int2 a = int2(gid) - sp.direction * i;
            int2 b = int2(gid) + sp.direction * i;
            if (before <= 0.0f && all(a >= 0) && all(a < int2(size))) before = src.read(uint2(a)).r;
            if (after <= 0.0f && all(b >= 0) && all(b < int2(size))) after = src.read(uint2(b)).r;
        }
        dst.write(float4(before > 0.0f && after > 0.0f ? 0.5f * (before + after) : 0.0f), gid);
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

/// Separable Gaussian on the thickness/foam buffer so sparse liquid reads as a continuous sheet.
kernel void blurThickness(texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]],
                          constant SmoothParams &sp [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint2 size = uint2(src.get_width(), src.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;
    float sigma = max(float(sp.radius) * 0.5f, 1.0f);
    float4 sum = float4(0.0f);
    float wsum = 0.0f;
    for (int i = -sp.radius; i <= sp.radius; i++) {
        int2 q = clamp(int2(gid) + sp.direction * i, int2(0), int2(size) - 1);
        float w = exp(-float(i * i) / (2.0f * sigma * sigma));
        sum += src.read(uint2(q)) * w;
        wsum += w;
    }
    dst.write(sum / wsum, gid);
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
    float3 thick = thickTex.sample(linearSampler, uv).rgb;
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
    float foamFraction = thick.g / max(thick.b, 1e-3f);
    float foam = smoothstep(0.08f, 0.45f, foamFraction) * saturate(u.material.w);

    // Transmission: refract into the liquid and shade the wall the ray lands on.
    // The front of the liquid lies flat against the glass, which barely bends the view; drifting
    // ripples in the refraction normal make the background visibly swim through the water.
    float rippleTime = u.ripplePhase * 0.9f;
    float3 rp = worldPos * 0.22f;
    float2 ripple = float2(valueNoise(rp + float3(0.0f, rippleTime, 0.0f)),
                           valueNoise(rp + float3(5.2f, -rippleTime * 0.8f, 1.3f))) - 0.5f;
    // Still liquid keeps a faint static wobble; motion brings the full drifting ripple.
    float3 Nr = normalize(N + float3(ripple * (0.12f + 0.43f * u.activity), 0.0f));
    float3 refr = refract(-V, Nr, 1.0f / 1.45f);
    float bend = u.absorption.w;
    float3 tdir = normalize(mix(-V, refr, bend));
    int wall;
    float t = traceBox(worldPos, tdir, u, wall);
    float3 behind;
    if (wall >= 0) {
        float3 hit = worldPos + tdir * t;
        behind = shadeWall(hit, wall, u, photo, linearSampler);
        float2 wc = wallCoords(hit, wall) * 0.32f + N.xy * 2.0f;
        float c = causticPattern(wc, u.ripplePhase * 1.4f) * u.causticStrength;
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
    float lambert = saturate(dot(N, L)) * 0.6f + 0.4f;
    // Creamy liquids scatter light internally: wrap lighting hides the particle-scale bumps.
    float wrapped = saturate((dot(N, L) + 1.0f) / 2.0f) * 0.5f + 0.45f;
    float diffuse = mix(lambert, wrapped, u.opaqueSoftness);
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

    // Break whitewater into bubbly clumps instead of a flat white wash.
    float bubbles = valueNoise(worldPos * 1.3f + float3(0.0f, u.time * 0.6f, 0.0f));
    bubbles = 0.45f + 0.55f * smoothstep(0.3f, 0.75f, bubbles);
    foam *= bubbles;
    float3 foamColor = (u.ambientColor.rgb * 0.7f + u.lightColor.rgb * 0.8f) * 0.95f;
    color = mix(color, foamColor, foam);

    float alpha = smoothstep(0.02f, 0.6f, thickness);
    alpha = max(alpha, foam);
    return float4(mix(sceneColor, color, alpha) + glass, 1.0f);
}
