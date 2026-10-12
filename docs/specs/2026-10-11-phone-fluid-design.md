# PhoneFluid — Design Spec

Date: 2026-10-11 · Status: approved (user waived section-by-section review)

## Intent

A personal toy for Paul's iPhone 18 Pro Max / 17 Pro Max that makes the phone look like a
glass tank holding a liquid. Visual wow beats polish; no App Store requirements.

What Paul asked for:
- Realistic 3D fluid simulation tied to the accelerometer.
- Selectable fluids that differ in **look and physics** (water, honey, oil, mercury, lava, slime).
- Strong perspective shift so the screen reads as a window into a deep box (reference photos:
  checkerboard back wall + recessed side walls, sloshing water with foam/spray).
- Perspective source: device tilt by default, front-camera face tracking as a toggle.
- Scene lighting follows the real environment.
- Extras: touch stirring/splashes, shake splash, haptics, procedural sound, adjustable fill level.
- Backdrops: checkerboard (default), glass tank, user photo, dark void.

Assumptions (not stated by Paul):
- iOS 26+, iPhone only, **portrait locked** (rotation would flip the gravity mapping).
- 120 Hz target via ProMotion; simulation quality is a setting.

## Success criteria

1. Tilting the phone makes the fluid flow downhill in real time with overturning waves and spray.
2. The tank interior shows convincing parallax (side walls visible, shifting with tilt/head).
3. Each preset is visibly distinct in motion (honey slow, mercury beads and reflects, lava glows).
4. Sustains ≥60 fps on iPhone 18 Pro Max at the default quality (measured on device).
5. Headless simulator test: N sim steps produce no NaNs, all particles inside the box, count conserved.

## Architecture

SwiftUI shell → `FluidMetalView` (`MTKView`, UIViewRepresentable) → `Renderer` (MainActor, owns
the frame loop). Per frame:

1. `InputState` snapshot: gravity vector, eye position, light estimate, touches, shake impulse.
2. `FluidSimulation.encode(commandBuffer:)` — N MLS-MPM substeps in compute.
3. `FluidRenderer.encode(...)` — backdrop/box pass, particle depth + thickness passes, depth
   smoothing, composite to the drawable.
4. Stats reduction written to a triple-buffered readback buffer; CPU reads the previous frame's
   result (never waits on the GPU) to drive haptics and audio.

### Units and frames

- Device frame = Core Motion device frame: +x right, +y toward the top of the screen, +z out of
  the screen toward the viewer. The simulation and world frames use the same axes.
- World units are millimetres. The glass is the plane z = 0, centred on the screen; the tank spans
  z ∈ [-D, 0] (D = tank depth, default 24 mm). Screen size in mm = points × 0.1657 (460 ppi @3x).
- Simulation units are grid cells; `world = simPos * cellMM - (W/2, H/2, D)`.

### Simulation — MLS-MPM (`Shaders/Simulation.metal`, `Sim/FluidSimulation.swift`)

Particle: position, velocity, affine C (3×3), all float. Grid node: mass + momentum as
`atomic<float>` (MSL 3.0 device float atomics; fallback to fixed-point `atomic_int` if the
compiler rejects it). Kernels per substep:

1. `clearGrid`
2. `p2g1` — scatter mass and APIC momentum (quadratic B-spline, 27 nodes).
3. `p2g2` — gather density → pressure via Tait EOS `k((ρ/ρ0)^γ − 1)`, clamped below at
   `-cohesion·k` (negative pressure = surface tension-ish cohesion for mercury/slime);
   viscous stress `μ(∇v + ∇vᵀ)`; scatter stress momentum.
4. `updateGrid` — velocity = momentum/mass, add gravity·dt, add touch forces, enforce walls
   (zero normal velocity into the box within a 2-cell margin; free-slip).
5. `g2p` — gather velocity and C, advect, clamp inside the box, write per-particle speed.
6. `stats` (once per frame) — wall-impact energy, kinetic energy, max speed into a small buffer.

Buffers are allocated at **max capacity**; `activeCount` drives dispatch. Fill level changes
`activeCount` (spawn new particles as a pour near the top / drop the tail).

Quality presets set cell size: Low 2.6 mm (default; the only preset that holds full simulation speed at 120 fps on iPhone 18 Pro Max), Medium 2.1 mm, High 1.7 mm. Particles are
seeded at rest density (spacing 0.63 cells, about 4 per cell). The sim uses a fixed dt of 0.2 sim units with gravity 0.04 per g, stiffness 3, Tait exponent 4 and cohesion 0.05 (stiffer settings exceed the acoustic CFL limit and make the resting surface boil); 15% extra particles offset the settled compression; each preset sets steps per second (800–1300), giving about 10 substeps per 120 Hz frame.

Gravity input = `(gravity − userAcceleration) × 9810 mm/s²` mapped to cells/s², times a
per-preset `gravityScale` and a global `timeScale` (the real 7 cm tank is fast; tuned for feel).

### Fluid presets (`Sim/FluidPreset.swift`)

Each preset = physics (`restDensity`, `stiffness`, `viscosity`, `cohesion`, `substeps`) + look
(`absorption` RGB, `tint`, `refraction`, `reflectivity`, `metallic`, `foamAmount`, `emission`,
`roughness`). Presets: Water, Ocean, Honey, Oil, Mercury, Lava, Slime, Milk.

### Rendering — screen-space fluid (`Shaders/Render.metal`, `Render/FluidRenderer.swift`)

Camera: off-axis (asymmetric) frustum from the eye position to the physical screen rectangle
(Kooima generalized perspective, specialized to the axis-aligned screen). Passes:

1. **Box pass** → `sceneTex`: back wall + four side walls as geometry with the selected backdrop
   (checkerboard procedural, glass tint, user photo texture, void), lit by the environment light,
   plus procedural caustics modulated by fluid thickness above.
2. **Depth pass** → `depthTex` (R32F view depth): sphere impostor sprites, depth-tested.
3. **Thickness pass** → `thickTex` (RG16F: thickness, foam): additive gaussian sprites; foam from
   particle speed above a threshold.
4. **Smooth** → bilateral filter on depth (two separable passes, narrow-range clamp).
5. **Composite** → drawable: reconstruct normals from smoothed depth; refraction samples
   `sceneTex` offset by normal × thickness; Beer–Lambert absorption; Fresnel reflection of a
   procedural environment tinted by the light estimate; specular from the light direction;
   metallic path for mercury; emission for lava; foam whitening; glass edge highlights.

### Perspective (`Input/PerspectiveController.swift`)

- **Tilt mode**: eye = R(ref⁻¹·current)ᵀ · (0, 0, E), E = 320 mm. The reference attitude slerps
  toward the current one (rate 0.4/s) so the view recentres when held still. Offset clamped.
- **Face mode** (ARKit `ARFaceTrackingConfiguration`, only if `isSupported`): face position via
  `camera.viewMatrix(for: .portrait)` → device frame, plus the front-camera offset from screen
  centre. Falls back to tilt mode when tracking is lost.

### Environment light (`Input/LightEstimator.swift`)

- Face mode: `ARDirectionalLightEstimate` → intensity, colour temperature → RGB, direction.
- Tilt mode: `UIScreen.brightness` (auto-brightness proxy) → intensity; default direction from
  above-front; neutral white. No camera use in tilt mode.
- Smoothed with a 1 s time constant.

### Interaction (`Input/TouchController.swift`, `Input/MotionController.swift`)

- Touch drag: up to 4 touches → cylinders through the tank depth applying velocity toward the drag
  velocity in `updateGrid`. Tap: radial outward/up impulse (splash).
- Shake: `|userAcceleration| > 1.8 g` → extra impulse burst + foam boost (gravity already includes
  user acceleration, so ordinary jerks slosh naturally).

### Feedback (`Feedback/HapticsController.swift`, `Feedback/SloshAudio.swift`)

- Haptics: Core Haptics transient events scaled by wall-impact energy (rate-limited 30 Hz).
- Audio: `AVAudioEngine` + `AVAudioSourceNode` — filtered noise bed following smoothed kinetic
  energy, plus transient splash bursts from impacts. Respects the silent switch (ambient category).

### UI (`UI/`)

Full-screen Metal view, status bar and home indicator hidden, idle timer disabled. A translucent
gear button fades after 3 s; two-finger tap brings it back. Settings sheet: fluid, backdrop
(+PhotosPicker), perspective mode, fill level, quality, haptics, sound, recentre. Settings persist
via `@AppStorage`.

### Concurrency (Swift 6 strict)

`Renderer`, controllers and settings are `@MainActor`. Core Motion is polled in the draw call (no
callbacks). ARKit runs its delegate on a private queue and publishes the latest pose/light through
a `Mutex`-protected value (`Synchronization`), read once per frame.

### Agent operability

Environment variables / launch arguments:
- `FLUID_GRAVITY=x,y,z` — fixed device-frame gravity in g (simulator has no accelerometer).
- `FLUID_GRAVITY_SWEEP=1` — scripted slow rotation of gravity.
- `FLUID_PRESET=<name>`, `FLUID_BACKDROP=<name>`, `FLUID_QUALITY=<name>`.
- `FLUID_SCREENSHOT_AFTER=<s>` — write `Documents/screenshot.png` from the drawable.
- `FLUID_STATS=1` — write `Documents/stats.json` (fps, sim ms, particle count) every second.
- `FLUID_EXIT_AFTER=<s>` — `exit(0)`.

## Testing

- Unit (Swift Testing, simulator): off-axis projection maps screen corners to NDC corners;
  gravity mapping; preset table sanity; colour temperature → RGB; perspective recentring.
- GPU (simulator Metal): build a small sim, run 300 substeps under tilted gravity, read back —
  all finite, all inside the box, count unchanged, fluid settled toward the gravity side.
- Device: performance via `FLUID_STATS`, screenshots via `devicectl device capture screenshot`.

## Out of scope

Multi-fluid mixing, App Store release work, iPad/landscape, recording/sharing.
