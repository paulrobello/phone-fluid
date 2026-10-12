# PhoneFluid — Implementation Plan

Spec: `docs/specs/2026-10-11-phone-fluid-design.md`. Execution: inline (the sim core and renderer
share buffers and conventions, so they are built serially). Each phase ends with `make checkall`
and a commit.

## Phase 1 — Scaffold
- XcodeGen `project.yml` (app + unit tests), Makefile from `new_swift_project_template`,
  pre-commit, `.gitignore`, bridging header `ShaderTypes.h`.
- Portrait lock, `CADisableMinimumFrameDurationOnPhone`, hidden status bar/home indicator,
  idle timer off, `NSCameraUsageDescription`.
- `MTKView` wrapper + `Renderer` clearing the drawable at 120 Hz.
- Verify: `make checkall` green; app launches in the simulator.

## Phase 2 — MLS-MPM simulation
- `ShaderTypes.h` (`SimParams`, `Particle`, `FrameStats`), `Simulation.metal` kernels
  (clearGrid, p2g1, p2g2, updateGrid, g2p, stats), `FluidSimulation.swift` (capacity buffers,
  seeding, activeCount, encode).
- `FluidPreset.swift` physics fields.
- Verify: GPU test — 300 substeps under tilted gravity: finite, in-box, count conserved,
  centre of mass moves toward gravity.

## Phase 3 — Motion + config
- `MotionController` (deviceMotion polling, gravity − userAcceleration, shake detection),
  `LaunchConfig` (FLUID_* env), fixed/sweep gravity override.
- Verify: unit tests for gravity mapping and env parsing.

## Phase 4 — Screen-space fluid rendering + off-axis camera
- `Projection.swift` (off-axis frustum), `Render.metal` (box, depth sprites, thickness, bilateral,
  composite), `FluidRenderer.swift` (textures, pipelines).
- Backdrops: checker, glass, photo, void. Procedural caustics.
- Screenshot + stats + auto-exit operability hooks.
- Verify: projection unit test; simulator screenshot shows tank + fluid settled under FLUID_GRAVITY.

## Phase 5 — Perspective + light
- `PerspectiveController` (tilt with recentring; face mode via ARKit), `LightEstimator`
  (ARKit directional estimate / screen brightness), `ARFaceTracker` (Mutex-published state).
- Verify: unit tests for recentring and colour temperature; device run.

## Phase 6 — Presets (look + physics) and fill level
- Full preset table, preset switching re-seeds smoothly, fill level changes activeCount.
- Verify: unit test preset table; screenshots per preset in the simulator.

## Phase 7 — Interaction + feedback
- `TouchController` (drag/tap → GPU touch forces), shake burst, `HapticsController`,
  `SloshAudio` driven by stats readback.

## Phase 8 — UI
- Settings sheet (fluid, backdrop + PhotosPicker, perspective, fill, quality, haptics, sound,
  recentre), auto-hiding gear, two-finger tap, `@AppStorage` persistence.

## Phase 9 — Device verification + tuning
- Install on Pauls iPhone 18 Pro Max, measure fps per quality, capture screenshots, tune
  timeScale / substeps / default quality.
