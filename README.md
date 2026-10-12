# PhoneFluid

An iOS toy that turns the iPhone into a glass tank of liquid. A 3D MLS-MPM fluid simulation runs
on the GPU (Metal compute), driven by the accelerometer, and is rendered through an off-axis
camera so the screen reads as a window into a deep box.

- **Fluids:** water, ocean, honey, oil, mercury, lava, slime, milk — each with its own physics
  (viscosity, cohesion, stiffness) and look (absorption, refraction, metallic, emission).
- **Perspective:** device tilt (default) or front-camera head tracking (ARKit TrueDepth).
- **Lighting:** follows the room via ARKit light estimation in head-tracking mode, or screen
  brightness (auto-brightness) otherwise. iOS has no public ambient-light-sensor API.
- **Interaction:** drag to stir, tap to splash, shake hard; haptics on wave impacts; procedural
  slosh audio (muted by the silent switch).
- **Backdrops:** checkerboard, glass tank, your photo, dark void.

Press and hold anywhere (or tap the faint button, top right) to open settings.

## Build and run

Requires Xcode 27, XcodeGen, and the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`).

```bash
make run-phone          # build, install and launch on "Pauls iPhone 18 Pro Max"
make run-sim            # dedicated simulator, fixed gravity (FLUID_GRAVITY, default 0,-1,0)
make checkall           # fmt, lint, typecheck, tests (incl. GPU simulation tests)
```

## Headless / agent overrides

| Variable | Effect |
| --- | --- |
| `FLUID_GRAVITY=x,y,z` | Fixed device-frame gravity in g (replaces Core Motion) |
| `FLUID_GRAVITY_SWEEP=1` | Slowly rotating gravity (only without `FLUID_GRAVITY`) |
| `FLUID_PRESET`, `FLUID_BACKDROP`, `FLUID_QUALITY` | Override settings at launch |
| `FLUID_STATS=1` | Write `Documents/stats.json` every second (fps, GPU ms, particles) |
| `FLUID_SCREENSHOT_AFTER=s` | Write `Documents/screenshot.png` from the drawable |
| `FLUID_EXIT_AFTER=s` | Exit after `s` seconds |

## Layout

- `PhoneFluid/Shaders/Simulation.metal` — MLS-MPM kernels (P2G ×2, grid update, G2P + stats).
- `PhoneFluid/Shaders/Render.metal` — tank walls, particle depth/thickness sprites, bilateral
  smoothing, composite (refraction via ray–box trace, Fresnel, caustics, foam, emission).
- `PhoneFluid/Sim/` — simulation host code, presets, tank geometry.
- `PhoneFluid/Render/` — frame loop and off-axis projection.
- `PhoneFluid/Input/`, `PhoneFluid/Feedback/`, `PhoneFluid/UI/` — sensors, haptics/audio, SwiftUI.
- `docs/specs/`, `docs/plans/` — design spec and implementation plan.
