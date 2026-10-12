<p align="center">
  <img src="https://raw.githubusercontent.com/paulrobello/phone-fluid/main/PhoneFluid/Assets.xcassets/AppIcon.appiconset/AppIcon.png" width="160" alt="PhoneFluid icon">
</p>

# PhoneFluid

An iOS toy that turns the iPhone into a glass tank of liquid. A 3D MLS-MPM fluid simulation runs
on the GPU (Metal compute), driven by the accelerometer, and is rendered through an off-axis
camera so the screen reads as a window into a deep box.

- **Fluids:** water, ocean, honey, oil, mercury, lava, slime, milk. Each has its own physics
  (viscosity, cohesion, thin-gap drag, wall stickiness) and look (absorption, refraction, metallic,
  emission, surface smoothing). Honey and lava creep; water sloshes.
- **Perspective:** device tilt (default) or front-camera head tracking (ARKit TrueDepth). When the
  face leaves the camera's view, the viewpoint follows Core Motion until tracking resumes.
- **Lighting:** follows the room via ARKit light estimation in head-tracking mode, or screen
  brightness (auto-brightness) otherwise. iOS has no public ambient-light-sensor API.
- **Look:** screen-space fluid rendering with refraction of the tank interior, Fresnel
  reflections, caustics, persistent whitewater foam, and thin films that stay continuous sheets.
- **Interaction:** drag to stir, tap to splash, shake hard; haptics on wave impacts; sampled slosh
  and splash audio (ElevenLabs, see `PhoneFluid/Resources/Audio-PROVENANCE.md`), muted by the
  silent switch.
- **Backdrops:** checkerboard, see-through glass tank, your photo, dark void.
- **Power:** settled liquid stops simulating and drops to 30 Hz; hot phones fall back to 60 Hz and
  half-speed liquid.

Press and hold anywhere (or tap the faint button, top right) to open settings.

## Quality tiers

Measured on a cool iPhone 18 Pro Max, all tiers hold full real-time speed:

| Tier | Cell size | Particles (40% fill) | Target |
| --- | --- | --- | --- |
| Low (default) | 2.6 mm | ~21k | 120 Hz (tilt mode) |
| Medium | 2.1 mm | ~43k | 60 Hz |
| High | 1.7 mm | ~76k | 60 Hz |

## Build and run

Requires Xcode 27, XcodeGen, swift-format, and the Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`). The Makefile defaults to the author's team and
phone; override them for your own device:

```bash
make run-phone TEAM_ID=<your team id> PHONE_DEVICE="<your iPhone name>"
```

```bash
make run-sim        # dedicated simulator, fixed gravity (FLUID_GRAVITY, default 0,-1,0)
make checkall       # fmt, lint, typecheck, tests (incl. GPU simulation tests in the simulator)
make bench          # time the simulation kernels on this Mac's GPU
```

Signing uses the `DEVELOPMENT_TEAM` in `project.yml`; change it (or pass `TEAM_ID`) to build for
your own device.

## Headless / agent overrides

| Variable | Effect |
| --- | --- |
| `FLUID_GRAVITY=x,y,z` | Fixed device-frame gravity in g (replaces Core Motion) |
| `FLUID_GRAVITY_SWEEP=1` | Slowly rotating gravity (only without `FLUID_GRAVITY`) |
| `FLUID_PRESET`, `FLUID_BACKDROP`, `FLUID_QUALITY`, `FLUID_PERSPECTIVE` | Override settings at launch |
| `FLUID_CELL_MM=mm` | Override the simulation cell size (profiling) |
| `FLUID_MAX_STEPS=n` | Cap simulation substeps per frame (profiling) |
| `FLUID_SCATTER=1` | Use the float-atomic scatter P2G instead of the gather path (A/B) |
| `FLUID_STATS=1` | Write `Documents/stats.json` every second (fps, GPU ms, sleep state, perspective trace) |
| `FLUID_SCREENSHOT_AFTER=s` | Write `Documents/screenshot.png` from the drawable |
| `FLUID_EXIT_AFTER=s` | Exit after `s` seconds |

## Layout

- `PhoneFluid/Shaders/Simulation.metal`: MLS-MPM kernels. Per substep: prefix scan, reorder
  particles by cell, gather mass, particle stress, gather stress fused with the grid update, G2P
  (which also bins particles for the next substep, generates foam and reduces stats).
- `PhoneFluid/Shaders/Render.metal`: tank walls and backdrops, particle depth/thickness sprites,
  bilateral smoothing with hole filling, composite (refraction via ray–box trace, Fresnel,
  caustics, foam, emission).
- `PhoneFluid/Sim/`: simulation host code, presets, tank geometry.
- `PhoneFluid/Render/`: frame loop, frame-rate controller, rest sleep, off-axis projection.
- `PhoneFluid/Input/`, `PhoneFluid/Feedback/`, `PhoneFluid/UI/`: sensors and ARKit, haptics and
  audio, SwiftUI.
- `tools/bench/`: headless macOS benchmark of the simulation kernels.
- `docs/specs/`, `docs/plans/`: design spec and implementation plan.

## License

MIT, see [LICENSE](LICENSE).
