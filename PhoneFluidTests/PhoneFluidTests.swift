import Foundation
import Metal
import Testing
import simd

@testable import PhoneFluid

struct ProjectionTests {
  @Test func screenCornersMapToNDCCorners() {
    let eye = SIMD3<Float>(23, -41, 300)
    let view = OffAxisProjection.view(eye: eye)
    let proj = OffAxisProjection.projection(
      eye: eye, screenMin: [-36, -79], screenMax: [36, 79], near: 150, far: 600)
    for (corner, expected) in [
      (SIMD3<Float>(-36, -79, 0), SIMD2<Float>(-1, -1)),
      (SIMD3<Float>(36, 79, 0), SIMD2<Float>(1, 1)),
      (SIMD3<Float>(36, -79, 0), SIMD2<Float>(1, -1)),
    ] {
      let clip = proj * view * SIMD4<Float>(corner, 1)
      let ndc = SIMD2<Float>(clip.x, clip.y) / clip.w
      #expect(simd_distance(ndc, expected) < 1e-4)
    }
  }

  @Test func depthIncreasesIntoTheTank() {
    let eye = SIMD3<Float>(0, 0, 320)
    let mvp =
      OffAxisProjection.projection(
        eye: eye, screenMin: [-36, -79], screenMax: [36, 79], near: 160, far: 600)
      * OffAxisProjection.view(eye: eye)
    let front = mvp * SIMD4<Float>(0, 0, 0, 1)
    let back = mvp * SIMD4<Float>(0, 0, -24, 1)
    let zf = front.z / front.w
    let zb = back.z / back.w
    #expect(zf > 0 && zf < 1)
    #expect(zb > zf && zb < 1)
  }
}

struct InputTests {
  @Test func effectiveGravitySubtractsUserAcceleration() {
    let g = MotionController.effectiveGravity(gravity: [0, -1, 0], user: [0, 0.5, 0])
    #expect(simd_distance(g, [0, -1.5, 0]) < 1e-6)
    let clamped = MotionController.effectiveGravity(gravity: [0, -1, 0], user: [0, 9, 0])
    #expect(abs(simd_length(clamped) - 3) < 1e-4)
  }

  @Test @MainActor func tiltEyeIsCentredAtReference() {
    let q = simd_quatf(angle: 0.3, axis: simd_normalize(SIMD3<Float>(1, 0.2, 0)))
    let eye = PerspectiveController.tiltEye(reference: q, current: q, gain: 1)
    #expect(simd_distance(eye, [0, 0, PerspectiveController.eyeDistance]) < 1e-3)
  }

  @Test @MainActor func tiltingShiftsEyeAndClampsOffset() {
    let ref = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    let tilted = simd_quatf(angle: 0.2, axis: [0, 1, 0])
    let eye = PerspectiveController.tiltEye(reference: ref, current: tilted, gain: 1)
    #expect(abs(eye.x) > 30)
    let extreme = simd_quatf(angle: 1.2, axis: [0, 1, 0])
    let far = PerspectiveController.tiltEye(reference: ref, current: extreme, gain: 3)
    #expect(simd_length(SIMD2<Float>(far.x, far.y)) <= PerspectiveController.maxOffset + 1e-3)
  }

  @Test func launchConfigParsesOverrides() {
    let config = LaunchConfig(environment: [
      "FLUID_GRAVITY": "0.5, -1, 0", "FLUID_STATS": "1", "FLUID_PRESET": "honey",
      "FLUID_EXIT_AFTER": "4",
    ])
    #expect(config.gravity == SIMD3<Float>(0.5, -1, 0))
    #expect(config.writeStats)
    #expect(config.preset == "honey")
    #expect(config.exitAfter == 4)
    #expect(LaunchConfig(environment: ["FLUID_GRAVITY": "bad"]).gravity == nil)
  }

  @Test func colourTemperatureIsWarmToCool() {
    let warm = LightEstimator.rgb(kelvin: 2700)
    let cool = LightEstimator.rgb(kelvin: 9000)
    let neutral = LightEstimator.rgb(kelvin: 6500)
    #expect(warm.x > warm.z)
    #expect(cool.z > cool.x)
    #expect(abs(neutral.x - neutral.z) < 0.08)
  }

  @Test func faceCameraMappingPutsUserInFront() {
    // A face 30 cm in front of the camera, centred: camera view z = -0.3 m.
    let eye = ARFaceTracker.deviceFromCameraView([0, 0, -0.3])
    #expect(eye.z > 290)
    #expect(abs(eye.x) < 1e-3)
  }
}

struct PresetTests {
  @Test func presetsAreUniqueAndPhysical() {
    let ids = Set(FluidPreset.all.map(\.id))
    #expect(ids.count == FluidPreset.all.count)
    for p in FluidPreset.all {
      #expect(p.restDensity > 0 && p.stiffness > 0 && p.viscosity >= 0)
      #expect(p.cohesion >= 0 && p.cohesion < 1)
      #expect(p.stepsPerSecond >= 120)
    }
    #expect(FluidPreset.named("HONEY").id == "honey")
    #expect(FluidPreset.named("nope").id == "water")
    #expect(FluidPreset.honey.viscosity > FluidPreset.water.viscosity * 5)
  }

  @Test func tankGeometryMatchesScreen() {
    let geo = TankGeometry(screenPoints: [440, 956], quality: .medium)
    #expect(abs(geo.widthMM - 440 * TankGeometry.mmPerPoint) < 1e-3)
    #expect(geo.gridSize.x == UInt32(geo.interior.x + 2 * TankGeometry.wallMargin))
    let p = SIMD3<Float>(10, 20, 5)
    #expect(simd_distance(geo.simFromWorld(geo.worldFromSim(p)), p) < 1e-3)
    #expect(simd_distance(geo.worldFromSim([2, 2, 2]), geo.boxMin) < 1e-3)
  }
}

@MainActor
struct SimulationTests {
  private func makeSim(preset: FluidPreset = .water) throws -> FluidSimulation {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let library = try #require(device.makeDefaultLibrary())
    let geometry = TankGeometry(widthMM: 40, heightMM: 60, depthMM: 16, quality: .low)
    let sim = try FluidSimulation(device: device, library: library, geometry: geometry)
    sim.preset = preset
    return sim
  }

  private func run(_ sim: FluidSimulation, steps: Int, gravity: SIMD3<Float>) throws {
    let device = sim.device
    let queue = try #require(device.makeCommandQueue())
    let stats = try #require(
      device.makeBuffer(length: MemoryLayout<FrameStats>.stride, options: .storageModeShared))
    var remaining = steps
    while remaining > 0 {
      let batch = min(remaining, 25)
      let cb = try #require(queue.makeCommandBuffer())
      sim.encode(into: cb, steps: batch, gravity: gravity, statsBuffer: stats, statsOffset: 0)
      cb.commit()
      cb.waitUntilCompleted()
      #expect(cb.error == nil)
      remaining -= batch
    }
  }

  private func particles(_ sim: FluidSimulation) -> [Particle] {
    let ptr = sim.particleBuffer.contents().bindMemory(to: Particle.self, capacity: sim.capacity)
    return Array(UnsafeBufferPointer(start: ptr, count: sim.activeCount))
  }

  private func centroid(_ ps: [Particle]) -> SIMD3<Float> {
    ps.reduce(SIMD3<Float>.zero) { $0 + $1.position } / Float(ps.count)
  }

  @Test func fluidStaysFiniteBoundedAndFlowsDownhill() throws {
    for preset in [FluidPreset.water, .honey, .mercury] {
      let sim = try makeSim(preset: preset)
      sim.reset(fill: 0.3, down: [0, -1, 0])
      let count = sim.activeCount
      #expect(count > 1000)
      let before = centroid(particles(sim))

      try run(sim, steps: 1500, gravity: simd_normalize(SIMD3<Float>(1, -0.6, 0)))

      let after = particles(sim)
      #expect(sim.activeCount == count)
      let size = SIMD3<Float>(sim.geometry.gridSize)
      for p in after {
        #expect(p.position.x.isFinite && p.position.y.isFinite && p.position.z.isFinite)
        #expect(all(p.position .>= SIMD3<Float>(repeating: 1)))
        #expect(all(p.position .<= size - 2))
      }
      let c = centroid(after)
      #expect(c.x > before.x + 1, "\(preset.name) should flow toward +x")
    }
  }

  @Test func restingWaterDoesNotCollapse() throws {
    let sim = try makeSim()
    sim.reset(fill: 0.3, down: [0, -1, 0])
    let top0 = particles(sim).map(\.position.y).max() ?? 0
    try run(sim, steps: 2000, gravity: [0, -1, 0])
    let ps = particles(sim)
    let top = ps.map(\.position.y).sorted()[Int(Double(ps.count) * 0.98)]
    // Weakly compressible: the surface settles within ~25% of its seeded height.
    #expect(top > top0 * 0.75, "top0=\(top0) top=\(top) ")
    #expect(top < top0 * 1.15)
  }

  @Test func fillLevelChangesParticleCount() throws {
    let sim = try makeSim()
    sim.reset(fill: 0.3)
    let base = sim.activeCount
    sim.setFill(0.1, up: [0, 1, 0])
    #expect(sim.activeCount < base)
    sim.setFill(0.3, up: [0, 1, 0])
    #expect(sim.activeCount > sim.particleCount(forFill: 0.1))
    #expect(sim.activeCount <= sim.capacity)
  }

  @Test func gatherMatchesScatter() throws {
    let a = try makeSim()
    let b = try makeSim()
    a.reset(fill: 0.3, down: [0, -1, 0])
    memcpy(
      b.particleBuffer.contents(), a.particleBuffer.contents(),
      a.capacity * MemoryLayout<Particle>.stride)
    b.reset(fill: 0.3, down: [0, -1, 0])
    memcpy(
      b.particleBuffer.contents(), a.particleBuffer.contents(),
      a.capacity * MemoryLayout<Particle>.stride)
    a.useGather = false
    b.useGather = true
    let g = simd_normalize(SIMD3<Float>(0.7, -1, 0.2))
    try run(a, steps: 5, gravity: g)
    try run(b, steps: 5, gravity: g)
    // The gather path reorders particles by cell, so compare order-independent statistics:
    // sorted per-axis position and velocity distributions.
    let pa = particles(a)
    let pb = particles(b)
    #expect(pa.count == pb.count)
    var worst: Float = 0
    for axis in 0..<3 {
      let xa = pa.map { $0.position[axis] }.sorted()
      let xb = pb.map { $0.position[axis] }.sorted()
      let va = pa.map { $0.velocity[axis] }.sorted()
      let vb = pb.map { $0.velocity[axis] }.sorted()
      for i in 0..<xa.count {
        worst = max(worst, abs(xa[i] - xb[i]), abs(va[i] - vb[i]))
      }
    }
    #expect(worst < 1e-3, "max divergence \(worst)")
  }

  @Test func timeScatterVersusGather() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let library = try #require(device.makeDefaultLibrary())
    let geometry = TankGeometry(widthMM: 66, heightMM: 145, depthMM: 24, quality: .medium)
    var report = ""
    for gather in [false, true] {
      let sim = try FluidSimulation(device: device, library: library, geometry: geometry)
      sim.useGather = gather
      sim.reset(fill: 0.4, down: [0, -1, 0])
      try run(sim, steps: 50, gravity: [0.5, -1, 0])
      let queue = try #require(device.makeCommandQueue())
      let stats = try #require(device.makeBuffer(length: 16, options: .storageModeShared))
      let cb = try #require(queue.makeCommandBuffer())
      sim.encode(into: cb, steps: 100, gravity: [0.5, -1, 0], statsBuffer: stats, statsOffset: 0)
      cb.commit()
      cb.waitUntilCompleted()
      report +=
        "gather=\(gather) n=\(sim.activeCount) ms/step=\((cb.gpuEndTime - cb.gpuStartTime) * 10) "
    }
    print("PERF \(report)")
  }

  @Test func restingSurfaceIsCalm() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let library = try #require(device.makeDefaultLibrary())
    let geometry = TankGeometry(widthMM: 72, heightMM: 158, depthMM: 24, quality: .low)
    let sim = try FluidSimulation(device: device, library: library, geometry: geometry)
    sim.reset(fill: 0.4, down: [0, -1, 0])
    var samples: [Float] = []
    for _ in 0..<8 {
      try run(sim, steps: 800, gravity: [0, -1, 0])
      let ps = particles(sim).sorted { $0.position.y < $1.position.y }
      let n = ps.count / 20
      samples.append(ps.suffix(n).map { abs($0.velocity.y) }.reduce(0, +) / Float(n))
    }
    print("CALM \(samples)")
    let topVy = samples.last ?? 1
    // Measured 0.13-0.18 with the old stiff EOS (visibly boiling surface), ~0.005 now.
    #expect(topVy < 0.02, "surface vertical jitter \(topVy)")
  }

  @Test func foamFormsWhenSloshingAndFadesAtRest() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let library = try #require(device.makeDefaultLibrary())
    let geometry = TankGeometry(widthMM: 72, heightMM: 158, depthMM: 24, quality: .low)
    let sim = try FluidSimulation(device: device, library: library, geometry: geometry)
    sim.reset(fill: 0.4, down: [0, -1, 0])
    func foamyFraction() -> Float {
      let foam = sim.foamBuffer.contents().bindMemory(to: Float.self, capacity: sim.capacity)
      let f = UnsafeBufferPointer(start: foam, count: sim.activeCount)
      return Float(f.filter { $0 > 0.2 }.count) / Float(f.count)
    }
    try run(sim, steps: 3000, gravity: [0, -1, 0])
    #expect(foamyFraction() < 0.01, "resting liquid should carry no foam")
    try run(sim, steps: 300, gravity: simd_normalize(SIMD3<Float>(1, -0.25, 0)))
    let sloshing = foamyFraction()
    // Measured ~16%: whitewater at crests and impacts, not the whole body.
    #expect(sloshing > 0.03 && sloshing < 0.4, "sloshing foam fraction \(sloshing)")
  }
}
