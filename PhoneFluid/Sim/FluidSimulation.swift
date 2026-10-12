import Metal
import simd

enum SimulationError: Error {
  case missingFunction(String)
  case allocationFailed
}

/// GPU MLS-MPM fluid. Buffers are allocated at max capacity; `activeCount` drives dispatch.
@MainActor
final class FluidSimulation {
  static let seedSpacing: Float = 0.63
  static let maxFill: Float = 0.85
  /// The weakly compressible liquid settles ~15% denser than seeded; seed extra so the visible
  /// fill level matches the setting.
  static let compressionAllowance: Float = 1.15
  /// Presets were tuned at this cell size, time step and gravity. Other resolutions scale
  /// gravity (∝ cell size, keeping compression constant) and step rate (∝ 1 / cell size) so
  /// the liquid moves at the same physical speed at every quality.
  static let referenceCellMM: Float = 2.6
  static let referenceDt: Float = 0.2
  /// Largest sim time step; limited by the acoustic CFL condition (c·dt ≈ 0.65 cells).
  static let maxDt: Float = 0.4
  /// Sim gravity magnitude for 1 g at the reference cell size (cells / sim time^2).
  static let referenceGravity: Float = 0.04

  let device: MTLDevice
  let geometry: TankGeometry
  let capacity: Int
  private(set) var activeCount: Int = 0

  /// Particles and foam are ping-ponged: the gather path reorders them by cell every substep.
  private let particleBuffers: [MTLBuffer]
  private let foamBuffers: [MTLBuffer]
  private var current = 0
  /// The buffer holding the latest particle state (CPU writes and rendering use this).
  var particleBuffer: MTLBuffer { particleBuffers[current] }
  /// Per-particle whitewater amount 0...1, read by the renderer.
  var foamBuffer: MTLBuffer { foamBuffers[current] }
  let gridBuffer: MTLBuffer
  let gridVelocityBuffer: MTLBuffer

  private let clearPipeline: MTLComputePipelineState
  private let p2g1Pipeline: MTLComputePipelineState
  private let p2g2Pipeline: MTLComputePipelineState
  private let updateGridPipeline: MTLComputePipelineState
  private let g2pPipeline: MTLComputePipelineState
  private let clearCountsPipeline: MTLComputePipelineState
  private let countPipeline: MTLComputePipelineState
  private let scanPipeline: MTLComputePipelineState
  private let reorderPipeline: MTLComputePipelineState
  private let gatherMassPipeline: MTLComputePipelineState
  private let stressPipeline: MTLComputePipelineState
  private let gatherStressPipeline: MTLComputePipelineState

  private let cellCountBuffer: MTLBuffer
  private let cellStartBuffer: MTLBuffer
  private let rankBuffer: MTLBuffer
  private let termBuffer: MTLBuffer
  private let densityBuffer: MTLBuffer

  /// Atomic-free P2G (counting sort + per-node gather) instead of float-atomic scatter.
  var useGather = true
  var wallStiffness: Float = 0.3
  var affineDamping: Float = 1
  private let scanThreads: Int

  var preset: FluidPreset = .water
  var touches: [TouchInput] = []
  private var rng = SystemRandomNumberGenerator()

  init(device: MTLDevice, library: MTLLibrary, geometry: TankGeometry) throws {
    self.device = device
    self.geometry = geometry
    let interiorVolume = Float(geometry.interior.x * geometry.interior.y * geometry.interior.z)
    let perCell = 1 / pow(Self.seedSpacing, 3)
    capacity = Int(interiorVolume * perCell * Self.maxFill) + 1024

    guard
      let particlesA = device.makeBuffer(
        length: capacity * MemoryLayout<Particle>.stride, options: .storageModeShared),
      let particlesB = device.makeBuffer(
        length: capacity * MemoryLayout<Particle>.stride, options: .storageModeShared),
      let grid = device.makeBuffer(
        length: geometry.nodeCount * MemoryLayout<SIMD4<Float>>.stride,
        options: .storageModePrivate),
      let gridVel = device.makeBuffer(
        length: geometry.nodeCount * MemoryLayout<SIMD4<Float>>.stride,
        options: .storageModePrivate)
    else { throw SimulationError.allocationFailed }
    let uintSize = MemoryLayout<UInt32>.stride
    guard
      let counts = device.makeBuffer(
        length: geometry.nodeCount * uintSize, options: .storageModePrivate),
      let starts = device.makeBuffer(
        length: (geometry.nodeCount + 1) * uintSize, options: .storageModePrivate),
      let rank = device.makeBuffer(length: capacity * uintSize, options: .storageModePrivate),
      let terms = device.makeBuffer(
        length: capacity * MemoryLayout<simd_float3x3>.stride, options: .storageModePrivate),
      let densities = device.makeBuffer(
        length: capacity * MemoryLayout<Float>.stride, options: .storageModePrivate),
      let foamA = device.makeBuffer(
        length: capacity * MemoryLayout<Float>.stride, options: .storageModeShared),
      let foamB = device.makeBuffer(
        length: capacity * MemoryLayout<Float>.stride, options: .storageModeShared)
    else { throw SimulationError.allocationFailed }
    densityBuffer = densities
    foamBuffers = [foamA, foamB]
    for buffer in foamBuffers { memset(buffer.contents(), 0, buffer.length) }
    cellCountBuffer = counts
    cellStartBuffer = starts
    rankBuffer = rank
    termBuffer = terms
    particlesA.label = "particlesA"
    particlesB.label = "particlesB"
    grid.label = "grid"
    gridVel.label = "gridVelocity"
    particleBuffers = [particlesA, particlesB]
    gridBuffer = grid
    gridVelocityBuffer = gridVel

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let fn = library.makeFunction(name: name) else {
        throw SimulationError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: fn)
    }
    clearPipeline = try pipeline("clearGrid")
    p2g1Pipeline = try pipeline("p2g1")
    p2g2Pipeline = try pipeline("p2g2")
    updateGridPipeline = try pipeline("updateGrid")
    g2pPipeline = try pipeline("g2p")
    clearCountsPipeline = try pipeline("clearCounts")
    countPipeline = try pipeline("countParticles")
    scanPipeline = try pipeline("scanCounts")
    reorderPipeline = try pipeline("reorderParticles")
    gatherMassPipeline = try pipeline("p2gGatherMass")
    stressPipeline = try pipeline("particleStress")
    gatherStressPipeline = try pipeline("p2gGatherStress")
    scanThreads = min(1024, scanPipeline.maxTotalThreadsPerThreadgroup / 32 * 32)
  }

  // MARK: - Particles

  /// Number of particles that fills `fraction` of the tank volume.
  func particleCount(forFill fraction: Float) -> Int {
    let f = min(max(fraction, 0.02), Self.maxFill)
    let interiorVolume = Float(geometry.interior.x * geometry.interior.y * geometry.interior.z)
    return min(
      capacity, Int(interiorVolume * f * Self.compressionAllowance / pow(Self.seedSpacing, 3)))
  }

  /// Fills the bottom of the tank (toward `down`, a device-frame direction) with a resting block
  /// seeded at exactly the rest spacing so the liquid starts at rest density.
  func reset(fill: Float, down: SIMD3<Float> = [0, -1, 0]) {
    let target = particleCount(forFill: fill)
    let lo = SIMD3<Float>(repeating: Float(TankGeometry.wallMargin))
    let interior = SIMD3<Float>(geometry.interior)
    let counts = SIMD3<Int>(
      max(1, Int((interior.x / Self.seedSpacing).rounded())),
      max(1, Int((interior.y / Self.seedSpacing).rounded())),
      max(1, Int((interior.z / Self.seedSpacing).rounded())))
    let spacing = interior / SIMD3<Float>(counts)

    let axis = Self.dominantAxis(down)
    let fromLow = down[axis] < 0
    let otherA = (axis + 1) % 3
    let otherB = (axis + 2) % 3
    let layerSize = counts[otherA] * counts[otherB]
    let layers = min(counts[axis], (target + layerSize - 1) / max(layerSize, 1))

    let ptr = particleBuffer.contents().bindMemory(to: Particle.self, capacity: capacity)
    var n = 0
    outer: for layer in 0..<layers {
      for a in 0..<counts[otherA] {
        for b in 0..<counts[otherB] {
          if n >= target { break outer }
          let l = fromLow ? layer : counts[axis] - 1 - layer
          var p = SIMD3<Float>.zero
          p[axis] = lo[axis] + (Float(l) + 0.5) * spacing[axis]
          p[otherA] = lo[otherA] + (Float(a) + 0.5) * spacing[otherA]
          p[otherB] = lo[otherB] + (Float(b) + 0.5) * spacing[otherB]
          p += SIMD3<Float>(jitter(), jitter(), jitter())
          ptr[n] = Particle(
            position: p, velocity: .zero, c0: .zero, c1: .zero, c2: .zero)
          n += 1
        }
      }
    }
    activeCount = n
    memset(foamBuffer.contents(), 0, foamBuffer.length)
  }

  /// Grows or shrinks the particle count toward the fill fraction. New particles pour in from
  /// the "up" side of the tank.
  func setFill(_ fill: Float, up: SIMD3<Float>) {
    let target = particleCount(forFill: fill)
    if target <= activeCount {
      activeCount = target
      return
    }
    let size = SIMD3<Float>(geometry.gridSize)
    let margin = Float(TankGeometry.wallMargin) + 1
    let centre = size / 2
    let upDir = simd_length(up) > 0.1 ? simd_normalize(up) : SIMD3<Float>(0, 1, 0)
    let halfExtent = size / 2 - margin
    // Pour from a column just inside the "up" wall.
    let reach = min(
      abs(upDir.x) > 1e-3 ? halfExtent.x / abs(upDir.x) : .infinity,
      abs(upDir.y) > 1e-3 ? halfExtent.y / abs(upDir.y) : .infinity,
      abs(upDir.z) > 1e-3 ? halfExtent.z / abs(upDir.z) : .infinity)
    let source = centre + upDir * (reach - 2)
    let ptr = particleBuffer.contents().bindMemory(to: Particle.self, capacity: capacity)
    let add = min(target - activeCount, 6000)
    for i in 0..<add {
      var p = source + SIMD3<Float>(jitter() * 8, jitter() * 8, jitter() * 8)
      p = simd_clamp(p, SIMD3<Float>(repeating: margin), size - margin)
      ptr[activeCount + i] = Particle(
        position: p, velocity: -upDir * 0.5, c0: .zero, c1: .zero, c2: .zero)
    }
    let foam = foamBuffer.contents().bindMemory(to: Float.self, capacity: capacity)
    for i in 0..<add { foam[activeCount + i] = 0 }
    activeCount += add
  }

  private func jitter() -> Float { Float.random(in: -0.12...0.12, using: &rng) }

  static func dominantAxis(_ v: SIMD3<Float>) -> Int {
    let a = abs(v)
    if a.x >= a.y && a.x >= a.z { return 0 }
    return a.y >= a.z ? 1 : 2
  }

  // MARK: - Stepping

  /// Device-frame effective gravity (in g) → sim gravity.
  func simGravity(_ g: SIMD3<Float>) -> SIMD3<Float> {
    g * Self.referenceGravity * resolutionScale * preset.gravityScale
  }

  /// Cell size relative to the reference (< 1 for finer qualities).
  var resolutionScale: Float { geometry.cellMM / Self.referenceCellMM }

  /// Viscosity adjusted so physical (mm²/s) viscosity is the same at every resolution.
  var effectiveViscosity: Float { preset.viscosity / resolutionScale }

  /// Sim time step: as large as stability allows. Explicit viscosity needs ν·dt ≤ 0.15 cell².
  var dt: Float {
    let nu = effectiveViscosity / preset.restDensity
    return min(Self.maxDt, nu > 0 ? 0.15 / nu : Self.maxDt)
  }

  /// Substeps per real second needed for real-time motion with the current time step.
  var stepsPerSecond: Float {
    preset.stepsPerSecond / resolutionScale * (Self.referenceDt / dt)
  }

  func makeParams(gravity: SIMD3<Float>) -> SimParams {
    var params = SimParams()
    params.gridSize = geometry.gridSize
    params.particleCount = UInt32(activeCount)
    params.gravity = simGravity(gravity)
    params.dt = dt
    params.restDensity = preset.restDensity
    params.stiffness = preset.stiffness
    params.eosPower = preset.eosPower
    params.viscosity = effectiveViscosity
    params.cohesion = preset.cohesion
    params.wallMargin = Float(TankGeometry.wallMargin)
    // At most ~0.75 cells of travel per step.
    params.maxSpeed = 0.75 / dt
    params.wallStiffness = wallStiffness
    params.affineDamping = affineDamping
    // Per-real-second rates → per-step fractions.
    params.drag = 1 - exp(-preset.drag / stepsPerSecond)
    params.wallFriction = 1 - exp(-preset.wallFriction / stepsPerSecond)
    params.foamSpeed = preset.foamSpeed
    // Foam half-life ~0.6 s of real time regardless of step rate.
    params.foamDecay = pow(0.5, 1 / (0.6 * stepsPerSecond))
    let count = min(touches.count, Int(MAX_TOUCHES))
    params.touchCount = UInt32(count)
    withUnsafeMutableBytes(of: &params.touches) { raw in
      let slots = raw.bindMemory(to: TouchInput.self)
      for i in 0..<count { slots[i] = touches[i] }
    }
    return params
  }

  /// Encodes `steps` substeps. Stats accumulate into `statsBuffer` (cleared by the caller).
  func encode(
    into commandBuffer: MTLCommandBuffer, steps: Int, gravity: SIMD3<Float>,
    statsBuffer: MTLBuffer, statsOffset: Int
  ) {
    guard activeCount > 0, steps > 0 else { return }
    var params = makeParams(gravity: gravity)
    params.countCells = useGather ? 1 : 0
    guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
    enc.label = "MLS-MPM"
    let nodes = geometry.nodeCount
    let particles = activeCount
    let paramSize = MemoryLayout<SimParams>.stride

    // Gather path: bin particles once per frame; afterwards g2p bins them for the next substep
    // and the prefix scan clears the counts again.
    if useGather {
      enc.setComputePipelineState(clearCountsPipeline)
      enc.setBuffer(cellCountBuffer, offset: 0, index: 0)
      enc.setBytes(&params, length: paramSize, index: 1)
      dispatch(enc, clearCountsPipeline, count: nodes)

      enc.setComputePipelineState(countPipeline)
      enc.setBuffer(particleBuffer, offset: 0, index: 0)
      enc.setBuffer(cellCountBuffer, offset: 0, index: 1)
      enc.setBuffer(rankBuffer, offset: 0, index: 2)
      enc.setBytes(&params, length: paramSize, index: 3)
      dispatch(enc, countPipeline, count: particles)
    }

    for step in 0..<steps {
      // Touch splashes are one-shot impulses; apply them on the first substep only.
      if step == 1 {
        withUnsafeMutableBytes(of: &params.touches) { raw in
          let slots = raw.bindMemory(to: TouchInput.self)
          for i in 0..<Int(params.touchCount) { slots[i].velocity.z = 0 }
        }
      }
      if useGather {
        encodeGatherP2G(enc, params: &params, nodes: nodes, particles: particles)
      } else {
        enc.setComputePipelineState(clearPipeline)
        enc.setBuffer(gridBuffer, offset: 0, index: 0)
        enc.setBytes(&params, length: paramSize, index: 1)
        dispatch(enc, clearPipeline, count: nodes)

        enc.setComputePipelineState(p2g1Pipeline)
        enc.setBuffer(particleBuffer, offset: 0, index: 0)
        enc.setBuffer(gridBuffer, offset: 0, index: 1)
        enc.setBytes(&params, length: paramSize, index: 2)
        dispatch(enc, p2g1Pipeline, count: particles)

        enc.setComputePipelineState(p2g2Pipeline)
        enc.setBuffer(densityBuffer, offset: 0, index: 3)
        dispatch(enc, p2g2Pipeline, count: particles)

        enc.setComputePipelineState(updateGridPipeline)
        enc.setBuffer(gridBuffer, offset: 0, index: 0)
        enc.setBuffer(gridVelocityBuffer, offset: 0, index: 1)
        enc.setBytes(&params, length: paramSize, index: 2)
        dispatch(enc, updateGridPipeline, count: nodes)
      }

      enc.setComputePipelineState(g2pPipeline)
      enc.setBuffer(particleBuffer, offset: 0, index: 0)
      enc.setBuffer(gridVelocityBuffer, offset: 0, index: 1)
      enc.setBytes(&params, length: paramSize, index: 2)
      enc.setBuffer(statsBuffer, offset: statsOffset, index: 3)
      enc.setBuffer(statsBuffer, offset: statsOffset, index: 4)
      enc.setBuffer(densityBuffer, offset: 0, index: 5)
      enc.setBuffer(foamBuffer, offset: 0, index: 6)
      enc.setBuffer(cellCountBuffer, offset: 0, index: 7)
      enc.setBuffer(rankBuffer, offset: 0, index: 8)
      dispatch(enc, g2pPipeline, count: particles)
    }
    enc.endEncoding()
  }

  /// Scan (also clears counts) → reorder particles by cell → gather mass/momentum → per-particle stress →
  /// gather stress fused with the grid update. Requires counts and ranks from the previous g2p
  /// (or the per-frame count pass).
  private func encodeGatherP2G(
    _ enc: MTLComputeCommandEncoder, params: inout SimParams, nodes: Int, particles: Int
  ) {
    let paramSize = MemoryLayout<SimParams>.stride

    enc.setComputePipelineState(scanPipeline)
    enc.setBuffer(cellCountBuffer, offset: 0, index: 0)
    enc.setBuffer(cellStartBuffer, offset: 0, index: 1)
    enc.setBytes(&params, length: paramSize, index: 2)
    enc.dispatchThreadgroups(
      MTLSize(width: 1, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: scanThreads, height: 1, depth: 1))

    let src = current
    current = 1 - current
    enc.setComputePipelineState(reorderPipeline)
    enc.setBuffer(particleBuffers[src], offset: 0, index: 0)
    enc.setBuffer(rankBuffer, offset: 0, index: 1)
    enc.setBuffer(cellStartBuffer, offset: 0, index: 2)
    enc.setBuffer(particleBuffer, offset: 0, index: 3)
    enc.setBytes(&params, length: paramSize, index: 4)
    enc.setBuffer(foamBuffers[src], offset: 0, index: 5)
    enc.setBuffer(foamBuffer, offset: 0, index: 6)
    dispatch(enc, reorderPipeline, count: particles)

    enc.setComputePipelineState(gatherMassPipeline)
    enc.setBuffer(particleBuffer, offset: 0, index: 0)
    enc.setBuffer(cellStartBuffer, offset: 0, index: 1)
    enc.setBuffer(gridBuffer, offset: 0, index: 4)
    enc.setBytes(&params, length: paramSize, index: 5)
    dispatch(enc, gatherMassPipeline, count: nodes)

    enc.setComputePipelineState(stressPipeline)
    enc.setBuffer(particleBuffer, offset: 0, index: 0)
    enc.setBuffer(gridBuffer, offset: 0, index: 1)
    enc.setBuffer(termBuffer, offset: 0, index: 2)
    enc.setBytes(&params, length: paramSize, index: 3)
    enc.setBuffer(densityBuffer, offset: 0, index: 4)
    dispatch(enc, stressPipeline, count: particles)

    enc.setComputePipelineState(gatherStressPipeline)
    enc.setBuffer(particleBuffer, offset: 0, index: 0)
    enc.setBuffer(cellStartBuffer, offset: 0, index: 1)
    enc.setBuffer(gridVelocityBuffer, offset: 0, index: 2)
    enc.setBuffer(gridBuffer, offset: 0, index: 4)
    enc.setBytes(&params, length: paramSize, index: 5)
    enc.setBuffer(termBuffer, offset: 0, index: 6)
    dispatch(enc, gatherStressPipeline, count: nodes)
  }

  private func dispatch(
    _ enc: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState, count: Int
  ) {
    let width = min(pipeline.maxTotalThreadsPerThreadgroup, 64)
    enc.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
  }
}
