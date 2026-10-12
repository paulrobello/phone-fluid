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
  static let simDt: Float = 0.2
  /// Sim gravity magnitude for 1 g (cells / sim time^2).
  static let gravityUnit: Float = 0.04

  let device: MTLDevice
  let geometry: TankGeometry
  let capacity: Int
  private(set) var activeCount: Int = 0

  let particleBuffer: MTLBuffer
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
  private let scatterIndicesPipeline: MTLComputePipelineState
  private let gatherMassPipeline: MTLComputePipelineState
  private let stressPipeline: MTLComputePipelineState
  private let gatherStressPipeline: MTLComputePipelineState

  private let cellCountBuffer: MTLBuffer
  private let cellStartBuffer: MTLBuffer
  private let rankBuffer: MTLBuffer
  private let sortedBuffer: MTLBuffer
  private let termBuffer: MTLBuffer

  /// Atomic-free P2G (counting sort + per-node gather) instead of float-atomic scatter.
  var useGather = true
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
      let particles = device.makeBuffer(
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
        length: geometry.nodeCount * uintSize, options: .storageModePrivate),
      let rank = device.makeBuffer(length: capacity * uintSize, options: .storageModePrivate),
      let sorted = device.makeBuffer(length: capacity * uintSize, options: .storageModePrivate),
      let terms = device.makeBuffer(
        length: capacity * MemoryLayout<simd_float3x3>.stride, options: .storageModePrivate)
    else { throw SimulationError.allocationFailed }
    cellCountBuffer = counts
    cellStartBuffer = starts
    rankBuffer = rank
    sortedBuffer = sorted
    termBuffer = terms
    particles.label = "particles"
    grid.label = "grid"
    gridVel.label = "gridVelocity"
    particleBuffer = particles
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
    scatterIndicesPipeline = try pipeline("scatterIndices")
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
    return min(capacity, Int(interiorVolume * f / pow(Self.seedSpacing, 3)))
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
    g * Self.gravityUnit * preset.gravityScale
  }

  func makeParams(gravity: SIMD3<Float>) -> SimParams {
    var params = SimParams()
    params.gridSize = geometry.gridSize
    params.particleCount = UInt32(activeCount)
    params.gravity = simGravity(gravity)
    params.dt = Self.simDt
    params.restDensity = preset.restDensity
    params.stiffness = preset.stiffness
    params.eosPower = preset.eosPower
    params.viscosity = preset.viscosity
    params.cohesion = preset.cohesion
    params.wallMargin = Float(TankGeometry.wallMargin)
    params.maxSpeed = 3.0
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
    guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
    enc.label = "MLS-MPM"
    let nodes = geometry.nodeCount
    let particles = activeCount
    let paramSize = MemoryLayout<SimParams>.stride

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
        dispatch(enc, p2g2Pipeline, count: particles)
      }

      enc.setComputePipelineState(updateGridPipeline)
      enc.setBuffer(gridBuffer, offset: 0, index: 0)
      enc.setBuffer(gridVelocityBuffer, offset: 0, index: 1)
      enc.setBytes(&params, length: paramSize, index: 2)
      dispatch(enc, updateGridPipeline, count: nodes)

      enc.setComputePipelineState(g2pPipeline)
      enc.setBuffer(particleBuffer, offset: 0, index: 0)
      enc.setBuffer(gridVelocityBuffer, offset: 0, index: 1)
      enc.setBytes(&params, length: paramSize, index: 2)
      enc.setBuffer(statsBuffer, offset: statsOffset, index: 3)
      enc.setBuffer(statsBuffer, offset: statsOffset, index: 4)
      dispatch(enc, g2pPipeline, count: particles)
    }
    enc.endEncoding()
  }

  private func encodeGatherP2G(
    _ enc: MTLComputeCommandEncoder, params: inout SimParams, nodes: Int, particles: Int
  ) {
    let paramSize = MemoryLayout<SimParams>.stride

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

    enc.setComputePipelineState(scanPipeline)
    enc.setBuffer(cellCountBuffer, offset: 0, index: 0)
    enc.setBuffer(cellStartBuffer, offset: 0, index: 1)
    enc.setBytes(&params, length: paramSize, index: 2)
    enc.dispatchThreadgroups(
      MTLSize(width: 1, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: scanThreads, height: 1, depth: 1))

    enc.setComputePipelineState(scatterIndicesPipeline)
    enc.setBuffer(particleBuffer, offset: 0, index: 0)
    enc.setBuffer(rankBuffer, offset: 0, index: 1)
    enc.setBuffer(cellStartBuffer, offset: 0, index: 2)
    enc.setBuffer(sortedBuffer, offset: 0, index: 3)
    enc.setBytes(&params, length: paramSize, index: 4)
    dispatch(enc, scatterIndicesPipeline, count: particles)

    enc.setComputePipelineState(gatherMassPipeline)
    enc.setBuffer(particleBuffer, offset: 0, index: 0)
    enc.setBuffer(cellStartBuffer, offset: 0, index: 1)
    enc.setBuffer(cellCountBuffer, offset: 0, index: 2)
    enc.setBuffer(sortedBuffer, offset: 0, index: 3)
    enc.setBuffer(gridBuffer, offset: 0, index: 4)
    enc.setBytes(&params, length: paramSize, index: 5)
    dispatch(enc, gatherMassPipeline, count: nodes)

    enc.setComputePipelineState(stressPipeline)
    enc.setBuffer(particleBuffer, offset: 0, index: 0)
    enc.setBuffer(gridBuffer, offset: 0, index: 1)
    enc.setBuffer(termBuffer, offset: 0, index: 2)
    enc.setBytes(&params, length: paramSize, index: 3)
    dispatch(enc, stressPipeline, count: particles)

    enc.setComputePipelineState(gatherStressPipeline)
    enc.setBuffer(particleBuffer, offset: 0, index: 0)
    enc.setBuffer(cellStartBuffer, offset: 0, index: 1)
    enc.setBuffer(cellCountBuffer, offset: 0, index: 2)
    enc.setBuffer(sortedBuffer, offset: 0, index: 3)
    enc.setBuffer(gridBuffer, offset: 0, index: 4)
    enc.setBytes(&params, length: paramSize, index: 5)
    enc.setBuffer(termBuffer, offset: 0, index: 6)
    dispatch(enc, gatherStressPipeline, count: nodes)
  }

  private func dispatch(
    _ enc: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState, count: Int
  ) {
    let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
    enc.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
  }
}
