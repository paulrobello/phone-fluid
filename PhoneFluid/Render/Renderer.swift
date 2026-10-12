import MetalKit
import UIKit
import simd

/// Owns the frame loop: input → simulation → rendering → feedback.
@MainActor
final class Renderer: NSObject {
  static let maxFramesInFlight = 3
  static let maxStepsPerFrame = 14

  let device: MTLDevice
  private let queue: MTLCommandQueue
  private let library: MTLLibrary
  private let fluidRenderer: FluidRenderer
  private let settings: AppSettings
  private let launch: LaunchConfig

  private(set) var simulation: FluidSimulation?
  let motion: MotionController
  let perspective = PerspectiveController()
  let light = LightEstimator()
  let touches = TouchController()
  let haptics = HapticsController()
  let audio = SloshAudio()
  private var faceTracker: ARFaceTracker?

  private let statsBuffer: MTLBuffer
  private let frameSemaphore = DispatchSemaphore(value: maxFramesInFlight)
  private var frameIndex = 0
  private var stepAccumulator: Float = 0
  private var lastTime: Double = 0
  private let startTime = CACurrentMediaTime()
  private var screenPoints: CGSize = .zero
  private var lastGravity = SIMD3<Float>(0, -1, 0)
  private var shakeBoost: Float = 0

  // Applied settings (compared each frame to pick up changes).
  private var appliedQuality: SimQuality?
  private var appliedPreset: String?
  private var appliedFill: Float?
  private var appliedMode: PerspectiveMode?
  private var appliedPhotoVersion = -1
  private var appliedRecenter = 0

  // Diagnostics.
  private var frameTimes: [Double] = []
  private(set) var gpuMilliseconds: Double = 0
  private var lastStatsWrite: Double = 0
  private var screenshotTaken = false
  private var lastSteps = 0

  init(device: MTLDevice, settings: AppSettings, launch: LaunchConfig = .current) throws {
    guard let queue = device.makeCommandQueue(), let library = device.makeDefaultLibrary(),
      let stats = device.makeBuffer(
        length: MemoryLayout<FrameStats>.stride * Self.maxFramesInFlight,
        options: .storageModeShared)
    else { throw SimulationError.allocationFailed }
    self.device = device
    self.queue = queue
    self.library = library
    self.settings = settings
    self.launch = launch
    self.statsBuffer = stats
    self.motion = MotionController(launch: launch)
    self.fluidRenderer = try FluidRenderer(
      device: device, library: library, colorFormat: .bgra8Unorm)
    super.init()
    memset(stats.contents(), 0, stats.length)
    motion.start()
  }

  // MARK: - Settings

  private func applySettings() {
    guard screenPoints.width > 0 else { return }
    if appliedQuality != settings.quality || simulation == nil {
      rebuildSimulation()
    }
    guard let sim = simulation else { return }
    if appliedPreset != settings.presetID {
      appliedPreset = settings.presetID
      sim.preset = settings.preset
    }
    if appliedFill != settings.fill {
      appliedFill = settings.fill
      sim.setFill(settings.fill, up: -lastGravity)
    }
    if appliedMode != settings.perspectiveMode {
      appliedMode = settings.perspectiveMode
      if settings.perspectiveMode == .face, ARFaceTracker.isSupported {
        if faceTracker == nil { faceTracker = ARFaceTracker() }
        faceTracker?.start()
      } else {
        faceTracker?.stop()
        perspective.clearFace()
      }
      perspective.recenter()
    }
    if appliedPhotoVersion != settings.photoVersion {
      appliedPhotoVersion = settings.photoVersion
      fluidRenderer.setPhoto(UIImage(contentsOfFile: AppSettings.photoURL.path))
    }
    if appliedRecenter != settings.recenterRequest {
      appliedRecenter = settings.recenterRequest
      perspective.recenter()
    }
    haptics.enabled = settings.hapticsEnabled
    if audio.enabled != settings.soundEnabled { audio.enabled = settings.soundEnabled }
  }

  private func rebuildSimulation() {
    let geometry = TankGeometry(
      screenPoints: SIMD2<Float>(Float(screenPoints.width), Float(screenPoints.height)),
      quality: settings.quality)
    do {
      let sim = try FluidSimulation(device: device, library: library, geometry: geometry)
      sim.preset = settings.preset
      sim.reset(fill: settings.fill, down: lastGravity)
      simulation = sim
      appliedQuality = settings.quality
      appliedPreset = settings.presetID
      appliedFill = settings.fill
    } catch {
      simulation = nil
    }
  }

  func resetFluid() {
    simulation?.reset(fill: settings.fill, down: lastGravity)
  }

  func didBecomeActive() {
    motion.start()
    if settings.soundEnabled { audio.start() }
    if settings.perspectiveMode == .face { faceTracker?.start() }
  }

  func willResignActive() {
    motion.stop()
    audio.stop()
    faceTracker?.stop()
    touches.cancelAll()
  }

  // MARK: - Frame

  func renderFrame(in view: MTKView) {
    let now = CACurrentMediaTime()
    let dt = lastTime == 0 ? 1.0 / 60.0 : min(now - lastTime, 0.1)
    lastTime = now
    screenPoints = view.bounds.size
    applySettings()
    guard let sim = simulation else { return }

    frameSemaphore.wait()
    let slot = frameIndex % Self.maxFramesInFlight
    frameIndex += 1

    // Read the stats this slot accumulated last time it was used, then clear it.
    let statsPtr = statsBuffer.contents().advanced(by: slot * MemoryLayout<FrameStats>.stride)
      .bindMemory(to: FrameStats.self, capacity: 1)
    let stats = statsPtr.pointee
    statsPtr.pointee = FrameStats()
    consumeStats(stats, time: now)

    // Inputs.
    let sample = motion.sample(time: now, dt: dt)
    lastGravity = sample.gravity
    updatePerspective(dt: Float(dt))
    light.update(
      face: settings.perspectiveMode == .face ? faceTracker?.snapshot() : nil,
      screenBrightness: Float(view.window?.screen.brightness ?? 0.6), dt: Float(dt))
    if sample.shook {
      shakeBoost = 1
      haptics.shake()
      audio.splash(strength: 1)
    }

    // Simulation steps for this frame (fixed sim dt; the step rate sets the speed).
    let stepsPerSecond = sim.preset.stepsPerSecond
    stepAccumulator += stepsPerSecond * Float(dt)
    var steps = Int(stepAccumulator)
    stepAccumulator -= Float(steps)
    steps = min(steps, Self.maxStepsPerFrame)
    lastSteps = steps
    sim.touches = touches.inputs(
      screenSize: screenPoints, geometry: sim.geometry,
      simTimePerSecond: stepsPerSecond * FluidSimulation.simDt,
      extraSplash: shakeBoost * 2)
    if shakeBoost > 0 { addShakeSplashes(sim) }
    shakeBoost = 0

    guard let commandBuffer = queue.makeCommandBuffer() else {
      frameSemaphore.signal()
      return
    }
    commandBuffer.label = "Frame"
    sim.encode(
      into: commandBuffer, steps: steps, gravity: sample.gravity,
      statsBuffer: statsBuffer, statsOffset: slot * MemoryLayout<FrameStats>.stride)

    guard let drawable = view.currentDrawable else {
      commandBuffer.addCompletedHandler { [frameSemaphore] _ in frameSemaphore.signal() }
      commandBuffer.commit()
      return
    }
    let uniforms = makeUniforms(sim: sim, viewport: view.drawableSize, time: now - startTime)
    fluidRenderer.encode(
      commandBuffer: commandBuffer, target: drawable.texture, uniforms: uniforms,
      particles: sim.particleBuffer, particleCount: sim.activeCount)

    let wantsScreenshot =
      !screenshotTaken && launch.screenshotAfter.map { now - startTime >= $0 } == true
    if wantsScreenshot { screenshotTaken = true }
    nonisolated(unsafe) let texture = drawable.texture
    commandBuffer.addCompletedHandler { [frameSemaphore, weak self] cb in
      let gpu = (cb.gpuEndTime - cb.gpuStartTime) * 1000
      frameSemaphore.signal()
      if wantsScreenshot { Self.saveScreenshot(texture) }
      Task { @MainActor in self?.gpuMilliseconds = gpu }
    }
    commandBuffer.present(drawable)
    commandBuffer.commit()

    recordFrame(now: now, dt: dt)
  }

  private func addShakeSplashes(_ sim: FluidSimulation) {
    let size = SIMD3<Float>(sim.geometry.gridSize)
    while sim.touches.count < Int(MAX_TOUCHES) {
      let x = Float.random(in: size.x * 0.2...size.x * 0.8)
      let y = Float.random(in: size.y * 0.15...size.y * 0.6)
      sim.touches.append(
        TouchInput(positionRadius: [x, y, size.x * 0.3, 0], velocity: [0, 0, 1.8, 0]))
    }
  }

  private func updatePerspective(dt: Float) {
    let gain = settings.parallax
    if settings.perspectiveMode == .face, let face = faceTracker?.snapshot(),
      let eye = face.eye
    {
      perspective.updateFace(faceEye: eye, dt: dt, gain: gain)
    } else if motion.hasAttitude {
      perspective.updateTilt(attitude: motion.attitude, dt: dt, gain: gain * 1.4)
    }
  }

  private func consumeStats(_ stats: FrameStats, time: Double) {
    guard let sim = simulation, sim.activeCount > 0, lastSteps > 0 else { return }
    let n = Float(sim.activeCount * lastSteps)
    let kinetic = stats.kineticEnergy / n
    let impact = stats.impactEnergy / n
    haptics.update(impact: impact, time: time)
    audio.update(kinetic: kinetic, impact: impact)
  }

  private func makeUniforms(sim: FluidSimulation, viewport: CGSize, time: Double) -> RenderUniforms
  {
    let geo = sim.geometry
    let eye = perspective.eye
    let mm = TankGeometry.mmPerPoint
    let halfW = Float(screenPoints.width) * mm / 2
    let halfH = Float(screenPoints.height) * mm / 2
    let view = OffAxisProjection.view(eye: eye)
    let proj = OffAxisProjection.projection(
      eye: eye, screenMin: [-halfW, -halfH], screenMax: [halfW, halfH],
      near: eye.z * 0.5, far: eye.z + geo.depthMM + 200)
    let l = light.current
    let preset = sim.preset

    var u = RenderUniforms()
    u.viewMatrix = view
    u.projectionMatrix = proj
    u.inverseProjection = proj.inverse
    u.eyeWorld = SIMD4<Float>(eye, 1)
    u.boxMin = SIMD4<Float>(geo.boxMin, 0)
    u.boxMax = SIMD4<Float>(geo.boxMax, 0)
    u.lightDirWorld = SIMD4<Float>(l.direction, 0)
    u.lightDirView = SIMD4<Float>(l.direction, 0)
    u.lightColor = SIMD4<Float>(l.color, 1)
    u.ambientColor = SIMD4<Float>(l.ambient, 1)
    u.absorption = SIMD4<Float>(preset.absorption, preset.refraction)
    u.tint = SIMD4<Float>(preset.tint, preset.opacity)
    u.emission = SIMD4<Float>(preset.emission, preset.emissionStrength)
    u.material = SIMD4<Float>(preset.reflectivity, preset.metallic, preset.roughness, preset.foam)
    u.viewportSize = SIMD2<Float>(Float(viewport.width), Float(viewport.height))
    u.cellMM = geo.cellMM
    u.particleRadiusMM = geo.cellMM * 0.75
    u.time = Float(time)
    u.backdrop = settings.backdrop.shaderIndex
    u.foamSpeed = preset.foamSpeed
    u.causticStrength = preset.causticStrength
    u.simOrigin = SIMD4<Float>(geo.simOrigin, 0)
    return u
  }

  // MARK: - Diagnostics

  private func recordFrame(now: Double, dt: Double) {
    frameTimes.append(dt)
    if frameTimes.count > 120 { frameTimes.removeFirst(frameTimes.count - 120) }
    if launch.writeStats, now - lastStatsWrite > 1 {
      lastStatsWrite = now
      writeStats()
    }
    if let exitAfter = launch.exitAfter, now - startTime > exitAfter {
      if launch.writeStats { writeStats() }
      exit(0)
    }
  }

  var fps: Double {
    guard !frameTimes.isEmpty else { return 0 }
    return Double(frameTimes.count) / frameTimes.reduce(0, +)
  }

  private func writeStats() {
    let payload: [String: Any] = [
      "fps": fps,
      "gpuMs": gpuMilliseconds,
      "particles": simulation?.activeCount ?? 0,
      "stepsPerFrame": lastSteps,
      "quality": settings.quality.rawValue,
      "preset": settings.presetID,
      "grid": simulation.map { g in
        let s = g.geometry.gridSize
        return [Int(s.x), Int(s.y), Int(s.z)]
      } ?? [],
      "eye": [perspective.eye.x, perspective.eye.y, perspective.eye.z],
      "gravity": [lastGravity.x, lastGravity.y, lastGravity.z],
    ]
    if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted]) {
      try? data.write(to: LaunchConfig.documentsURL.appendingPathComponent("stats.json"))
    }
  }

  private nonisolated static func saveScreenshot(_ texture: MTLTexture) {
    let w = texture.width
    let h = texture.height
    var bytes = [UInt8](repeating: 0, count: w * h * 4)
    texture.getBytes(
      &bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    let info = CGBitmapInfo(
      rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
    guard let provider = CGDataProvider(data: Data(bytes) as CFData),
      let image = CGImage(
        width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info, provider: provider,
        decode: nil, shouldInterpolate: false, intent: .defaultIntent),
      let png = UIImage(cgImage: image).pngData()
    else { return }
    try? png.write(to: LaunchConfig.documentsURL.appendingPathComponent("screenshot.png"))
  }
}

extension Renderer: MTKViewDelegate {
  nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

  nonisolated func draw(in view: MTKView) {
    MainActor.assumeIsolated { self.renderFrame(in: view) }
  }
}
