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
  private(set) var simMilliseconds: Double = 0
  /// Smoothed GPU cost of one simulation substep.
  private var msPerStep: Double = 0.4
  private var targetFPS = 120
  private var stepCap = maxStepsPerFrame
  /// Resting RMS particle speed floors measured 0.04-0.065 (interior reshuffling); visible
  /// sloshing is 0.15+.
  static let sleepSpeed: Float = 0.085
  static let sleepDelay: Double = 2.0
  static let wakeGravityDelta: Float = 0.03
  private(set) var asleep = false
  private var calmSince: Double = 0
  private var calmGravity = SIMD3<Float>(0, -1, 0)
  private var sleepGravity = SIMD3<Float>(0, -1, 0)
  private var recentRMSSpeed: Float = 1
  /// Bulk motion of the liquid (centre of mass, cells per real second). Particle jitter averages
  /// out here, so slowly creeping honey still counts as moving.
  private var centroidDrift: Float = 1
  private var centroidHistory: [(time: Double, centroid: SIMD3<Float>)] = []
  /// Resting water drifts 0.03-0.047 (residual sway); creeping honey measures 0.2+.
  static let sleepDrift: Float = 0.08
  private var activity: Float = 1
  private var ripplePhase: Float = 0
  private var lastEye = SIMD3<Float>.zero
  private var slotSteps = [Int](repeating: 0, count: maxFramesInFlight)
  private var preferredFPS = 0
  private var capCheckTime: Double = 0
  private var lastStatsWrite: Double = 0
  private var screenshotTaken = false
  private var lastSteps = 0
  private var faceHistory: [[String: Any]] = []
  private var trace: [[String: Any]] = []
  private var lastTraceTime: Double = 0

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
      wake()
    }
    guard let sim = simulation else { return }
    if appliedPreset != settings.presetID {
      appliedPreset = settings.presetID
      sim.preset = settings.preset
      audio.setPreset(settings.preset)
      wake()
    }
    if appliedFill != settings.fill {
      appliedFill = settings.fill
      sim.setFill(settings.fill, up: -lastGravity)
      wake()
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
      quality: settings.quality, cellOverrideMM: launch.cellMM)
    do {
      let sim = try FluidSimulation(device: device, library: library, geometry: geometry)
      sim.preset = settings.preset
      audio.setPreset(settings.preset)
      sim.useGather = !launch.scatterP2G
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
    wake()
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
    consumeStats(stats, steps: slotSteps[slot], time: now)

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
    let stepsPerSecond = sim.stepsPerSecond
    stepAccumulator += stepsPerSecond * Float(dt)
    var steps = Int(stepAccumulator)
    stepAccumulator -= Float(steps)
    steps = min(
      steps,
      launch.maxSteps ?? affordableSteps(stepsPerSecond: stepsPerSecond, view: view, now: now))
    updateSleep(gravity: sample.gravity, shook: sample.shook, now: now)
    if asleep { steps = 0 }
    view.preferredFramesPerSecond = displayFPS()
    updateActivity(dt: Float(dt))
    lastSteps = steps
    slotSteps[slot] = steps
    sim.touches = touches.inputs(
      screenSize: screenPoints, geometry: sim.geometry,
      simTimePerSecond: stepsPerSecond * sim.dt,
      extraSplash: shakeBoost * 2)
    if shakeBoost > 0 { addShakeSplashes(sim) }
    shakeBoost = 0

    guard let simBuffer = queue.makeCommandBuffer(), let commandBuffer = queue.makeCommandBuffer()
    else {
      frameSemaphore.signal()
      return
    }
    simBuffer.label = "Simulation"
    sim.encode(
      into: simBuffer, steps: steps, gravity: sample.gravity,
      statsBuffer: statsBuffer, statsOffset: slot * MemoryLayout<FrameStats>.stride)
    simBuffer.addCompletedHandler { [weak self, steps] cb in
      let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
      Task { @MainActor in self?.recordSimTime(ms: ms, steps: steps) }
    }
    simBuffer.commit()
    commandBuffer.label = "Render"

    guard let drawable = view.currentDrawable else {
      commandBuffer.addCompletedHandler { [frameSemaphore] _ in frameSemaphore.signal() }
      commandBuffer.commit()
      return
    }
    let uniforms = makeUniforms(sim: sim, viewport: view.drawableSize, time: now - startTime)
    fluidRenderer.encode(
      commandBuffer: commandBuffer, target: drawable.texture, uniforms: uniforms,
      particles: sim.particleBuffer, foam: sim.foamBuffer, density: sim.densityBuffer,
      particleCount: sim.activeCount)

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

  private func recordSimTime(ms: Double, steps: Int) {
    simMilliseconds = ms
    guard steps > 0, ms > 0 else { return }
    msPerStep += (ms / Double(steps) - msPerStep) * 0.1
  }

  /// Keeps the frame rate up. Low quality in tilt mode starts at 120 Hz; head tracking (ARKit
  /// delivers poses at 60 Hz) and finer qualities run at 60 Hz. Missing the target at 120 Hz
  /// drops to 60 Hz with the simulation still at full speed; missing it at 60 Hz lowers the
  /// substep cap, so the liquid runs in slow motion instead of the display stuttering.
  private func affordableSteps(stepsPerSecond: Float, view: MTKView, now: Double) -> Int {
    let hot =
      ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue
    let preferred =
      settings.quality == .low && settings.perspectiveMode == .tilt && !hot
        && !ProcessInfo.processInfo.isLowPowerModeEnabled ? 120 : 60
    if preferred != preferredFPS {
      preferredFPS = preferred
      targetFPS = preferred
      stepCap = Self.maxStepsPerFrame * 120 / targetFPS
      capCheckTime = now
      frameTimes.removeAll()
    }
    let maxCap = Self.maxStepsPerFrame * 120 / targetFPS
    let thermal = ProcessInfo.processInfo.thermalState
    if thermal == .critical { return 0 }
    if !asleep, now - capCheckTime > 0.5, frameTimes.count >= 30 {
      capCheckTime = now
      let recent = frameTimes.suffix(30)
      let achieved = Double(recent.count) / recent.reduce(0, +)
      if achieved < Double(targetFPS) * 0.9 {
        if targetFPS == 120 {
          targetFPS = 60
          stepCap = Self.maxStepsPerFrame * 2
          frameTimes.removeAll()
        } else {
          stepCap = max(2, stepCap * 4 / 5)
        }
      } else if achieved > Double(targetFPS) * 0.97, lastSteps >= stepCap {
        stepCap = min(maxCap, stepCap + 1)
      }
    }
    // A hot phone runs the liquid at half speed rather than heating further.
    return hot ? max(2, min(stepCap, maxCap / 2)) : stepCap
  }

  // MARK: - Power

  /// Smoothed 0...1 motion level for animated surface effects (ripples, caustics), so a still
  /// liquid looks still.
  private func updateActivity(dt: Float) {
    let target: Float =
      asleep
      ? 0
      : max(
        min(max((recentRMSSpeed - 0.06) / 0.2, 0), 1), min(centroidDrift / 0.5, 1))
    activity += (target - activity) * (1 - exp(-dt / 0.8))
    ripplePhase += dt * activity
  }

  /// Settled liquid with no input stops simulating and the display drops to 30 Hz; tilting,
  /// touching or shaking wakes it immediately.
  private func updateSleep(gravity: SIMD3<Float>, shook: Bool, now: Double) {
    let touching = touches.isActive
    if asleep {
      if shook || touching || simd_distance(gravity, sleepGravity) > Self.wakeGravityDelta {
        wake(now: now)
      }
      return
    }
    if touching || shook || recentRMSSpeed > Self.sleepSpeed || centroidDrift > Self.sleepDrift
      || simd_distance(gravity, calmGravity) > Self.wakeGravityDelta
    {
      calmSince = now
      calmGravity = gravity
      return
    }
    if now - calmSince > Self.sleepDelay {
      asleep = true
      sleepGravity = gravity
    }
  }

  func wake(now: Double = CACurrentMediaTime()) {
    asleep = false
    centroidDrift = 1
    centroidHistory.removeAll()
    calmSince = now
    frameTimes.removeAll()
  }

  private func displayFPS() -> Int {
    guard asleep else { return targetFPS }
    // Head tracking keeps the parallax smooth even while the liquid sleeps.
    let eyeMoving = simd_distance(perspective.eye, lastEye) > 0.4
    lastEye = perspective.eye
    return eyeMoving && settings.perspectiveMode == .face ? 60 : 30
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
    defer { recordTrace() }
    let attitude = motion.hasAttitude ? motion.attitude : nil
    if settings.perspectiveMode == .face, let eye = faceTracker?.snapshot().eye {
      perspective.updateFace(faceEye: eye, attitude: attitude, dt: dt, gain: gain)
    } else if settings.perspectiveMode == .face, let attitude,
      perspective.updateFaceLost(attitude: attitude, dt: dt, gain: gain)
    {
      // Head predicted from the device rotation since the face was last seen.
    } else if motion.hasAttitude {
      perspective.updateTilt(attitude: motion.attitude, dt: dt, gain: gain * 1.4)
    }
  }

  /// 10 Hz ring buffer of the perspective pipeline, written with stats.json (diagnostics).
  private func recordTrace() {
    guard launch.writeStats else { return }
    let now = CACurrentMediaTime()
    guard now - lastTraceTime >= 0.1 else { return }
    lastTraceTime = now
    let face = faceTracker?.snapshot()
    let q = motion.attitude.vector
    func r(_ v: Float) -> Double { (Double(v) * 10).rounded() / 10 }
    var entry: [String: Any] = [
      "t": ((now - startTime) * 100).rounded() / 100,
      "tracked": face?.eye != nil,
      "eye": [r(perspective.eye.x), r(perspective.eye.y), r(perspective.eye.z)],
      "att": [q.x, q.y, q.z, q.w].map { (Double($0) * 1000).rounded() / 1000 },
      "asleep": asleep,
      "hasAtt": motion.hasAttitude,
      "rms": (Double(recentRMSSpeed) * 1000).rounded() / 1000,
      "drift": (Double(centroidDrift) * 1000).rounded() / 1000,
      "g": [lastGravity.x, lastGravity.y, lastGravity.z].map {
        (Double($0) * 1000).rounded() / 1000
      },
    ]
    if let e = face?.eye { entry["face"] = [r(e.x), r(e.y), r(e.z)] }
    if let f = face { entry["arT"] = (f.timestamp * 100).rounded() / 100 }
    trace.append(entry)
    if trace.count > 400 { trace.removeFirst(trace.count - 400) }
  }

  private func consumeStats(_ stats: FrameStats, steps: Int, time: Double) {
    guard let sim = simulation, sim.activeCount > 0, steps > 0 else { return }
    let n = Float(sim.activeCount * steps)
    let kinetic = stats.kineticEnergy / n
    let impact = stats.impactEnergy / n
    recentRMSSpeed = sqrt(max(kinetic, 0))
    let centroid =
      SIMD3<Float>(stats.positionSum.x, stats.positionSum.y, stats.positionSum.z)
      / Float(sim.activeCount)
    centroidHistory.append((time, centroid))
    centroidHistory.removeAll { time - $0.time > 1.5 }
    if let oldest = centroidHistory.first, time - oldest.time > 0.5 {
      centroidDrift = simd_distance(centroid, oldest.centroid) / Float(time - oldest.time)
    }
    haptics.update(impact: impact, time: time)
    audio.update(kinetic: kinetic, impact: impact, time: time)
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
    u.activity = activity
    u.restDensity = preset.restDensity
    u.opaqueSoftness = min(max(preset.surfaceSmoothing - 1, 0), 1)
    u.ripplePhase = ripplePhase
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

  private func faceDiagnosticsPayload() -> [String: Any] {
    guard let tracker = faceTracker, tracker.running else { return [:] }
    let d = tracker.takeDiagnostics()
    return [
      "frames": d.frames, "tracked": d.trackedFrames, "maxFrameGap": d.maxFrameGap,
      "maxTrackedGap": d.maxTrackedGap, "format": d.videoFormat,
    ]
  }

  private func writeStats() {
    let face = faceDiagnosticsPayload()
    if !face.isEmpty {
      faceHistory.append(face)
      if faceHistory.count > 40 { faceHistory.removeFirst() }
    }
    let payload: [String: Any] = [
      "fps": fps,
      "renderMs": gpuMilliseconds,
      "simMs": simMilliseconds,
      "gather": simulation?.useGather ?? false,
      "attitude": motion.debugAttitude,
      "reference": perspective.debugReference,
      "hasAttitude": motion.hasAttitude,
      "mode": settings.perspectiveMode.rawValue,
      "faceHistory": faceHistory,
      "trace": trace,
      "thermal": ProcessInfo.processInfo.thermalState.rawValue,
      "particles": simulation?.activeCount ?? 0,
      "stepsPerFrame": lastSteps,
      "msPerStep": msPerStep,
      "cellMM": simulation?.geometry.cellMM ?? 0,
      "targetFPS": targetFPS,
      "asleep": asleep,
      "rmsSpeed": recentRMSSpeed,
      "centroidDrift": centroidDrift,
      "activity": activity,
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
