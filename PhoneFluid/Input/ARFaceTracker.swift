import ARKit
import Synchronization
import simd

/// Latest head pose and light estimate from the front TrueDepth camera.
struct FaceState: Sendable {
  /// Eye midpoint in the device frame (mm, origin at screen centre, +z out of the screen).
  var eye: SIMD3<Float>?
  var timestamp: Double = 0
  /// Ambient intensity in lumens (1000 = neutral), colour temperature in kelvin.
  var ambientIntensity: Float?
  var colorTemperature: Float?
  /// Direction toward the primary light, device frame.
  var lightDirection: SIMD3<Float>?
}

/// Rolling counters for diagnosing tracking stalls (exposed through stats.json).
struct FaceDiagnostics: Sendable {
  var frames = 0
  var trackedFrames = 0
  var maxFrameGap: Double = 0
  var maxTrackedGap: Double = 0
  var lastFrame: Double = 0
  var lastTracked: Double = 0
  var videoFormat = ""
}

/// Runs ARKit face tracking on its own delegate queue and publishes `FaceState` via a mutex.
final class ARFaceTracker: NSObject, ARSessionDelegate, @unchecked Sendable {
  static var isSupported: Bool { ARFaceTrackingConfiguration.isSupported }

  /// Front camera position relative to the screen centre (mm, device frame). The TrueDepth
  /// module sits in the Dynamic Island near the top edge.
  static let cameraOffset = SIMD3<Float>(0, 72, 0)

  private let session = ARSession()
  private let queue = DispatchQueue(label: "phonefluid.arkit")
  private let state = Mutex(FaceState())
  private let diagnostics = Mutex(FaceDiagnostics())
  private(set) var running = false

  override init() {
    super.init()
    session.delegate = self
    session.delegateQueue = queue
  }

  func start() {
    guard Self.isSupported, !running else { return }
    let config = ARFaceTrackingConfiguration()
    config.isLightEstimationEnabled = true
    config.maximumNumberOfTrackedFaces = 1
    // The lightest 60 fps format leaves the GPU and camera pipeline headroom for the simulation.
    if let format = ARFaceTrackingConfiguration.supportedVideoFormats
      .filter({ $0.framesPerSecond >= 60 })
      .min(by: { $0.imageResolution.width < $1.imageResolution.width })
    {
      config.videoFormat = format
    }
    let format = config.videoFormat
    diagnostics.withLock {
      $0 = FaceDiagnostics()
      $0.videoFormat =
        "\(Int(format.imageResolution.width))x\(Int(format.imageResolution.height))@\(format.framesPerSecond)"
    }
    session.run(config, options: [.resetTracking, .removeExistingAnchors])
    running = true
  }

  func stop() {
    guard running else { return }
    session.pause()
    running = false
    state.withLock { $0 = FaceState() }
  }

  func snapshot() -> FaceState { state.withLock { $0 } }

  /// Returns and resets the rolling counters.
  func takeDiagnostics() -> FaceDiagnostics {
    diagnostics.withLock { d in
      let out = d
      d.frames = 0
      d.trackedFrames = 0
      d.maxFrameGap = 0
      d.maxTrackedGap = 0
      return out
    }
  }

  func session(_ session: ARSession, didUpdate frame: ARFrame) {
    let view = frame.camera.viewMatrix(for: .portrait)
    var update = FaceState(timestamp: frame.timestamp)
    if let face = frame.anchors.compactMap({ $0 as? ARFaceAnchor }).first, face.isTracked {
      // Midpoint between the eyes, in face space, then into portrait camera view space.
      let l = face.leftEyeTransform.columns.3
      let r = face.rightEyeTransform.columns.3
      let midFace = SIMD4<Float>((l.x + r.x) / 2, (l.y + r.y) / 2, (l.z + r.z) / 2, 1)
      let world = face.transform * midFace
      let v = view * world
      update.eye = Self.deviceFromCameraView(SIMD3<Float>(v.x, v.y, v.z))
    }
    if let estimate = frame.lightEstimate {
      update.ambientIntensity = Float(estimate.ambientIntensity)
      update.colorTemperature = Float(estimate.ambientColorTemperature)
      if let directional = estimate as? ARDirectionalLightEstimate {
        let d = directional.primaryLightDirection
        // primaryLightDirection points from the light toward the scene.
        let toLight = view * SIMD4<Float>(-d.x, -d.y, -d.z, 0)
        let dev = SIMD3<Float>(toLight.x, toLight.y, -toLight.z)
        if simd_length(dev) > 1e-3 { update.lightDirection = simd_normalize(dev) }
      }
    }
    let tracked = update.eye != nil
    let now = frame.timestamp
    diagnostics.withLock { d in
      d.frames += 1
      if d.lastFrame > 0 { d.maxFrameGap = max(d.maxFrameGap, now - d.lastFrame) }
      d.lastFrame = now
      if tracked {
        d.trackedFrames += 1
        if d.lastTracked > 0 { d.maxTrackedGap = max(d.maxTrackedGap, now - d.lastTracked) }
        d.lastTracked = now
      }
    }
    state.withLock { current in
      let keepEye = update.eye == nil && frame.timestamp - current.timestamp < 0.05
      let previousEye = current.eye
      current = update
      if keepEye { current.eye = previousEye }
    }
  }

  /// Portrait camera view space (metres, camera looks down -z, the user is in front) → device
  /// frame (mm, origin at the screen centre). The front camera faces the user, so "in front of
  /// the camera" is +z in the device frame; x keeps its sign (verified on device: negating it
  /// mirrored the parallax).
  static func deviceFromCameraView(_ v: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3<Float>(v.x, v.y, -v.z) * 1000 + cameraOffset
  }
}
