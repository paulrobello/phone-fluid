import CoreMotion
import simd

/// Polls Core Motion once per frame. Provides the effective gravity felt by the liquid, the
/// device attitude for tilt parallax, and shake detection.
@MainActor
final class MotionController {
  private let manager = CMMotionManager()
  private let override: SIMD3<Float>?
  private let sweep: Bool
  private var sweepTime: Double = 0
  private var lastShake: Double = 0

  /// Most recent attitude (device → reference frame), identity when unavailable.
  private(set) var attitude = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
  private(set) var hasAttitude = false

  static let shakeThreshold: Float = 1.8

  var debugAttitude: [Float] {
    [attitude.vector.x, attitude.vector.y, attitude.vector.z, attitude.vector.w]
  }

  init(launch: LaunchConfig = .current) {
    override = launch.gravity
    sweep = launch.gravitySweep
  }

  func start() {
    guard override == nil, !sweep, manager.isDeviceMotionAvailable else { return }
    manager.deviceMotionUpdateInterval = 1.0 / 120.0
    manager.startDeviceMotionUpdates(using: .xArbitraryZVertical)
  }

  func stop() { manager.stopDeviceMotionUpdates() }

  struct Sample {
    /// Effective gravity in g, device frame (+x right, +y up the screen, +z out of the screen).
    var gravity: SIMD3<Float>
    /// Magnitude of user acceleration in g.
    var jolt: Float
    var shook: Bool
  }

  func sample(time: Double, dt: Double) -> Sample {
    if let override {
      return Sample(gravity: override, jolt: 0, shook: false)
    }
    if sweep {
      sweepTime += dt
      let a = Float(sweepTime * 0.6)
      let g = SIMD3<Float>(sin(a) * 0.8, -cos(a * 0.7), -0.35)
      return Sample(gravity: simd_normalize(g), jolt: 0, shook: false)
    }
    guard let motion = manager.deviceMotion else {
      return Sample(gravity: [0, -1, 0], jolt: 0, shook: false)
    }
    let q = motion.attitude.quaternion
    attitude = simd_quatf(ix: Float(q.x), iy: Float(q.y), iz: Float(q.z), r: Float(q.w))
    hasAttitude = true
    let gravity = SIMD3<Float>(
      Float(motion.gravity.x), Float(motion.gravity.y), Float(motion.gravity.z))
    let user = SIMD3<Float>(
      Float(motion.userAcceleration.x), Float(motion.userAcceleration.y),
      Float(motion.userAcceleration.z))
    let jolt = simd_length(user)
    var shook = false
    if jolt > Self.shakeThreshold, time - lastShake > 0.35 {
      lastShake = time
      shook = true
    }
    return Sample(
      gravity: Self.effectiveGravity(gravity: gravity, user: user), jolt: jolt, shook: shook)
  }

  /// The liquid feels gravity minus the container's own acceleration.
  nonisolated static func effectiveGravity(gravity: SIMD3<Float>, user: SIMD3<Float>) -> SIMD3<
    Float
  > {
    let g = gravity - user
    let m = simd_length(g)
    return m > 3 ? g * (3 / m) : g
  }
}
