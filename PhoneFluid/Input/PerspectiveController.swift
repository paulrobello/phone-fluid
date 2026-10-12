import simd

/// Computes the viewer's eye position (device frame, mm) for the off-axis camera.
///
/// Tilt mode assumes the head stays put while the phone rotates: the eye is the calibrated
/// straight-ahead point rotated into the current device frame. The reference attitude drifts
/// toward the current one so the view re-centres when the phone is held still.
@MainActor
final class PerspectiveController {
  static let eyeDistance: Float = 320
  static let recenterRate: Float = 0.4
  static let maxOffset: Float = 140

  private var reference: simd_quatf?
  private(set) var eye = SIMD3<Float>(0, 0, eyeDistance)
  private var smoothedFaceEye: SIMD3<Float>?

  func recenter() { reference = nil }

  var debugReference: [Float] {
    guard let r = reference else { return [] }
    return [r.vector.x, r.vector.y, r.vector.z, r.vector.w]
  }

  func updateTilt(attitude: simd_quatf, dt: Float, gain: Float) {
    guard let ref = reference else {
      reference = attitude
      eye = [0, 0, Self.eyeDistance]
      return
    }
    let k = 1 - exp(-Self.recenterRate * dt)
    let newRef = simd_slerp(ref, attitude, k)
    reference = newRef
    eye = Self.tiltEye(reference: newRef, current: attitude, gain: gain)
  }

  /// Head position in the current device frame for a head fixed along the reference +z.
  static func tiltEye(reference: simd_quatf, current: simd_quatf, gain: Float) -> SIMD3<Float> {
    let relative = current.inverse * reference
    let raw = relative.act(SIMD3<Float>(0, 0, eyeDistance))
    var offset = SIMD2<Float>(raw.x, raw.y) * gain
    let len = simd_length(offset)
    if len > maxOffset { offset *= maxOffset / len }
    let z = max(sqrt(max(eyeDistance * eyeDistance - simd_length_squared(offset), 0)), 120)
    return [offset.x, offset.y, z]
  }

  /// Face mode: `faceEye` is the measured eye midpoint in the device frame (mm).
  func updateFace(faceEye: SIMD3<Float>, dt: Float, gain: Float) {
    let target = SIMD3<Float>(faceEye.x * gain, faceEye.y * gain, max(faceEye.z, 120))
    let k = 1 - exp(-18 * dt)
    let smoothed = smoothedFaceEye.map { $0 + (target - $0) * k } ?? target
    smoothedFaceEye = smoothed
    eye = smoothed
  }

  func clearFace() { smoothedFaceEye = nil }
}
