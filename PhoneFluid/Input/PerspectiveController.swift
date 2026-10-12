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
  private var lastFace: (eye: SIMD3<Float>, attitude: simd_quatf)?

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

  /// Face mode: `faceEye` is the measured eye midpoint in the device frame (mm). The device
  /// attitude at that moment is remembered so the head can be predicted when tracking drops.
  func updateFace(faceEye: SIMD3<Float>, attitude: simd_quatf?, dt: Float, gain: Float) {
    if let attitude { lastFace = (faceEye, attitude) }
    moveEye(toward: faceEye, dt: dt, gain: gain)
  }

  /// Face lost (typically the phone tilted the face out of the camera's view): assume the head
  /// stayed where it was last seen and rotate it into the current device frame, so the parallax
  /// keeps following the tilt until tracking resumes. Returns false with no face history.
  func updateFaceLost(attitude: simd_quatf, dt: Float, gain: Float) -> Bool {
    guard let last = lastFace else { return false }
    let predicted = Self.predictedEye(
      lastEye: last.eye, lastAttitude: last.attitude, current: attitude)
    moveEye(toward: predicted, dt: dt, gain: gain)
    return true
  }

  /// A world-fixed head seen at `lastEye` (device frame) when the device had `lastAttitude`.
  static func predictedEye(
    lastEye: SIMD3<Float>, lastAttitude: simd_quatf, current: simd_quatf
  ) -> SIMD3<Float> {
    (current.inverse * lastAttitude).act(lastEye)
  }

  private func moveEye(toward faceEye: SIMD3<Float>, dt: Float, gain: Float) {
    var offset = SIMD2<Float>(faceEye.x, faceEye.y) * gain
    let len = simd_length(offset)
    if len > Self.maxOffset * 1.5 { offset *= Self.maxOffset * 1.5 / len }
    let target = SIMD3<Float>(offset.x, offset.y, max(faceEye.z, 120))
    let k = 1 - exp(-18 * dt)
    let smoothed = smoothedFaceEye.map { $0 + (target - $0) * k } ?? target
    smoothedFaceEye = smoothed
    eye = smoothed
  }

  func clearFace() {
    smoothedFaceEye = nil
    lastFace = nil
  }
}
