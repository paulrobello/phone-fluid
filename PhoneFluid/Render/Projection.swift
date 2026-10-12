import simd

/// Off-axis (asymmetric frustum) camera for a physical screen rectangle in the z = 0 plane.
///
/// Kooima's generalized perspective projection, specialised to an axis-aligned screen: the view
/// matrix is a pure translation by `-eye`, and the frustum is the pyramid from the eye through the
/// screen's edges. Metal clip space (z in 0...1, right-handed view looking down -z).
enum OffAxisProjection {
  static func view(eye: SIMD3<Float>) -> simd_float4x4 {
    var m = matrix_identity_float4x4
    m.columns.3 = SIMD4<Float>(-eye, 1)
    return m
  }

  static func projection(
    eye: SIMD3<Float>, screenMin: SIMD2<Float>, screenMax: SIMD2<Float>,
    near: Float, far: Float
  ) -> simd_float4x4 {
    let d = max(eye.z, 1e-3)
    let scale = near / d
    let l = (screenMin.x - eye.x) * scale
    let r = (screenMax.x - eye.x) * scale
    let b = (screenMin.y - eye.y) * scale
    let t = (screenMax.y - eye.y) * scale
    return frustum(left: l, right: r, bottom: b, top: t, near: near, far: far)
  }

  static func frustum(
    left l: Float, right r: Float, bottom b: Float, top t: Float, near n: Float, far f: Float
  ) -> simd_float4x4 {
    simd_float4x4(
      columns: (
        SIMD4<Float>(2 * n / (r - l), 0, 0, 0),
        SIMD4<Float>(0, 2 * n / (t - b), 0, 0),
        SIMD4<Float>((r + l) / (r - l), (t + b) / (t - b), f / (n - f), -1),
        SIMD4<Float>(0, 0, n * f / (n - f), 0)
      ))
  }
}
