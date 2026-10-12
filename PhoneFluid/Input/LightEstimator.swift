import UIKit
import simd

/// Scene lighting that follows the real environment.
///
/// iOS exposes no ambient-light-sensor API. With head tracking on, ARKit's light estimate gives
/// intensity, colour temperature and a dominant direction; otherwise the screen brightness (which
/// auto-brightness drives from the light sensor) stands in for intensity.
@MainActor
final class LightEstimator {
  struct Light {
    var color: SIMD3<Float>
    var ambient: SIMD3<Float>
    /// Toward the light, device/world frame.
    var direction: SIMD3<Float>
  }

  static let defaultDirection = simd_normalize(SIMD3<Float>(0.35, 0.85, 0.6))

  private var intensity: Float = 1
  private var tint = SIMD3<Float>(1, 1, 1)
  private var direction = defaultDirection

  var current: Light {
    Light(
      color: tint * intensity * 1.05,
      ambient: tint * (0.35 + 0.65 * intensity),
      direction: direction)
  }

  func update(face: FaceState?, screenBrightness: Float, dt: Float) {
    var targetIntensity: Float
    var targetTint = SIMD3<Float>(1, 1, 1)
    var targetDir = Self.defaultDirection
    if let face, let lumens = face.ambientIntensity {
      targetIntensity = Self.intensity(lumens: lumens)
      if let kelvin = face.colorTemperature { targetTint = Self.rgb(kelvin: kelvin) }
      if let dir = face.lightDirection, dir.y > -0.2 {
        // Lights are almost always above; keep a front bias so the liquid stays readable.
        targetDir = simd_normalize(dir + SIMD3<Float>(0, 0.3, 0.4))
      }
    } else {
      targetIntensity = Self.intensity(brightness: screenBrightness)
    }
    let k = 1 - exp(-dt / 1.0)
    intensity += (targetIntensity - intensity) * k
    tint += (targetTint - tint) * k
    direction = simd_normalize(direction + (targetDir - direction) * k)
  }

  /// ARKit ambient intensity: ~1000 lm is neutral indoor lighting.
  nonisolated static func intensity(lumens: Float) -> Float {
    min(max(pow(max(lumens, 1) / 1000, 0.6), 0.25), 1.6)
  }

  /// Auto-brightness proxy: dim rooms → dim scene.
  nonisolated static func intensity(brightness: Float) -> Float {
    0.35 + min(max(brightness, 0), 1) * 0.85
  }

  /// Approximate black-body colour (Tanner Helland's fit), normalised so 6500 K ≈ white.
  nonisolated static func rgb(kelvin: Float) -> SIMD3<Float> {
    let t = min(max(kelvin, 1500), 12000) / 100
    var r: Float
    var g: Float
    var b: Float
    if t <= 66 {
      r = 255
      g = 99.4708025861 * log(t) - 161.1195681661
      b = t <= 19 ? 0 : 138.5177312231 * log(t - 10) - 305.0447927307
    } else {
      r = 329.698727446 * pow(t - 60, -0.1332047592)
      g = 288.1221695283 * pow(t - 60, -0.0755148492)
      b = 255
    }
    let c = simd_clamp(SIMD3<Float>(r, g, b) / 255, .zero, .one)
    let white = SIMD3<Float>(1, 0.996, 0.98)
    return simd_clamp(c / white, .zero, SIMD3<Float>(repeating: 1.2))
  }
}
