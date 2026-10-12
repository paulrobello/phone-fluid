import Foundation
import simd

/// Agent-operability overrides read from the environment (`FLUID_*`). See the design spec.
struct LaunchConfig: Sendable {
  /// Fixed device-frame gravity in g; replaces Core Motion (the simulator has no sensors).
  var gravity: SIMD3<Float>?
  /// Slowly rotate the gravity vector to exercise sloshing without a device.
  var gravitySweep = false
  var preset: String?
  var backdrop: String?
  var quality: String?
  var screenshotAfter: TimeInterval?
  var exitAfter: TimeInterval?
  var writeStats = false

  static let current = LaunchConfig(environment: ProcessInfo.processInfo.environment)

  init(environment env: [String: String]) {
    gravity = env["FLUID_GRAVITY"].flatMap(Self.parseVector)
    gravitySweep = Self.flag(env["FLUID_GRAVITY_SWEEP"])
    preset = Self.nonEmpty(env["FLUID_PRESET"])
    backdrop = Self.nonEmpty(env["FLUID_BACKDROP"])
    quality = Self.nonEmpty(env["FLUID_QUALITY"])
    screenshotAfter = env["FLUID_SCREENSHOT_AFTER"].flatMap(TimeInterval.init)
    exitAfter = env["FLUID_EXIT_AFTER"].flatMap(TimeInterval.init)
    writeStats = Self.flag(env["FLUID_STATS"])
  }

  var needsReadableDrawable: Bool { screenshotAfter != nil }

  static func parseVector(_ text: String) -> SIMD3<Float>? {
    let parts = text.split(separator: ",").map {
      Float($0.trimmingCharacters(in: .whitespaces))
    }
    guard parts.count == 3, let x = parts[0], let y = parts[1], let z = parts[2] else {
      return nil
    }
    return [x, y, z]
  }

  private static func flag(_ value: String?) -> Bool {
    guard let value else { return false }
    return ["1", "true", "yes"].contains(value.lowercased())
  }

  private static func nonEmpty(_ value: String?) -> String? {
    guard let value, !value.isEmpty else { return nil }
    return value
  }

  static var documentsURL: URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
  }
}
