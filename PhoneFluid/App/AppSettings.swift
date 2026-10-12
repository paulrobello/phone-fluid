import Foundation
import Observation
import UIKit

enum Backdrop: String, CaseIterable, Identifiable, Sendable {
  case checker, glass, photo, void

  var id: String { rawValue }
  var label: String {
    switch self {
    case .checker: "Checkerboard"
    case .glass: "Glass Tank"
    case .photo: "Photo"
    case .void: "Void"
    }
  }
  var shaderIndex: UInt32 {
    switch self {
    case .checker: 0
    case .glass: 1
    case .photo: 2
    case .void: 3
    }
  }
}

enum PerspectiveMode: String, CaseIterable, Identifiable, Sendable {
  case tilt, face

  var id: String { rawValue }
  var label: String { self == .tilt ? "Tilt" : "Head Tracking" }
}

/// User settings, persisted to UserDefaults. Launch-config overrides win at startup.
@MainActor
@Observable
final class AppSettings {
  private let defaults: UserDefaults

  var presetID: String { didSet { defaults.set(presetID, forKey: "presetID") } }
  var backdrop: Backdrop { didSet { defaults.set(backdrop.rawValue, forKey: "backdrop") } }
  var perspectiveMode: PerspectiveMode {
    didSet { defaults.set(perspectiveMode.rawValue, forKey: "perspectiveMode") }
  }
  var fill: Float { didSet { defaults.set(fill, forKey: "fill") } }
  var quality: SimQuality { didSet { defaults.set(quality.rawValue, forKey: "quality") } }
  var hapticsEnabled: Bool { didSet { defaults.set(hapticsEnabled, forKey: "haptics") } }
  var soundEnabled: Bool { didSet { defaults.set(soundEnabled, forKey: "sound") } }
  /// Multiplier on the perspective shift.
  var parallax: Float { didSet { defaults.set(parallax, forKey: "parallax") } }
  /// Bumped when the user picks a new backdrop photo.
  var photoVersion = 0

  /// Bumped to ask the renderer to re-centre the tilt reference.
  var recenterRequest = 0

  var preset: FluidPreset { FluidPreset.named(presetID) }

  init(defaults: UserDefaults = .standard, launch: LaunchConfig = .current) {
    self.defaults = defaults
    presetID = launch.preset ?? defaults.string(forKey: "presetID") ?? FluidPreset.water.id
    backdrop =
      (launch.backdrop ?? defaults.string(forKey: "backdrop")).flatMap(Backdrop.init) ?? .checker
    perspectiveMode =
      (launch.perspective ?? defaults.string(forKey: "perspectiveMode")).flatMap(
        PerspectiveMode.init)
      ?? .tilt
    fill = defaults.object(forKey: "fill") as? Float ?? 0.4
    quality =
      (launch.quality ?? defaults.string(forKey: "quality")).flatMap(SimQuality.init) ?? .low
    hapticsEnabled = defaults.object(forKey: "haptics") as? Bool ?? true
    soundEnabled = defaults.object(forKey: "sound") as? Bool ?? true
    parallax = defaults.object(forKey: "parallax") as? Float ?? 1.0
  }

  static var photoURL: URL { LaunchConfig.documentsURL.appendingPathComponent("backdrop.jpg") }

  func savePhoto(_ image: UIImage) {
    guard let data = image.jpegData(compressionQuality: 0.9) else { return }
    try? data.write(to: Self.photoURL, options: .atomic)
    photoVersion += 1
    backdrop = .photo
  }
}
