import CoreHaptics
import Foundation

/// Converts wall-impact energy spikes into transient haptic taps.
@MainActor
final class HapticsController {
  private var engine: CHHapticEngine?
  private var baseline: Float = 0
  private var lastFire: Double = 0
  var enabled = true

  init() {
    guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else { return }
    engine = try? CHHapticEngine()
    engine?.isAutoShutdownEnabled = true
    engine?.resetHandler = { [weak self] in
      Task { @MainActor in try? self?.engine?.start() }
    }
    try? engine?.start()
  }

  /// `impact` is the per-particle wall-impact energy for the last frame.
  func update(impact: Float, time: Double) {
    // Resting liquid presses on the floor every step; react to spikes above that baseline.
    let spike = impact - baseline * 1.6
    baseline += (impact - baseline) * 0.05
    guard enabled, let engine, spike > 0.0004, time - lastFire > 1.0 / 30.0 else { return }
    lastFire = time
    let intensity = min(1, sqrt(spike) * 18)
    guard intensity > 0.08 else { return }
    let event = CHHapticEvent(
      eventType: .hapticTransient,
      parameters: [
        CHHapticEventParameter(parameterID: .hapticIntensity, value: intensity),
        CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.25 + intensity * 0.3),
      ],
      relativeTime: 0)
    if let pattern = try? CHHapticPattern(events: [event], parameters: []),
      let player = try? engine.makePlayer(with: pattern)
    {
      try? player.start(atTime: CHHapticTimeImmediate)
    }
  }

  /// A shake gets a heavier, longer rumble.
  func shake() {
    guard enabled, let engine else { return }
    let events = [
      CHHapticEvent(
        eventType: .hapticContinuous,
        parameters: [
          CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.8),
          CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.15),
        ],
        relativeTime: 0, duration: 0.18)
    ]
    if let pattern = try? CHHapticPattern(events: events, parameters: []),
      let player = try? engine.makePlayer(with: pattern)
    {
      try? player.start(atTime: CHHapticTimeImmediate)
    }
  }
}
