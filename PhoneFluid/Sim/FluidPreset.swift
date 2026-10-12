import simd

/// A selectable liquid: physics for the MLS-MPM solver plus the surface look for the renderer.
struct FluidPreset: Identifiable, Equatable, Sendable {
  let id: String
  let name: String

  // Physics (sim units: dx = 1 cell, dt = 0.2 sim time).
  var restDensity: Float = 4
  var stiffness: Float = 3
  var eosPower: Float = 4
  var viscosity: Float = 0.35
  /// Small cohesion keeps the free surface from boiling (pressure clamped at 0 lets surface
  /// particles chatter between compression and free fall).
  var cohesion: Float = 0.05
  /// Multiplier on device gravity (heavy liquids feel snappier).
  var gravityScale: Float = 1
  /// Sim steps per real second (higher = faster-moving liquid).
  var stepsPerSecond: Float = 1200

  // Look.
  /// Beer–Lambert absorption per mm of thickness, per RGB channel.
  var absorption: SIMD3<Float> = [0.085, 0.03, 0.012]
  var tint: SIMD3<Float> = [0.8, 0.97, 1.0]
  /// Minimum opacity of the body (milk and mercury are opaque).
  var opacity: Float = 0
  var refraction: Float = 1
  /// Fresnel F0.
  var reflectivity: Float = 0.02
  var metallic: Float = 0
  var roughness: Float = 0.05
  var foam: Float = 1
  var emission: SIMD3<Float> = .zero
  var emissionStrength: Float = 0
  var causticStrength: Float = 1
  /// Particle speed (cells / sim time) above which foam starts to appear.
  var foamSpeed: Float = 0.65

  static let water = FluidPreset(id: "water", name: "Water")

  static let ocean = FluidPreset(
    id: "ocean", name: "Ocean",
    absorption: [0.16, 0.05, 0.025], tint: [0.55, 0.85, 1.0], foam: 1.4, foamSpeed: 0.5)

  static let honey = FluidPreset(
    id: "honey", name: "Honey",
    viscosity: 2.4, cohesion: 0.15, stepsPerSecond: 900,
    absorption: [0.04, 0.12, 0.45], tint: [1.0, 0.78, 0.35], opacity: 0.15, refraction: 1.3,
    reflectivity: 0.04, roughness: 0.03, foam: 0, causticStrength: 0.7)

  static let oil = FluidPreset(
    id: "oil", name: "Oil",
    viscosity: 0.9, stepsPerSecond: 1100,
    absorption: [0.05, 0.07, 0.25], tint: [0.95, 0.88, 0.45], refraction: 1.2,
    reflectivity: 0.035, roughness: 0.02, foam: 0.1, causticStrength: 0.8)

  static let mercury = FluidPreset(
    id: "mercury", name: "Mercury",
    viscosity: 0.25, cohesion: 0.6, gravityScale: 1.25, stepsPerSecond: 1300,
    absorption: [4, 4, 4], tint: [0.86, 0.88, 0.92], opacity: 1, refraction: 0,
    reflectivity: 0.85, metallic: 1, roughness: 0.04, foam: 0, causticStrength: 0)

  static let lava = FluidPreset(
    id: "lava", name: "Lava",
    viscosity: 2.8, cohesion: 0.1, stepsPerSecond: 800,
    absorption: [0.4, 0.9, 1.4], tint: [0.25, 0.08, 0.04], opacity: 0.9, refraction: 0,
    reflectivity: 0.03, roughness: 0.4, foam: 0, emission: [1.0, 0.32, 0.05],
    emissionStrength: 2.2, causticStrength: 0)

  static let slime = FluidPreset(
    id: "slime", name: "Slime",
    viscosity: 1.6, cohesion: 0.45, stepsPerSecond: 1000,
    absorption: [0.3, 0.03, 0.35], tint: [0.55, 1.0, 0.3], opacity: 0.25, refraction: 1.4,
    reflectivity: 0.05, roughness: 0.02, foam: 0, emission: [0.25, 0.9, 0.1],
    emissionStrength: 0.15, causticStrength: 0.6)

  static let milk = FluidPreset(
    id: "milk", name: "Milk",
    viscosity: 0.45,
    absorption: [0.8, 0.8, 0.8], tint: [0.97, 0.96, 0.92], opacity: 0.92, refraction: 0,
    reflectivity: 0.03, roughness: 0.15, foam: 0.6, causticStrength: 0)

  static let all: [FluidPreset] = [water, ocean, honey, oil, mercury, lava, slime, milk]

  static func named(_ id: String?) -> FluidPreset {
    guard let id else { return .water }
    return all.first { $0.id == id.lowercased() } ?? .water
  }
}
