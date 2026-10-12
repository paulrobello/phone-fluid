import Foundation
import simd

enum SimQuality: String, CaseIterable, Identifiable, Sendable {
  case low, medium, high

  var id: String { rawValue }

  /// Target grid cell size in millimetres.
  var cellMM: Float {
    switch self {
    case .low: 2.6
    case .medium: 2.1
    case .high: 1.7
    }
  }

  var label: String { rawValue.capitalized }
}

/// Physical tank dimensions and their mapping onto the simulation grid.
///
/// World frame: millimetres, +x right, +y up the screen, +z out of the screen. The glass is z = 0,
/// centred on the screen; the tank interior spans z in [-depth, 0].
struct TankGeometry: Equatable, Sendable {
  static let wallMargin: Int = 2
  /// iPhones since the X are 460 ppi at @3x: 25.4 mm / (460 / 3).
  static let mmPerPoint: Float = 25.4 / (460.0 / 3.0)

  let widthMM: Float
  let heightMM: Float
  let depthMM: Float
  let cellMM: Float
  /// Interior cells per axis.
  let interior: SIMD3<Int>

  init(widthMM: Float, heightMM: Float, depthMM: Float = 24, quality: SimQuality) {
    let nx = max(8, Int((widthMM / quality.cellMM).rounded()))
    let cell = widthMM / Float(nx)
    let ny = max(8, Int((heightMM / cell).rounded()))
    let nz = max(4, Int((depthMM / cell).rounded()))
    self.widthMM = widthMM
    self.heightMM = heightMM
    self.depthMM = Float(nz) * cell
    self.cellMM = cell
    self.interior = [nx, ny, nz]
  }

  init(screenPoints: SIMD2<Float>, quality: SimQuality) {
    self.init(
      widthMM: screenPoints.x * Self.mmPerPoint,
      heightMM: screenPoints.y * Self.mmPerPoint,
      quality: quality)
  }

  var gridSize: SIMD3<UInt32> {
    SIMD3<UInt32>(truncatingIfNeeded: interior &+ Self.wallMargin * 2)
  }

  var nodeCount: Int { Int(gridSize.x) * Int(gridSize.y) * Int(gridSize.z) }

  var boxMin: SIMD3<Float> { [-widthMM / 2, -Float(interior.y) * cellMM / 2, -depthMM] }
  var boxMax: SIMD3<Float> { [widthMM / 2, Float(interior.y) * cellMM / 2, 0] }

  /// World position (mm) of simulation coordinate (0, 0, 0).
  var simOrigin: SIMD3<Float> { boxMin - Float(Self.wallMargin) * cellMM }

  func worldFromSim(_ p: SIMD3<Float>) -> SIMD3<Float> { simOrigin + p * cellMM }
  func simFromWorld(_ p: SIMD3<Float>) -> SIMD3<Float> { (p - simOrigin) / cellMM }
}
