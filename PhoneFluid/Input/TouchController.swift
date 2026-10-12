import UIKit
import simd

/// Tracks finger touches on the glass and turns them into stirring forces and tap splashes.
@MainActor
final class TouchController {
  private struct Track {
    var point: CGPoint
    var previous: CGPoint
    var lastTime: Double
    var startTime: Double
    var startPoint: CGPoint
    var velocity = SIMD2<Float>.zero  // points per second, screen axes
  }

  private var tracks: [ObjectIdentifier: Track] = [:]
  private var pendingSplashes: [CGPoint] = []

  func began(_ touch: UITouch, in view: UIView, time: Double) {
    let p = touch.location(in: view)
    tracks[ObjectIdentifier(touch)] = Track(
      point: p, previous: p, lastTime: time, startTime: time, startPoint: p)
  }

  func moved(_ touch: UITouch, in view: UIView, time: Double) {
    let id = ObjectIdentifier(touch)
    guard var t = tracks[id] else { return }
    let p = touch.location(in: view)
    let dt = max(time - t.lastTime, 1.0 / 240.0)
    let v = SIMD2<Float>(Float((p.x - t.point.x) / dt), Float((p.y - t.point.y) / dt))
    t.velocity = t.velocity * 0.4 + v * 0.6
    t.previous = t.point
    t.point = p
    t.lastTime = time
    tracks[id] = t
  }

  func ended(_ touch: UITouch, in view: UIView, time: Double) {
    let id = ObjectIdentifier(touch)
    guard let t = tracks.removeValue(forKey: id) else { return }
    let travel = hypot(t.point.x - t.startPoint.x, t.point.y - t.startPoint.y)
    if time - t.startTime < 0.25, travel < 12 {
      pendingSplashes.append(t.point)
    }
  }

  func cancelAll() { tracks.removeAll() }

  /// Builds this frame's GPU touch inputs. `screenSize` is in points.
  func inputs(
    screenSize: CGSize, geometry: TankGeometry, simTimePerSecond: Float, extraSplash: Float
  ) -> [TouchInput] {
    var result: [TouchInput] = []
    let mm = TankGeometry.mmPerPoint
    func simPoint(_ p: CGPoint) -> SIMD2<Float> {
      let world = SIMD3<Float>(
        (Float(p.x) - Float(screenSize.width) / 2) * mm,
        (Float(screenSize.height) / 2 - Float(p.y)) * mm, 0)
      let s = geometry.simFromWorld(world)
      return [s.x, s.y]
    }
    let radius = 9 / geometry.cellMM
    for t in tracks.values.prefix(Int(MAX_TOUCHES)) {
      // points/s → mm/s → cells/s → cells per sim time; screen y is flipped.
      var v = SIMD2<Float>(t.velocity.x, -t.velocity.y) * mm / geometry.cellMM
      v /= max(simTimePerSecond, 1)
      let speed = simd_length(v)
      if speed > 3 { v *= 3 / speed }
      let pos = simPoint(t.point)
      result.append(
        TouchInput(
          positionRadius: [pos.x, pos.y, radius, 0.35],
          velocity: [v.x, v.y, 0, 0]))
    }
    for splash in pendingSplashes where result.count < Int(MAX_TOUCHES) {
      let pos = simPoint(splash)
      result.append(
        TouchInput(
          positionRadius: [pos.x, pos.y, radius * 1.6, 0],
          velocity: [0, 0, 2.2 + extraSplash, 0]))
    }
    pendingSplashes.removeAll()
    return result
  }
}
