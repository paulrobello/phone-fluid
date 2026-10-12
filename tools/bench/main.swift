// Headless GPU benchmark of the simulation kernels on the host Mac (reliable GPU timestamps,
// no thermal throttling). Relative costs guide optimisation; absolute phone numbers differ.
import Foundation
import Metal
import simd

@MainActor
func bench() throws {
  guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
  let libURL = URL(fileURLWithPath: CommandLine.arguments[1])
  let library = try device.makeLibrary(URL: libURL)
  let queue = device.makeCommandQueue()!
  let stats = device.makeBuffer(length: 64, options: .storageModeShared)!
  let cells: [Float] = CommandLine.arguments.count > 2
    ? CommandLine.arguments[2].split(separator: ",").compactMap { Float($0) } : [2.6, 2.1, 1.7]
  print("device: \(device.name)")
  for cell in cells {
    for gather in [true, false] {
      let geometry = TankGeometry(
        widthMM: 66.6, heightMM: 144.8, depthMM: 24, quality: .low, cellOverrideMM: cell)
      let sim = try FluidSimulation(device: device, library: library, geometry: geometry)
      sim.useGather = gather
      sim.reset(fill: 0.4, down: [0, -1, 0])
      // Settle, then slosh so particles are shuffled like in real use.
      for g in [SIMD3<Float>(0, -1, 0), simd_normalize(SIMD3<Float>(1, -0.3, 0))] {
        let cb = queue.makeCommandBuffer()!
        sim.encode(into: cb, steps: 300, gravity: g, statsBuffer: stats, statsOffset: 0)
        cb.commit()
        cb.waitUntilCompleted()
      }
      var times: [Double] = []
      for _ in 0..<5 {
        let cb = queue.makeCommandBuffer()!
        sim.encode(into: cb, steps: 40, gravity: [0.6, -0.8, 0], statsBuffer: stats, statsOffset: 0)
        cb.commit()
        cb.waitUntilCompleted()
        times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000 / 40)
      }
      times.sort()
      print(
        String(
          format: "cell %.2f  particles %6d  nodes %6d  %@  ms/step %.3f",
          geometry.cellMM, sim.activeCount, geometry.nodeCount, gather ? "gather " : "scatter",
          times[2]))
    }
  }
}

try MainActor.assumeIsolated { try bench() }
