import AVFoundation
import Synchronization

/// Parameters shared with the real-time audio thread.
final class SloshAudioState: Sendable {
  let level = Atomic<UInt32>(0)  // Float bits, 0...1 slosh loudness
  let brightness = Atomic<UInt32>(0)  // Float bits, 0...1 filter openness
  let splashCount = Atomic<UInt32>(0)  // incremented per splash
  let splashStrength = Atomic<UInt32>(0)

  func set(_ atomic: borrowing Atomic<UInt32>, _ value: Float) {
    atomic.store(value.bitPattern, ordering: .relaxed)
  }

  func get(_ atomic: borrowing Atomic<UInt32>) -> Float {
    Float(bitPattern: atomic.load(ordering: .relaxed))
  }
}

/// Procedural slosh: filtered noise whose loudness follows the liquid's kinetic energy, plus short
/// bright bursts for splashes. Uses the ambient category so the silent switch mutes it.
@MainActor
final class SloshAudio {
  private let engine = AVAudioEngine()
  private let state = SloshAudioState()
  private var source: AVAudioSourceNode?
  private var smoothedEnergy: Float = 0
  private var baseline: Float = 0
  private(set) var running = false

  var enabled = true {
    didSet { enabled ? start() : stop() }
  }

  func start() {
    guard enabled, !running else { return }
    try? AVAudioSession.sharedInstance().setCategory(.ambient, options: [.mixWithOthers])
    try? AVAudioSession.sharedInstance().setActive(true)
    if source == nil {
      let format = engine.outputNode.inputFormat(forBus: 0)
      let sampleRate = Float(format.sampleRate > 0 ? format.sampleRate : 48000)
      let node = Self.makeSourceNode(state: state, sampleRate: sampleRate)
      engine.attach(node)
      let mono = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)
      engine.connect(node, to: engine.mainMixerNode, format: mono)
      engine.mainMixerNode.outputVolume = 0.8
      source = node
    }
    do {
      try engine.start()
      running = true
    } catch {
      running = false
    }
  }

  func stop() {
    guard running else { return }
    engine.pause()
    running = false
  }

  /// `kinetic` = mean squared particle speed this frame, `impact` = per-particle wall impact.
  func update(kinetic: Float, impact: Float) {
    guard running else { return }
    smoothedEnergy += (kinetic - smoothedEnergy) * 0.15
    let level = min(1, sqrt(max(smoothedEnergy, 0)) * 0.9)
    state.set(state.level, level)
    state.set(state.brightness, min(1, level * 1.4))
    let spike = impact - baseline * 1.6
    baseline += (impact - baseline) * 0.05
    if spike > 0.0006 {
      state.set(state.splashStrength, min(1, sqrt(spike) * 14))
      state.splashCount.add(1, ordering: .relaxed)
    }
  }

  func splash(strength: Float) {
    state.set(state.splashStrength, min(1, strength))
    state.splashCount.add(1, ordering: .relaxed)
  }

  private nonisolated static func makeSourceNode(
    state: SloshAudioState, sampleRate: Float
  ) -> AVAudioSourceNode {
    // Render-thread state, owned by the closure.
    final class Voice: @unchecked Sendable {
      var seed: UInt32 = 0x1234_5678
      var low: Float = 0
      var band: Float = 0
      var amp: Float = 0
      var splashEnv: Float = 0
      var splashSeen: UInt32 = 0
      var splashLow: Float = 0
    }
    let voice = Voice()
    return AVAudioSourceNode { _, _, frameCount, audioBufferList -> OSStatus in
      let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
      let targetAmp = state.get(state.level)
      let bright = state.get(state.brightness)
      let count = state.splashCount.load(ordering: .relaxed)
      if count != voice.splashSeen {
        voice.splashSeen = count
        voice.splashEnv = max(voice.splashEnv, state.get(state.splashStrength))
      }
      let cutoff = 180 + bright * 1400
      let k = min(1, 2 * Float.pi * cutoff / sampleRate)
      let splashK = min(1, 2 * Float.pi * 2600 / sampleRate)
      let splashDecay = exp(-1 / (0.09 * sampleRate))
      for frame in 0..<Int(frameCount) {
        voice.seed = voice.seed &* 1_664_525 &+ 1_013_904_223
        let white = Float(voice.seed >> 8) / Float(1 << 24) * 2 - 1
        // Two cascaded one-pole low-passes give a watery rumble.
        voice.low += (white - voice.low) * k
        voice.band += (voice.low - voice.band) * k
        voice.amp += (targetAmp - voice.amp) * 0.0008
        voice.splashLow += (white - voice.splashLow) * splashK
        voice.splashEnv *= splashDecay
        let sample =
          (voice.low - voice.band * 0.6) * voice.amp * 1.8
          + voice.splashLow * voice.splashEnv * 0.5
        for buffer in buffers {
          buffer.mData?.assumingMemoryBound(to: Float.self)[frame] = sample
        }
      }
      return noErr
    }
  }
}
