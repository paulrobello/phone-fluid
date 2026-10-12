import AVFoundation

/// Sample-based liquid audio (ElevenLabs-generated, see Resources/Audio-PROVENANCE.md): a
/// looping slosh bed whose volume and playback rate follow the liquid's motion, plus splash
/// one-shots on wave impacts. Uses the ambient category so the silent switch mutes it.
@MainActor
final class SloshAudio {
  private let engine = AVAudioEngine()
  private let loopPlayer = AVAudioPlayerNode()
  private let loopVarispeed = AVAudioUnitVarispeed()
  private let loopFilter = AVAudioUnitEQ(numberOfBands: 1)
  private var splashPlayers: [(player: AVAudioPlayerNode, speed: AVAudioUnitVarispeed)] = []
  private var nextSplash = 0
  private var loopBuffer: AVAudioPCMBuffer?
  private var smallSplash: AVAudioPCMBuffer?
  private var bigSplash: AVAudioPCMBuffer?
  private var configured = false
  private(set) var running = false

  private var level: Float = 0
  private var baseline: Float = 0
  private var lastSplash: Double = 0
  /// Thick liquids slosh lower and slower.
  private var presetRate: Float = 1

  var enabled = true {
    didSet { enabled ? start() : stop() }
  }

  func setPreset(_ preset: FluidPreset) {
    let thick = min(max(1.0 - preset.viscosity * 0.12, 0.6), 1.05)
    presetRate = thick * (preset.metallic > 0.5 ? 1.15 : 1)
  }

  func start() {
    guard enabled, !running else { return }
    try? AVAudioSession.sharedInstance().setCategory(.ambient, options: [.mixWithOthers])
    try? AVAudioSession.sharedInstance().setActive(true)
    configureIfNeeded()
    guard let loopBuffer else { return }
    do {
      try engine.start()
      running = true
      loopPlayer.volume = 0
      loopPlayer.scheduleBuffer(loopBuffer, at: nil, options: .loops)
      loopPlayer.play()
    } catch {
      running = false
    }
  }

  func stop() {
    guard running else { return }
    loopPlayer.stop()
    for s in splashPlayers { s.player.stop() }
    engine.pause()
    running = false
  }

  private func configureIfNeeded() {
    guard !configured else { return }
    configured = true
    loopBuffer = Self.load("slosh_loop")
    smallSplash = Self.load("splash_small")
    bigSplash = Self.load("splash_big")
    guard let format = loopBuffer?.format ?? smallSplash?.format else { return }

    let band = loopFilter.bands[0]
    band.filterType = .lowPass
    band.frequency = 1200
    band.bypass = false
    engine.attach(loopPlayer)
    engine.attach(loopVarispeed)
    engine.attach(loopFilter)
    engine.connect(loopPlayer, to: loopVarispeed, format: format)
    engine.connect(loopVarispeed, to: loopFilter, format: format)
    engine.connect(loopFilter, to: engine.mainMixerNode, format: format)

    for _ in 0..<4 {
      let player = AVAudioPlayerNode()
      let speed = AVAudioUnitVarispeed()
      engine.attach(player)
      engine.attach(speed)
      engine.connect(player, to: speed, format: format)
      engine.connect(speed, to: engine.mainMixerNode, format: format)
      splashPlayers.append((player, speed))
    }
    engine.mainMixerNode.outputVolume = 0.9
  }

  private static func load(_ name: String) -> AVAudioPCMBuffer? {
    guard let url = Bundle.main.url(forResource: name, withExtension: "m4a"),
      let file = try? AVAudioFile(forReading: url),
      let buffer = AVAudioPCMBuffer(
        pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
    else { return nil }
    try? file.read(into: buffer)
    return buffer
  }

  /// `kinetic` = mean squared particle speed this frame, `impact` = per-particle wall impact.
  func update(kinetic: Float, impact: Float, time: Double) {
    guard running else { return }
    // Resting jitter sits near 0.003 (RMS 0.05); sloshing reaches 0.05-0.2.
    let target = min(max((kinetic - 0.005) / 0.08, 0), 1)
    level += (target - level) * (target > level ? 0.25 : 0.06)
    loopPlayer.volume = pow(level, 0.8) * 0.9
    loopVarispeed.rate = presetRate * (0.85 + 0.3 * level)
    loopFilter.bands[0].frequency = 900 + 5000 * level

    let spike = impact - baseline * 1.6
    baseline += (impact - baseline) * 0.05
    if spike > 0.0006, time - lastSplash > 0.12 {
      lastSplash = time
      splash(strength: min(1, sqrt(spike) * 14))
    }
  }

  func splash(strength: Float) {
    guard running else { return }
    let buffer = strength > 0.55 ? bigSplash : smallSplash
    guard let buffer else { return }
    let slot = splashPlayers[nextSplash]
    nextSplash = (nextSplash + 1) % splashPlayers.count
    slot.player.stop()
    slot.player.volume = 0.25 + 0.75 * strength
    slot.speed.rate = presetRate * Float.random(in: 0.9...1.1)
    slot.player.scheduleBuffer(buffer, at: nil)
    slot.player.play()
  }
}
