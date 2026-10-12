import Metal
import SwiftUI

@main
struct PhoneFluidApp: App {
  @Environment(\.scenePhase) private var scenePhase
  @State private var settings: AppSettings
  @State private var renderer: Renderer?

  init() {
    let settings = AppSettings()
    _settings = State(initialValue: settings)
    let renderer = MTLCreateSystemDefaultDevice().flatMap {
      try? Renderer(device: $0, settings: settings)
    }
    _renderer = State(initialValue: renderer)
    UIApplication.shared.isIdleTimerDisabled = true
  }

  var body: some Scene {
    WindowGroup {
      ContentView(renderer: renderer, settings: settings)
        .preferredColorScheme(.dark)
    }
    .onChange(of: scenePhase) { _, phase in
      switch phase {
      case .active: renderer?.didBecomeActive()
      case .background, .inactive: renderer?.willResignActive()
      @unknown default: break
      }
    }
  }
}
