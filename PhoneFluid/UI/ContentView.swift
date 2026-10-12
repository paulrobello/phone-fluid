import SwiftUI

struct ContentView: View {
  let renderer: Renderer?
  @Bindable var settings: AppSettings

  @State private var showSettings = false
  @State private var controlsVisible = true
  @State private var hideTask: Task<Void, Never>?

  var body: some View {
    ZStack(alignment: .topTrailing) {
      if let renderer {
        FluidMetalView(renderer: renderer, onTwoFingerTap: revealControls)
          .ignoresSafeArea()
      } else {
        Color.black.ignoresSafeArea()
        Text("Metal is unavailable on this device.")
          .foregroundStyle(.white)
          .frame(maxWidth: .infinity, maxHeight: .infinity)
      }

      Button {
        showSettings = true
      } label: {
        Image(systemName: "slider.horizontal.3")
          .font(.system(size: 17, weight: .semibold))
          .foregroundStyle(.white)
          .frame(width: 44, height: 44)
          .background(.ultraThinMaterial, in: Circle())
      }
      .accessibilityLabel("Settings")
      .padding(.trailing, 18)
      .padding(.top, 8)
      .opacity(controlsVisible ? 1 : 0)
      .allowsHitTesting(controlsVisible)
      .animation(.easeInOut(duration: 0.4), value: controlsVisible)
    }
    .persistentSystemOverlays(.hidden)
    .statusBarHidden()
    .sheet(isPresented: $showSettings, onDismiss: scheduleHide) {
      SettingsView(settings: settings, onRefill: { renderer?.resetFluid() })
        .presentationDetents([.medium, .large])
        .presentationBackground(.ultraThinMaterial)
    }
    .onAppear(perform: scheduleHide)
  }

  private func revealControls() {
    controlsVisible = true
    scheduleHide()
  }

  private func scheduleHide() {
    hideTask?.cancel()
    hideTask = Task {
      try? await Task.sleep(for: .seconds(3))
      guard !Task.isCancelled, !showSettings else { return }
      controlsVisible = false
    }
  }
}
