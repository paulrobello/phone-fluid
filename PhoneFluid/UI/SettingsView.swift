import PhotosUI
import SwiftUI

struct SettingsView: View {
  @Bindable var settings: AppSettings
  var onRefill: () -> Void

  @State private var photoItem: PhotosPickerItem?

  var body: some View {
    NavigationStack {
      Form {
        Section("Liquid") {
          Picker("Fluid", selection: $settings.presetID) {
            ForEach(FluidPreset.all) { preset in
              Text(preset.name).tag(preset.id)
            }
          }
          VStack(alignment: .leading) {
            Text("Fill level \(Int(settings.fill * 100))%")
            Slider(value: $settings.fill, in: 0.05...0.8)
          }
          Button("Refill tank", action: onRefill)
        }

        Section("Tank") {
          Picker("Backdrop", selection: $settings.backdrop) {
            ForEach(Backdrop.allCases) { Text($0.label).tag($0) }
          }
          PhotosPicker(selection: $photoItem, matching: .images) {
            Label("Choose backdrop photo", systemImage: "photo")
          }
        }

        Section("Perspective") {
          Picker("Mode", selection: $settings.perspectiveMode) {
            ForEach(PerspectiveMode.allCases) { Text($0.label).tag($0) }
          }
          .pickerStyle(.segmented)
          if settings.perspectiveMode == .face && !ARFaceTracker.isSupported {
            Text("Head tracking needs a TrueDepth camera.")
              .font(.footnote)
              .foregroundStyle(.secondary)
          }
          VStack(alignment: .leading) {
            Text("Depth effect \(String(format: "%.1f", settings.parallax))×")
            Slider(value: $settings.parallax, in: 0.3...2.5)
          }
          Button("Re-centre view") { settings.recenterRequest += 1 }
        }

        Section("Quality") {
          Picker("Simulation", selection: $settings.quality) {
            ForEach(SimQuality.allCases) { Text($0.label).tag($0) }
          }
          .pickerStyle(.segmented)
          Toggle("Haptics", isOn: $settings.hapticsEnabled)
          Toggle("Sound", isOn: $settings.soundEnabled)
        }

        Section {
          Text(
            "Tilt to slosh. Drag to stir, tap to splash, shake hard. Two-finger tap shows the controls."
          )
          .font(.footnote)
          .foregroundStyle(.secondary)
        }
      }
      .navigationTitle("Fluid")
      .navigationBarTitleDisplayMode(.inline)
    }
    .onChange(of: photoItem) { _, item in
      guard let item else { return }
      Task {
        if let data = try? await item.loadTransferable(type: Data.self),
          let image = UIImage(data: data)
        {
          settings.savePhoto(image)
        }
      }
    }
  }
}
