import MetalKit
import SwiftUI

/// MTKView that forwards raw touches to the renderer's touch controller.
final class FluidMTKView: MTKView {
  var touchController: TouchController?
  var onTwoFingerTap: (() -> Void)?

  override init(frame: CGRect, device: MTLDevice?) {
    super.init(frame: frame, device: device)
    isMultipleTouchEnabled = true
    let tap = UITapGestureRecognizer(target: self, action: #selector(handleTwoFingerTap))
    tap.numberOfTouchesRequired = 2
    tap.cancelsTouchesInView = false
    addGestureRecognizer(tap)
  }

  required init(coder: NSCoder) { fatalError("init(coder:) is not supported") }

  @objc private func handleTwoFingerTap() { onTwoFingerTap?() }

  override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
    let now = CACurrentMediaTime()
    for t in touches { touchController?.began(t, in: self, time: now) }
  }

  override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
    let now = CACurrentMediaTime()
    for t in touches { touchController?.moved(t, in: self, time: now) }
  }

  override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
    let now = CACurrentMediaTime()
    for t in touches { touchController?.ended(t, in: self, time: now) }
  }

  override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
    let now = CACurrentMediaTime()
    for t in touches { touchController?.ended(t, in: self, time: now) }
  }
}

struct FluidMetalView: UIViewRepresentable {
  let renderer: Renderer
  var onTwoFingerTap: () -> Void

  func makeUIView(context: Context) -> FluidMTKView {
    let view = FluidMTKView(frame: .zero, device: renderer.device)
    view.colorPixelFormat = .bgra8Unorm
    view.depthStencilPixelFormat = .invalid
    view.preferredFramesPerSecond = 120
    view.framebufferOnly = !LaunchConfig.current.needsReadableDrawable
    view.delegate = renderer
    view.touchController = renderer.touches
    view.onTwoFingerTap = onTwoFingerTap
    view.backgroundColor = .black
    return view
  }

  func updateUIView(_ uiView: FluidMTKView, context: Context) {
    uiView.onTwoFingerTap = onTwoFingerTap
  }
}
