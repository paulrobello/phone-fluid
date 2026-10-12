import MetalKit
import UIKit
import simd

struct SmoothParams {
  var direction: SIMD2<Int32>
  var radius: Int32
  var depthFalloff: Float
}

/// Screen-space fluid renderer: tank interior → particle depth/thickness → smoothing → composite.
@MainActor
final class FluidRenderer {
  /// Fluid buffers render at a fraction of the drawable size; the surface is smooth anyway.
  static let fluidScale: Double = 0.5

  private let device: MTLDevice
  private let boxPipeline: MTLRenderPipelineState
  private let depthPipeline: MTLRenderPipelineState
  private let thicknessPipeline: MTLRenderPipelineState
  private let compositePipeline: MTLRenderPipelineState
  private let smoothPipeline: MTLComputePipelineState
  private let blurThicknessPipeline: MTLComputePipelineState
  private let depthState: MTLDepthStencilState
  private let noDepthState: MTLDepthStencilState
  private let linearSampler: MTLSamplerState
  private let depthSampler: MTLSamplerState

  private var sceneTexture: MTLTexture?
  private var depthTexture: MTLTexture?
  private var depthBuffer: MTLTexture?
  private var smoothA: MTLTexture?
  private var smoothB: MTLTexture?
  private var thicknessTexture: MTLTexture?
  private var thicknessBlurTexture: MTLTexture?
  private(set) var photoTexture: MTLTexture
  private var size: CGSize = .zero

  init(device: MTLDevice, library: MTLLibrary, colorFormat: MTLPixelFormat) throws {
    self.device = device

    func fn(_ name: String) throws -> MTLFunction {
      guard let f = library.makeFunction(name: name) else {
        throw SimulationError.missingFunction(name)
      }
      return f
    }

    let box = MTLRenderPipelineDescriptor()
    box.label = "box"
    box.vertexFunction = try fn("boxVertex")
    box.fragmentFunction = try fn("boxFragment")
    box.colorAttachments[0].pixelFormat = .rgba16Float
    boxPipeline = try device.makeRenderPipelineState(descriptor: box)

    let depth = MTLRenderPipelineDescriptor()
    depth.label = "particleDepth"
    depth.vertexFunction = try fn("particleSprite")
    depth.fragmentFunction = try fn("particleDepth")
    depth.colorAttachments[0].pixelFormat = .r32Float
    depth.depthAttachmentPixelFormat = .depth32Float
    depthPipeline = try device.makeRenderPipelineState(descriptor: depth)

    let thick = MTLRenderPipelineDescriptor()
    thick.label = "particleThickness"
    thick.vertexFunction = try fn("particleSprite")
    thick.fragmentFunction = try fn("particleThickness")
    let attachment = thick.colorAttachments[0]!
    attachment.pixelFormat = .rgba16Float
    attachment.isBlendingEnabled = true
    attachment.rgbBlendOperation = .add
    attachment.alphaBlendOperation = .add
    attachment.sourceRGBBlendFactor = .one
    attachment.destinationRGBBlendFactor = .one
    attachment.sourceAlphaBlendFactor = .one
    attachment.destinationAlphaBlendFactor = .one
    thicknessPipeline = try device.makeRenderPipelineState(descriptor: thick)

    let composite = MTLRenderPipelineDescriptor()
    composite.label = "composite"
    composite.vertexFunction = try fn("fullscreenVertex")
    composite.fragmentFunction = try fn("compositeFragment")
    composite.colorAttachments[0].pixelFormat = colorFormat
    compositePipeline = try device.makeRenderPipelineState(descriptor: composite)

    smoothPipeline = try device.makeComputePipelineState(function: try fn("smoothDepth"))
    blurThicknessPipeline = try device.makeComputePipelineState(
      function: try fn("blurThickness"))

    let ds = MTLDepthStencilDescriptor()
    ds.depthCompareFunction = .less
    ds.isDepthWriteEnabled = true
    depthState = device.makeDepthStencilState(descriptor: ds)!
    noDepthState = device.makeDepthStencilState(descriptor: MTLDepthStencilDescriptor())!

    let linear = MTLSamplerDescriptor()
    linear.minFilter = .linear
    linear.magFilter = .linear
    linear.sAddressMode = .clampToEdge
    linear.tAddressMode = .clampToEdge
    linearSampler = device.makeSamplerState(descriptor: linear)!
    let depthSamp = MTLSamplerDescriptor()
    let filterable = device.supports32BitFloatFiltering
    depthSamp.minFilter = filterable ? .linear : .nearest
    depthSamp.magFilter = filterable ? .linear : .nearest
    depthSamp.sAddressMode = .clampToEdge
    depthSamp.tAddressMode = .clampToEdge
    depthSampler = device.makeSamplerState(descriptor: depthSamp)!

    photoTexture = Self.solidTexture(device: device)
  }

  func setPhoto(_ image: UIImage?) {
    guard let cg = image?.cgImage else {
      photoTexture = Self.solidTexture(device: device)
      return
    }
    let loader = MTKTextureLoader(device: device)
    if let tex = try? loader.newTexture(
      cgImage: cg, options: [.SRGB: false, .generateMipmaps: false])
    {
      photoTexture = tex
    }
  }

  private static func solidTexture(device: MTLDevice) -> MTLTexture {
    let desc = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
    let tex = device.makeTexture(descriptor: desc)!
    var pixel: [UInt8] = [40, 60, 90, 255]
    tex.replace(
      region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &pixel, bytesPerRow: 4)
    return tex
  }

  private func ensureTextures(for drawableSize: CGSize) {
    guard drawableSize != size || sceneTexture == nil else { return }
    size = drawableSize
    let w = max(Int(drawableSize.width), 1)
    let h = max(Int(drawableSize.height), 1)
    let fw = max(Int(Double(w) * Self.fluidScale), 1)
    let fh = max(Int(Double(h) * Self.fluidScale), 1)

    func make(
      _ format: MTLPixelFormat, _ width: Int, _ height: Int, _ usage: MTLTextureUsage,
      _ label: String
    ) -> MTLTexture? {
      let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: format, width: width, height: height, mipmapped: false)
      d.usage = usage
      d.storageMode = .private
      let t = device.makeTexture(descriptor: d)
      t?.label = label
      return t
    }
    sceneTexture = make(.rgba16Float, w, h, [.renderTarget, .shaderRead], "scene")
    depthTexture = make(.r32Float, fw, fh, [.renderTarget, .shaderRead], "fluidDepth")
    depthBuffer = make(.depth32Float, fw, fh, [.renderTarget], "fluidZ")
    smoothA = make(.r32Float, fw, fh, [.shaderRead, .shaderWrite], "smoothA")
    smoothB = make(.r32Float, fw, fh, [.shaderRead, .shaderWrite], "smoothB")
    // Quarter resolution stair-stepped the silhouette (thickness drives the edge alpha).
    thicknessTexture = make(
      .rgba16Float, fw, fh, [.renderTarget, .shaderRead, .shaderWrite], "thickness")
    thicknessBlurTexture = make(.rgba16Float, fw, fh, [.shaderRead, .shaderWrite], "thicknessBlur")
  }

  func encode(
    commandBuffer: MTLCommandBuffer, target: MTLTexture, uniforms: RenderUniforms,
    particles: MTLBuffer, foam: MTLBuffer, density: MTLBuffer, particleCount: Int,
    smoothing: Float = 1
  ) {
    ensureTextures(for: CGSize(width: target.width, height: target.height))
    guard let scene = sceneTexture, let depth = depthTexture, let zbuf = depthBuffer,
      let smoothA, let smoothB, let thickness = thicknessTexture
    else { return }
    var u = uniforms
    let uSize = MemoryLayout<RenderUniforms>.stride

    // 1. Tank interior.
    let boxPass = MTLRenderPassDescriptor()
    boxPass.colorAttachments[0].texture = scene
    boxPass.colorAttachments[0].loadAction = .clear
    boxPass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    boxPass.colorAttachments[0].storeAction = .store
    if let enc = commandBuffer.makeRenderCommandEncoder(descriptor: boxPass) {
      enc.label = "Tank"
      enc.setRenderPipelineState(boxPipeline)
      enc.setVertexBytes(&u, length: uSize, index: 0)
      enc.setFragmentBytes(&u, length: uSize, index: 0)
      enc.setFragmentTexture(photoTexture, index: 0)
      enc.setFragmentSamplerState(linearSampler, index: 0)
      enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 30)
      enc.endEncoding()
    }

    do {
      // 2. Nearest-surface depth.
      let depthPass = MTLRenderPassDescriptor()
      depthPass.colorAttachments[0].texture = depth
      depthPass.colorAttachments[0].loadAction = .clear
      depthPass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
      depthPass.colorAttachments[0].storeAction = .store
      depthPass.depthAttachment.texture = zbuf
      depthPass.depthAttachment.loadAction = .clear
      depthPass.depthAttachment.clearDepth = 1
      depthPass.depthAttachment.storeAction = .dontCare
      if let enc = commandBuffer.makeRenderCommandEncoder(descriptor: depthPass) {
        enc.label = "Fluid depth"
        enc.setRenderPipelineState(depthPipeline)
        enc.setDepthStencilState(depthState)
        var scale: Float = 1
        enc.setVertexBuffer(particles, offset: 0, index: 0)
        enc.setVertexBuffer(foam, offset: 0, index: 3)
        enc.setVertexBuffer(density, offset: 0, index: 4)
        enc.setVertexBytes(&u, length: uSize, index: 1)
        enc.setVertexBytes(&scale, length: 4, index: 2)
        enc.setFragmentBytes(&u, length: uSize, index: 0)
        if particleCount > 0 {
          enc.drawPrimitives(
            type: .triangleStrip, vertexStart: 0, vertexCount: 4,
            instanceCount: particleCount)
        }
        enc.endEncoding()
      }

      // 3. Thickness and foam.
      let thickPass = MTLRenderPassDescriptor()
      thickPass.colorAttachments[0].texture = thickness
      thickPass.colorAttachments[0].loadAction = .clear
      thickPass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
      thickPass.colorAttachments[0].storeAction = .store
      if let enc = commandBuffer.makeRenderCommandEncoder(descriptor: thickPass) {
        enc.label = "Fluid thickness"
        enc.setRenderPipelineState(thicknessPipeline)
        enc.setDepthStencilState(noDepthState)
        var scale: Float = 1.6
        enc.setVertexBuffer(particles, offset: 0, index: 0)
        enc.setVertexBuffer(foam, offset: 0, index: 3)
        enc.setVertexBuffer(density, offset: 0, index: 4)
        enc.setVertexBytes(&u, length: uSize, index: 1)
        enc.setVertexBytes(&scale, length: 4, index: 2)
        enc.setFragmentBytes(&u, length: uSize, index: 0)
        if particleCount > 0 {
          enc.drawPrimitives(
            type: .triangleStrip, vertexStart: 0, vertexCount: 4,
            instanceCount: particleCount)
        }
        enc.endEncoding()
      }

      // 4. Bilateral smoothing.
      if let enc = commandBuffer.makeComputeCommandEncoder() {
        enc.label = "Smooth depth"
        enc.setComputePipelineState(smoothPipeline)
        let w = smoothPipeline.threadExecutionWidth
        let h = smoothPipeline.maxTotalThreadsPerThreadgroup / w
        let grid = MTLSize(width: depth.width, height: depth.height, depth: 1)
        let group = MTLSize(width: w, height: h, depth: 1)
        // Blur across roughly one and a half particle radii, in fluid-buffer pixels.
        let pxPerMM = Float(depth.width) / max(uniforms.boxMax.x - uniforms.boxMin.x, 1)
        let radius = Int32(
          min(max(Int(uniforms.particleRadiusMM * pxPerMM * 1.8 * smoothing), 6), 48))
        // Dense separable pairs (sparse taps stair-stepped the silhouette); creamy liquids get a
        // second pair.
        var passes: [(MTLTexture, MTLTexture, SIMD2<Int32>)] = [
          (depth, smoothA, [1, 0]), (smoothA, smoothB, [0, 1]),
        ]
        if smoothing > 1.5 {
          passes += [
            (smoothB, smoothA, SIMD2<Int32>(1, 0)), (smoothA, smoothB, SIMD2<Int32>(0, 1)),
          ]
        }
        for (src, dst, dir) in passes {
          var sp = SmoothParams(
            direction: dir, radius: radius, depthFalloff: uniforms.particleRadiusMM * 3)
          enc.setTexture(src, index: 0)
          enc.setTexture(dst, index: 1)
          enc.setBytes(&sp, length: MemoryLayout<SmoothParams>.stride, index: 0)
          enc.dispatchThreads(grid, threadsPerThreadgroup: group)
        }
        // Thickness/foam blur over about one particle radius (H into blur, V back).
        if let thicknessBlur = thicknessBlurTexture {
          enc.setComputePipelineState(blurThicknessPipeline)
          let blurRadius = Int32(min(max(Int(uniforms.particleRadiusMM * pxPerMM), 3), 16))
          for (src, dst, dir) in [
            (thickness, thicknessBlur, SIMD2<Int32>(1, 0)),
            (thicknessBlur, thickness, SIMD2<Int32>(0, 1)),
          ] {
            var sp = SmoothParams(direction: dir, radius: blurRadius, depthFalloff: 0)
            enc.setTexture(src, index: 0)
            enc.setTexture(dst, index: 1)
            enc.setBytes(&sp, length: MemoryLayout<SmoothParams>.stride, index: 0)
            enc.dispatchThreads(grid, threadsPerThreadgroup: group)
          }
        }
        enc.endEncoding()
      }
    }

    // 5. Composite to the drawable.
    let finalPass = MTLRenderPassDescriptor()
    finalPass.colorAttachments[0].texture = target
    finalPass.colorAttachments[0].loadAction = .dontCare
    finalPass.colorAttachments[0].storeAction = .store
    if let enc = commandBuffer.makeRenderCommandEncoder(descriptor: finalPass) {
      enc.label = "Composite"
      enc.setRenderPipelineState(compositePipeline)
      enc.setFragmentBytes(&u, length: uSize, index: 0)
      enc.setFragmentTexture(scene, index: 0)
      enc.setFragmentTexture(smoothB, index: 1)
      enc.setFragmentTexture(thickness, index: 2)
      enc.setFragmentTexture(photoTexture, index: 3)
      enc.setFragmentSamplerState(linearSampler, index: 0)
      enc.setFragmentSamplerState(depthSampler, index: 1)
      enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
      enc.endEncoding()
    }
  }
}
