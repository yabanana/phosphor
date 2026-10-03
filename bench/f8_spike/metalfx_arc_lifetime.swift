// Independent Swift ARC control. Build with xcrun swiftc -swift-version 5.
// Exit 1 reproduces surviving targets on MetalFX 40.9; it is not a passing gate.
import Foundation
import Metal
import MetalFX

final class Observer { weak var scaler: AnyObject?; init(_ scaler: AnyObject) { self.scaler = scaler } }
var observers: [Observer] = []
autoreleasepool {
    let device = MTLCreateSystemDefaultDevice()!
    let compiler = try! device.makeCompiler(descriptor: MTL4CompilerDescriptor())
    let initial = device.currentAllocatedSize
    for i in 1...8 {
        autoreleasepool {
            let descriptor = MTLFXTemporalScalerDescriptor()
            descriptor.inputWidth = 640; descriptor.inputHeight = 360
            descriptor.outputWidth = 640; descriptor.outputHeight = 360
            descriptor.colorTextureFormat = .rgba16Float; descriptor.outputTextureFormat = .rgba16Float
            descriptor.depthTextureFormat = .depth32Float; descriptor.motionTextureFormat = .rg16Float
            descriptor.requiresSynchronousInitialization = true
            let scaler = descriptor.makeTemporalScaler(device: device, compiler: compiler)!
            observers.append(Observer(scaler))
        }
        print("Swift iteration \(i), live \(observers.filter {$0.scaler != nil}.count), device delta \(device.currentAllocatedSize-initial)")
    }
}
DispatchQueue.main.asyncAfter(deadline: .now()+3) {
    let alive = observers.filter {$0.scaler != nil}.count
    print("Swift ARC after main dispatch drain: \(alive)")
    exit(alive == 0 ? 0 : 1)
}
dispatchMain()
