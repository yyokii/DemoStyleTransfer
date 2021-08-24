//
//  ViewController.swift
//  DemoStyleTransfer
//
//  Created by Higashihara Yoki on 2021/08/19.
//

import AVFoundation
import MetalKit
import UIKit
import VideoToolbox
import Vision

class ViewController: UIViewController {
    
    // Camera Capture
    private var captureSession : AVCaptureSession!
    
    // Metal
    private var metalDevice : MTLDevice!
    private var metalCommandQueue : MTLCommandQueue!
    
    // Core Image
    private var ciContext : CIContext!
    private var currentCIImage : CIImage?
    private var outputWidth: CGFloat! // pixel
    private var outputHeight: CGFloat! // pixel
    
    // UI Component
    private var mtkView: MTKView = MTKView()
    private let modelConfigControl = UISegmentedControl(items: ["Off","CPU", "GPU", "Neural Engine"])
    private let parentStack = UIStackView()
    
    var currentModelConfig = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupUI()
        setupMetal()
        setupCoreImage()
        setupAndStartCaptureSession()
    }
    
    func setupUI(){
        mtkView.translatesAutoresizingMaskIntoConstraints = false
        modelConfigControl.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(mtkView)
        view.addSubview(modelConfigControl)
        
        NSLayoutConstraint.activate([
            mtkView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            mtkView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            mtkView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            mtkView.topAnchor.constraint(equalTo: view.topAnchor),
            
            modelConfigControl.centerXAnchor.constraint(equalTo: view.safeAreaLayoutGuide.centerXAnchor),
            modelConfigControl.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 10),
            modelConfigControl.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -10),
            modelConfigControl.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -40),
        ])
        
        modelConfigControl.selectedSegmentIndex = 0
        modelConfigControl.addTarget(self, action: #selector(modelConfigChanged(_:)), for: .valueChanged)
    }
    
    @objc func modelConfigChanged(_ sender: UISegmentedControl) {
        currentModelConfig = sender.selectedSegmentIndex
    }
    
    private func setupMetal() {
        metalDevice = MTLCreateSystemDefaultDevice()
        mtkView.device = metalDevice
        
        metalCommandQueue = metalDevice.makeCommandQueue()
        
        mtkView.delegate = self
        mtkView.framebufferOnly = false
        
        mtkView.isPaused = true
        mtkView.enableSetNeedsDisplay = true
    }
    
    private func setupCoreImage() {
        ciContext = CIContext(mtlDevice: metalDevice)
    }
    
    private func setupAndStartCaptureSession(){
        self.captureSession = AVCaptureSession()
        
        // setup capture session
        self.captureSession.beginConfiguration()
        if self.captureSession.canSetSessionPreset(.photo) {
            self.captureSession.sessionPreset = .photo
        }
        self.captureSession.automaticallyConfiguresCaptureDeviceForWideColor = true
        self.setupInputs()
        self.setupOutput()
        self.captureSession.commitConfiguration()
        
        self.captureSession.startRunning()
    }
    
    private func setupInputs(){
        guard let backCameraDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            fatalError("❌ no capture device")
        }
        
        guard let backCameraInput = try? AVCaptureDeviceInput(device: backCameraDevice) else {
            fatalError("❌ could not create a capture input")
        }
        
        if !captureSession.canAddInput(backCameraInput) {
            fatalError("❌ could not add capture input to capture session")
        }
        
        captureSession.addInput(backCameraInput)
    }
    
    private func setupOutput(){
        let videoOutput = AVCaptureVideoDataOutput()
        let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInteractive)
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        } else {
            fatalError("❌ could not add video output")
        }
        
        videoOutput.connections.first?.videoOrientation = .portrait
    }
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        guard currentModelConfig != 0 else {
            guard let cvBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                return
            }
            let ciImage = CIImage(cvImageBuffer: cvBuffer)
            
            if (outputWidth == nil) || (outputHeight == nil) {
                outputWidth = ciImage.extent.width
                outputHeight = ciImage.extent.height
            }
            
            self.currentCIImage = ciImage
            
            DispatchQueue.main.async {
                self.mtkView.setNeedsDisplay()
            }
            return
        }
        
        let config = MLModelConfiguration()
        switch currentModelConfig {
        case 1:
            config.computeUnits = .cpuOnly
        case 2:
            config.computeUnits = .cpuAndGPU
        default:
            config.computeUnits = .all
        }
        
        guard let model = try? VNCoreMLModel(for: DemoStyleTransfer02_usecase_image.init(configuration: config).model) else { return }
        
        // Create Vision Request
        let request = VNCoreMLRequest(model: model) { [weak self] (finishedRequest, error) in
            guard let self = self else { return }
            guard let results = finishedRequest.results as? [VNPixelBufferObservation] else { return }
            
            guard let observation = results.first else { return }
            
            let pixelBuffer = observation.pixelBuffer
            var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let scaleX = self.outputWidth / ciImage.extent.width
            let scaleY = self.outputHeight / ciImage.extent.height
            
            ciImage = ciImage.resizeAffine(scaleX: scaleX, scaleY: scaleY)!

            self.currentCIImage = ciImage
                        
            DispatchQueue.main.async {
                self.mtkView.setNeedsDisplay()
            }
        }
        
        request.imageCropAndScaleOption = .scaleFill
        
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }
}

extension ViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        print("\(self.classForCoder)/" + #function)
    }
    
    func draw(in view: MTKView) {
        guard let commandBuffer = metalCommandQueue.makeCommandBuffer() else {
            return
        }
        
        guard let ciImage = currentCIImage else {
            return
        }
        
        guard let currentDrawable = view.currentDrawable else {
            return
        }
        
        let heightOfciImage = ciImage.extent.height
        let heightOfDrawable = view.drawableSize.height
        let yOffsetFromBottom = (heightOfDrawable - heightOfciImage)/2
        
        ciContext.render(ciImage,
                         to: currentDrawable.texture,
                         commandBuffer: commandBuffer,
                         bounds: CGRect(origin: CGPoint(x: 0, y: -yOffsetFromBottom), size: view.drawableSize),
                         colorSpace: CGColorSpaceCreateDeviceRGB())
        
        commandBuffer.present(currentDrawable)
        commandBuffer.commit()
    }
}

extension CIImage {

   func resizeAffine(scaleX: CGFloat, scaleY: CGFloat) -> CIImage? {
         let matrix = CGAffineTransform(scaleX: scaleX, y: scaleY)
       return transformed(by: matrix)
    }
}
