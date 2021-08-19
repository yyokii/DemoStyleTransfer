//
//  ViewController.swift
//  DemoStyleTransfer
//
//  Created by Higashihara Yoki on 2021/08/19.
//

import AVFoundation
import UIKit
import VideoToolbox
import Vision


class ViewController: UIViewController {
    
    // UI Component
    let imageView = UIImageView()
    let modelConfigControlItems = ["Off","CPU", "GPU", "Neural Engine"]
    let modelConfigControl = UISegmentedControl(items: ["Off","CPU", "GPU", "Neural Engine"])
    let parentStack = UIStackView()
    
    var currentModelConfig = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupUI()
        setupCapture()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        parentStack.frame = CGRect(x: 0, y: 0, width: view.frame.width, height: view.frame.height * 0.8)
    }
    
    func setupUI(){
        view.addSubview(parentStack)
        parentStack.axis = NSLayoutConstraint.Axis.vertical
        parentStack.distribution = UIStackView.Distribution.fill
        
        parentStack.addArrangedSubview(imageView)
        parentStack.addArrangedSubview(modelConfigControl)
        
        imageView.contentMode = UIView.ContentMode.scaleAspectFit
        
        modelConfigControl.selectedSegmentIndex = 0
        
        modelConfigControl.addTarget(self, action: #selector(modelConfigChanged(_:)), for: .valueChanged)
    }
    
    @objc func modelConfigChanged(_ sender: UISegmentedControl) {
        currentModelConfig = sender.selectedSegmentIndex
    }
    
    func setupCapture() {
        // Input Device Settings
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = AVCaptureSession.Preset.medium
        let availableDevices = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: AVMediaType.video, position: .back).devices
        do {
            if let captureDevice = availableDevices.first {
                captureSession.addInput(try AVCaptureDeviceInput(device: captureDevice))
            }
        } catch {
            print(error.localizedDescription)
        }
        
        // Video output settings
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        if captureSession.canAddOutput(videoOutput){
            captureSession.addOutput(videoOutput)
        }
        guard let connection = videoOutput.connection(with: .video) else { return }
        guard connection.isVideoOrientationSupported else { return }
        connection.videoOrientation = .portrait
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        view.layer.addSublayer(previewLayer)
        captureSession.startRunning()
    }
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        guard currentModelConfig != 0 && (currentModelConfig < modelConfigControlItems.count) else {
            DispatchQueue.main.async(execute: {
                self.imageView.image = .init(buffer: sampleBuffer)
            })
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
        let request = VNCoreMLRequest(model: model) { (finishedRequest, error) in
            guard let results = finishedRequest.results as? [VNPixelBufferObservation] else { return }
            
            guard let observation = results.first else { return }
            
            DispatchQueue.main.async(execute: {
                self.imageView.image = UIImage(pixelBuffer: observation.pixelBuffer)
            })
        }
        
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
        
    }
}

extension UIImage {
    
    /// CVPixelBuffer to UIImage
    public convenience init?(pixelBuffer: CVPixelBuffer) {
        var cgImage: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &cgImage)
        
        if let cgImage = cgImage {
            self.init(cgImage: cgImage)
        } else {
            return nil
        }
    }
    
    /// CMSampleBuffer to UIImage
    public convenience init?(buffer: CMSampleBuffer) {
        let pixelBuffer: CVImageBuffer = CMSampleBufferGetImageBuffer(buffer)!
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        let pixelBufferWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let pixelBufferHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let imageRect: CGRect = CGRect(x: 0, y: 0, width: pixelBufferWidth, height: pixelBufferHeight)
        let ciContext = CIContext.init()
        let cgImage = ciContext.createCGImage(ciImage, from: imageRect )
        
        if let cgImage = cgImage {
            self.init(cgImage: cgImage)
        } else {
            return nil
        }
    }
}
