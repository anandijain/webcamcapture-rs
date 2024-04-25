# webcamcapture-rs 

a simple script to capture webcam images and run realtime onnx face detection inference on them, drawing a bounding box around faces

i also added another model for doing real time style transfer, however, the latency on that model is ~100ms so its not really "real-time" 

eventually the plan is to use rtmp and rav1e to stream the video to a server

deps: nokhwa, ort, image, minifb


## issues

the biggest issue is that this script takes 70% cpu on my windows machine. so optimization is clearly needed 

## the two supported models are
https://github.com/onnx/models/tree/main/validated/vision/body_analysis/ultraface
https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style
