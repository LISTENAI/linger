![linger_logo](doc/image/linger_logo.png)
--------------------------------------------------------------------------------
English | [Chinese](README.md)


Linger is an open source neural network quantization training component by LISTENAI, designed for use with the AIOT chip CSK60XX. This component combines Linger's open source high-performance neural network inference framework
[thinker](https://github.com/LISTENAI/thinker) can achieve training and inference integration, helping AI developers to quickly give business with AI capabilities based on CSK chip. Currently linger + thinker tool chain has supported the use of CSK chip in more than 10 AI application scenarios such as computer vision, voice wakeup, speech recognition, offline translation, etc.


## Introduction
The linger is based on PyTorch to deeply customize the LISTENAI LUNA series chip, quantize the activation and weight to 8bit in the forward process of neural network training, and get the quantized lossless 8bit model by parameter adjustment.

![doc/image/solution.png](doc/image/solution.png)

## Technical Highlights
### 1. High Ease of Use
linger is a PyTorch-based quantization scheme. Adding one line of linger-related code to the original floating-point training code can complete the replacement of quantization operators, and the quantization training can be completed using the original training process without other complicated settings.

### 2. Good Scalability
linger is based on PyTorch to build quantization operators, so you can add any quantization operator to linger to complete your quantization needs as long as it meets the specifications of PyTorch extension operators.

### 3. Complete Toolchain
The backend is adapted to [thinker](https://github.com/LISTENAI/thinker) inference engine, thinker inference engine for CSK60XX, which is fully functional and seamlessly integrates quantization training and inference process, while the binary consistency of training and inference is guaranteed.


## Quick Start
1. [Installation](doc/tutorial/install.md)：support pip, source code, docker and other installation methods
2. [Floating-point-fixed-point two-stage quantization training](doc/tutorial/get_started_for_two_stage.md): first the constraint training of floating-point network, and then the quantization training fine-tuning for the quantization-friendly floating-point model
3. [ONNX export tutorial](doc/tutorial/from_mode_to_onnx.md)：exporting quantized lossless PyTorch models to ONNX format
4. [Complete introductory examples](examples/)：provide several newbie-friendly introductory quantization examples

## Quantitative Advancement
  - [Floating-point-fixed-point two-stage quantization training program detailed explanation](doc/tutorial/two_stage_quant_aware_train.md)
  - [Use of weight analysis tools and debugging of quantitative onnx export errors](doc/tutorial/wb_analyse_tool_and_onnx_export_debug_tool.md)

## Frequently Asked Questions
- [Installation problem solving](doc/tutorial/install_bugs.md)
- [Quantification of common problems and notes](doc/tutorial/quant_faq.md)

## Data Search
- [linger API](doc/tutorial/linger_api.md)
- [List of supported quantization OPs and their restrictions](doc/tutorial/support_quant_ops.md)

## Communication and Feedback
- You are welcome to submit bugs and suggestions via Github Issues
- Toolchain course
- Technical Exchange QQ Group
- Technical Communication WeChat Group

## Reference
- [PyTorch](https://github.com/pytorch/pytorch)
- [ONNX](https://github.com/onnx/onnx)

## License
- linger is provided by the [Apache-2.0 license](LICENSE)