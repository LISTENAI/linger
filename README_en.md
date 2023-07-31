![linger_logo](doc/image/linger_logo.png)
--------------------------------------------------------------------------------
#### English | [Chinese](README.md)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pylinger.svg)](https://pypi.org/project/pylinger)
[![PiPI](https://badge.fury.io/py/pylinger.svg)](https://pypi.org/project/pylinger/)
[![License](https://img.shields.io/github/license/LISTENAI/thinker.svg?style=flat-square)](https://github.com/LISTENAI/linger/blob/main/LICENSE)
[![linux](https://github.com/LISTENAI/linger/actions/workflows/auto_test.yml/badge.svg)](https://github.com/LISTENAI/linger/actions/workflows/auto_test.yml)

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

## Demo
The implementation of AI algorithms basically covers six stages: model specification check, floating-point training, quantization training, model packaging, simulation engine execution, firmware burning and chip operation. The firmware programming and chip operation need to be completed on the development board of Lenses. If necessary, please contact us, and no further introduction will be made here. The flow chart of the other five stages is as follows:  
![lnn_flow_path](doc/image/lnn_flow_path.png)   
Among them, the function of model regularity check is interspersed in quantization training and model packaging.  
We first assume that the model structure is fully compatible with the underlying hardware, introduce each stage in the process, and then introduce the specific implementation of the model convention check (in the actual development process, the convention check should be carried out initially on the model structure to avoid rework in subsequent work).
### 1. Floating-point training
We are based on [pythoch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100) for function demonstration
First of all, Make sure that in the current environment, the floating-point model training can run based on pytorch.  
```Shell
python train.py -net resnet50 -gpu
```
It is recommended to use two-stage quantization training to restrict the range of floating-point training data, and only need to [add a small amount of code](https://github.com/LISTENAI/thinker/blob/main/thinker/docs/tutorial/resnet_modify1.md).  
To avoid conflicts, turn tesnorboard[function off](https://github.com/LISTENAI/thinker/blob/main/thinker/docs/tutorial/resnet_modify2.md). Start the training with the same command, and after running several epochs, a **.pth file is generated in the checkpoint/resnet50 folder

### 2. Quantization training and Export
Load the floating-point model **.pth saved in step 1, and [modify the constraint code](https://github.com/LISTENAI/thinker/blob/main/thinker/docs/images/linger_set2.png) to replace the floating-point operator with a quantized operator. The same command starts quantization training. After several epochs are trained, a **.pth file is also generated in the checkpoint/resnet50 folder.  
Use linger's model conversion tool to [convert the model into an onnx calculation graph](https://github.com/LISTENAI/thinker/blob/main/thinker/docs/images/onnx_export.png).

### 3. Model analysis and packaging
Use the thinker offline tool tpacker to pack the onnx calculation graph generated in step 2   
```Shell
tpacker -g demo/resnet28/resnet18-12-regular.onnx -d Ture -o demo/resnet28/model.bin
```
we  can acquire resource from [thinker/demo/resnet18](https://github.com/LISTENAI/thinker/demo/resnet18/)
### 4. Engine Execution
Use the sample project test_thinker to run the simulation code by specifying the input data, resource file and output file name.  
```Shell
chmod +x ./bin/test_thinker
./bin/test_thinker demo/resnet28/input.bin demo/resnet28/model.bin demo/resnet28/output.bin 3 32 32 6
```
Simplify the overall processing process here, with the engine input being a normalized 3x32x32 image and the output taking max_ The ID corresponding to value is used as the classification result. The processing of input images can refer to the [Image Processing Script](tools/image_process.py), or the processed test set images can be taken from Pytorch cifar100 for testing.

### 5. Conventional check
At this stage, we do not pay attention to the effect of the model, but only pay attention to whether the structure of the model is compatible with the underlying hardware, and the function realization runs through steps 1~4
* In step 1, the model file can be exported by initializing the model parameters or training a few epochs without model convergence.
* Load the model file of step 1 in step 2. When performing quantitative training, the compliance of operator parameters will be checked. If there are any settings that do not meet the requirements, an error will be reported and exit
[error example](https://github.com/LISTENAI/thinker/blob/main/thinker/docs/images/resnet50_linger_err.png). The user modifies the layer parameters according to the error message and returns to step 1 until step 2 is passed.
* Load the calculation graph of step 2 in step 3, the tool will check the tensor size of the node, [if the tensor size exceeds the limit, an error will be reported and exit](https://github.com/LISTENAI/thinker/blob/main/thinker/docs/images/Resnet50_err.png). Otherwise, enter the memory analysis stage, and generate a [memory analysis report](https://github.com/LISTENAI/thinker/blob/main/thinker/docs/images/Resnet50_Mem1.png) in the root directory, and prompt the overall flash /psram/share-memory occupied. For errors that exceed the hardware limit, users can combine the error information and [Memory Analysis Report](https://github.com/LISTENAI/thinker/blob/main/thinker/docs/images/Resnet50_Mem2.png) to locate the calculation graph The overrun operator returns to step 1 to adjust the model structure until [through the packaging process of step 3](https://github.com/LISTENAI/thinker/blob/main/thinker/docs/images/Resnet50_sucess.png ).
So far, the model compliance check has been completed, ensuring that the model can run on the chip. Model efficiency evaluation currently only supports deployment and operation on chips, please contact us for specific needs.

## Quantitative Advancement
  - [Floating-point-fixed-point two-stage quantization training program detailed explanation](doc/tutorial/two_stage_quant_aware_train.md)
  - [Use of weight analysis tools and debugging of quantitative onnx export errors](doc/tutorial/wb_analyse_tool_and_onnx_export_debug_tool.md)

## Frequently Asked Questions
- [Installation problem solving](doc/tutorial/install_bugs.md)
- [Quantification of common problems and notes](doc/tutorial/quant_faq.md)

## Data Search
- [linger API](doc/tutorial/linger_api.md)
- [List of supported quantization OPs](doc/tutorial/support_quant_ops.md) and [their restrictions](https://github.com/LISTENAI/thinker/blob/main/thinker/docs/tutorial/restrain_of_model.md)

## Communication and Feedback
- You are welcome to submit bugs and suggestions via Github Issues
- Technical Communication WeChat Group  
![concat us](doc/image/contact_me_qr.png)

## Reference
- [PyTorch](https://github.com/pytorch/pytorch)
- [ONNX](https://github.com/onnx/onnx)
- [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)
- 
## Applications
* snoring detect[https://github.com/mywang44/snoring_net]
  
## License
- linger is provided by the [Apache-2.0 license](LICENSE)
