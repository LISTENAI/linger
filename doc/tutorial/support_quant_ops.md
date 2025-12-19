
# 目前支持量化的op列表如下

| PyTorch(float32)   | linger算子名称                            | linger导出onnx算子名称                              | 支持关闭的设置                     |
| ------------------ | ----------------------------------------- | --------------------------------------------------- | ---------------------------------- |
| nn.BatchNorm2d     | [QBatchNorm2d]                            | QBatchNorm2d                                      | -                                  |
| nn.LayerNorm2d     | [QLayerNorm2d]                            | QLayerNorm2d                                      | -                                  |
| nn.Linear          | [QLinear]                                 | QLinear                                           | -                                  |
| nn.Conv1d          | [QConv1d]                                 | QConv1d                                           | -                                  |
| nn.Conv2d          | [QConv2d]                                 | QConv2d                                           | -                                  |
| nn.ConvTranspose1d | [QConvTranspose1d]                        | QConvTranspose1d                                  | -                                  |
| nn.ConvTranspose2d | [QConvTranspose2d]                        | QConvTranspose2d                                  | -                                  |
| nn.AvgPool1d       | [QAvgPool1d]                              | QAvgPool1d                                        | -                                  |
| nn.AvgPool2d       | [QAvgPool2d]                              | QAvgPool2d                                        | -                                  |
| nn.MaxPool1d       | [QMaxPool1d]                              | QMaxPool1d                                        | -                                  |
| nn.MaxPool2d       | [QMaxPool2d]                              | QMaxPool2d                                        | -                                  |
| nn.GRU             | [QGRU]                                    | QGRU                                              | -                                  |
| nn.LSTM            | [QLSTM]                                   | QLSTM                                             | -                                  |
| nn.Relu            | [Relu]                                    | Relu                                              | -                                  |
| torch.bmm          | [QBmm]                                    | QBmm                                              | -                                  |
| torch.sigmoid      | [QSigmoid]                                | QSigmoid                                          | -                                  |
| torch.tanh         | [QTanh]                                   | QTanh                                             | -                                  |
| torch.clamp        | [Clamp]                                   | Clamp                                             | -                                  |
| torch.cat          | [QCat]                                    | QCat                                              | -                                  |
| torch.transpose    | [Transpose]                               | Transpose                                         | -                                  |
| view               | [view]                                    | Reshape                                           | -                                  |
| reshape            | [reshape]                                 | Reshape                                           | -                                  |
| squeeze            | [squeeze]                                 | Squeeze                                           | -                                  |
| unsqueeze          | [unsqueeze]                               | Unsqueeze                                         | -                                  |
| flatten            | [flatten]                                 | Flatten                                           | -                                  |
| split              | -                                         | -                                                 | -                                  |
| slice              | [slice]                                   | Slice                                             | -                                  |
| add                | [QAdd]                                    | QAdd                                              | -                                  |
| mul                | [QMul]                                    | QMul                                              | -                                  |
| nn.Embedding       | [QEmbedding]                              | QEmbedding                                        | -                                  |
| layernorm          | [QLayerNorm]                              | QLayerNorm                                        | -                                  |
| softmax            | [QSoftmax]                                | QSoftmax                                          | -                                  |
