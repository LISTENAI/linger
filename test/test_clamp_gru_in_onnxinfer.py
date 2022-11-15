


def notest_clamp_gru_in_onnxinfer():
    pass

    # import onnxinfer
    # import torch
    # import numpy
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # numpy.random.seed(1)

    # dummy_input = torch.randn(3, 5, 10).cuda()
    # h0 = torch.randn(2, 3, 20).cuda()

    # sessoption = onnxinfer.InferSessionOptions()
    # sess = onnxinfer.InferSession(
    #     'data.ignore/normalize_torch_gru.onnx', sessoption, is_fuse=False, save_transform_model=None)
    # data = {sess.GetInputNames()[0]: dummy_input, sess.GetInputNames()[1]: h0}
    # rlt = sess.Run(data_in=data)
    # output = rlt[0].AsReadOnlyNumpy()

    # sessoption1 = onnxinfer.InferSessionOptions()
    # sess1 = onnxinfer.InferSession(
    #     'data.ignore/torch_gru.onnx', sessoption1, is_fuse=False, save_transform_model=None)
    # data1 = {sess1.GetInputNames()[0]: dummy_input,
    #          sess1.GetInputNames()[1]: h0}
    # rlt1 = sess1.Run(data_in=data1)
    # output1 = rlt1[0].AsReadOnlyNumpy()
