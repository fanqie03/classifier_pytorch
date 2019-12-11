import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnxruntime
import cv2

from classify import Classifier, get_default_args

if __name__ == '__main__':
    cfg = get_default_args()

    classify = Classifier(cfg)
    torch_model = classify.model
    torch_model.to(torch.device('cpu'))

    x = torch.randn(1, 3, 224, 224)


    frame = cv2.imread('/home/cmf/datasets/spoof/NUAADetectedface/Detectedface/ImposterFace/0001/0001_00_00_01_0.jpg')
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))
    # img = np.float32(img)
    img = img.astype(np.float32)
    # print(img.max(), img.min())
    img = img / 255
    # print(img.max(), img.min())
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img)
    x = img

    with torch.no_grad():
        torch_out = torch_model(x)

    print(torch_out, torch_out.shape)
    model_file = '{}_{}.onnx'.format(cfg.theme, cfg.model.type)

    torch.onnx.export(torch_model,
                      x,
                      model_file,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}}
                      )

    ort_session = onnxruntime.InferenceSession(model_file)


    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)
    print(ort_outs[0].shape)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
