ncnn_home='/home/cmf/dev/ncnn-20191113'
ncnn_build='/home/cmf/dev/ncnn-20191113/build-host-gcc-linux'

model='spoof_mobilenet_v2'
#model='spoof-FeatherNetB-_FeatherNetB'
#model='spoof-Mb_Tiny_RFB-_Mb_Tiny_RFB'
model='spoof-ir-Mb_Tiny_RFB-_Mb_Tiny_RFB'


#将pytorch模型转成onnx
#python tools/pt2onnx.py --config classify/config/defaults.yaml configs.spoof_mbtinyrfb.py --ckpt checkpoint/spoof-Mb_Tiny_RFB-2019-12-06-10-13-14/ckpt/Mb_Tiny_RFB-Epoch-34-Loss-0.0029347729379017103.pth
#model='spoof-Mb_Tiny_RFB-_Mb_Tiny_RFB'

echo 'ncnn path: ' $ncnn_build

# 尝试简化onnx，pytorch1.3的似乎不支持onnxsim
#python3 -m onnxsim $model'.onnx' $model'-sim.onnx'
#model=$model'-sim'

echo 'save to: ' 'weight/'$model

mkdir -p 'weight/'$model

${ncnn_build}/'tools/onnx/onnx2ncnn' $model'.onnx' 'weight/'$model'/classify.param' 'weight/'$model'/classify.bin'
${ncnn_build}/'tools/ncnn2mem' 'weight/'$model'/classify.param' 'weight/'$model'/classify.bin' \
 'weight/'$model'/classify.id.h' 'weight/'$model'/classify.h'

cp $model'.onnx' 'weight/'$model'/classify.onnx'