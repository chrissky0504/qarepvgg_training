# Evaluate model (checkpoint path needed)
python eval_model.py --checkpoint /path/to/checkpoint.pt

# Run IJBC evaluation with ONNX model
python onnx_ijbc.py --model-root qarepvgg_b1_40.onnx --result-dir IJBC_result

# Run IJBC evaluation with pt model
python eval_ijbc.py --model-prefix qae20.pt --network qarepvgg_b1

# Benchmark ONNX model using TensorRT trtexec
/usr/src/tensorrt/bin/trtexec   --onnx=qarepvgg_b1_40.onnx   --fp16   --memPoolSize=workspace:16384   --iterations=100   --avgTiming=100

# Training
python train_v2.py configs/my_qarepvgg_run.py 

