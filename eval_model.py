import argparse
import os
import torch
import numpy as np
import onnxruntime
from eval.verification import load_bin, test
from backbones import get_model 

# Try to import tqdm for progress visualization
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def main(args):
    print(f"Loading checkpoint: {args.checkpoint}")
    
    if args.checkpoint.endswith('.onnx'):
        class ONNXWrapper:
            def __init__(self, model_path):
                self.options = onnxruntime.SessionOptions()
                self.options.log_severity_level = 4 
                
                # Verify CUDA provider availability
                available = onnxruntime.get_available_providers()
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                providers = [p for p in providers if p in available]
                print(f"[ONNX] Using providers: {providers}")
                
                self.session = onnxruntime.InferenceSession(
                    model_path, sess_options=self.options, providers=providers
                )
                self.input_cfg = self.session.get_inputs()[0]
                self.input_name = self.input_cfg.name
                self.input_shape = self.input_cfg.shape
                
                self.is_static_batch = False
                self.batch_size = 1
                self.use_fallback = False 

                if len(self.input_shape) > 0:
                    first_dim = self.input_shape[0]
                    # If batch dim is not a symbolic name (str) or -1, it's likely static
                    if isinstance(first_dim, int) and first_dim > 0:
                        self.is_static_batch = True
                        self.batch_size = first_dim
                    elif isinstance(first_dim, str):
                         # Usually 'batch_size' or 'None' means dynamic
                         pass
                    else:
                        # Defensive default for common fixed-batch exports
                        self.is_static_batch = True
                        self.batch_size = 1

            def __call__(self, x):
                blob = x.cpu().numpy().astype(np.float32)
                
                if self.use_fallback or (self.is_static_batch and blob.shape[0] != self.batch_size):
                    return self._fallback_run(blob)

                try:
                    outputs = self.session.run(None, {self.input_name: blob})
                    return torch.from_numpy(outputs[0]).cuda()
                except Exception as e:
                    print(f"\n[ONNX] Batch {blob.shape[0]} failed. Switching to iterative mode.")
                    self.use_fallback = True
                    return self._fallback_run(blob)

            def _fallback_run(self, blob):
                outs = []
                step = self.batch_size
                n = blob.shape[0]
                # If we are doing a lot of iterations, show a mini progress bar if tqdm is available
                iterable = range(0, n, step)
                if n > 1 and tqdm:
                    iterable = tqdm(iterable, desc="   Inferring", leave=False)
                
                for i in iterable:
                    batch_out = self.session.run(None, {self.input_name: blob[i:i+step]})
                    outs.append(torch.from_numpy(batch_out[0]))
                return torch.cat(outs).cuda()

            def eval(self): pass
            
        backbone = ONNXWrapper(args.checkpoint)
        # If model is static, force batch_size to 1 to avoid the 'stuck' feeling of internal looping
        if backbone.is_static_batch:
            print(f"Static batch size {backbone.batch_size} detected. Reducing batch_size.")
            args.batch_size = backbone.batch_size
    else:
        # Load the state dictionary first to detect deployment status
        state_dict = torch.load(args.checkpoint, map_location='cuda')
        actual_state_dict = state_dict.get('state_dict', state_dict)

        # Auto-detect if this is a reparameterized (deploy) model
        is_deploy = args.deploy
        if not is_deploy:
            for key in actual_state_dict.keys():
                if 'rbr_reparam' in key:
                    is_deploy = True
                    print("[PyTorch] Auto-detected reparameterized (deploy) model structure.")
                    break
        
        # Instantiate the backbone
        backbone = get_model(args.network, dropout=0, fp16=False, deploy=is_deploy).cuda()
        
        # If get_model didn't honor the deploy flag, manually attempt conversion
        if is_deploy:
            # Check if the model still has training-time layers (like rbr_dense)
            has_rbr_dense = any('rbr_dense' in n for n, _ in backbone.named_modules())
            if has_rbr_dense:
                print("[PyTorch] Model created in training mode. Attempting manual conversion to deploy mode...")
                # Search for the RepVGG backbone and convert it
                for module in backbone.modules():
                    if hasattr(module, 'switch_to_deploy'):
                        module.switch_to_deploy()
                    # Some implementations use different names or require calling conversion on the internal backbone
                    elif hasattr(backbone, 'backbone') and hasattr(backbone.backbone, 'switch_to_deploy'):
                        backbone.backbone.switch_to_deploy()
                        break

        backbone.load_state_dict(actual_state_dict)
        backbone.eval()

        class DeviceWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x.to(next(self.model.parameters()).device))

        backbone = DeviceWrapper(backbone)

    targets = ['lfw', 'cfp_fp', 'agedb_30']
    image_size = [112, 112]
    highest_acc_dict = {target: 0.0 for target in targets}

    for target in targets:
        bin_path = os.path.join(args.data_dir, f"{target}.bin")
        if not os.path.exists(bin_path):
            print(f"Skipping {target}: {bin_path} not found.")
            continue
            
        print(f"Loading {target}...")
        data_set = load_bin(bin_path, image_size)
        
        acc1, std1, acc2, std2, xnorm, embeddings_list = test(
            data_set, backbone, args.batch_size, nfolds=10
        )
        
        if acc2 > highest_acc_dict[target]:
            highest_acc_dict[target] = acc2
            
        print(f"[{target}] Accuracy-Flip: {acc2:.5f}+-{std2:.5f}, XNorm: {xnorm:.2f}")
        print(f"[{target}] Accuracy-Highest: {highest_acc_dict[target]:.5f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation on LFW, CFP_FP, AGEDB_30')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/ms1m-retinaface-t1', help='Path to verification bin files')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--network', type=str, default='qarepvgg_b1', help='Network backbone name')
    parser.add_argument('--deploy', action='store_true', help='Set to True if loading a reparameterized (deploy) model')
    
    args = parser.parse_args()
    main(args)
