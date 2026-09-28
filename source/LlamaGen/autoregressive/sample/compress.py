# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate, estimate_rate
from torchvision import transforms
from PIL import Image
import torch
import os 

def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Compression timing requires a CUDA-capable GPU.")
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 

    # Labels to condition the model with (feel free to change):
    # class_labels = [20,2,49]
    # c_indices = torch.tensor(class_labels, device=device)
    # qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]
# 
    # t1 = time.time()
    # index_sample = generate(
        # gpt_model, c_indices, latent_size ** 2,
        # cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
        # temperature=args.temperature, top_k=args.top_k,
        # top_p=args.top_p, sample_logits=True, 
        # )
    # sampling_time = time.time() - t1
    # print(f"gpt sampling takes about {sampling_time:.2f} seconds.")    
    
    # t2 = time.time()
    image = Image.open(args.image_path).convert('RGB')  # ensure RGB
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts [0,255] to [0,1] and HWC to CHW
    ])
    
    
    
    start_encoding = torch.cuda.Event(enable_timing=True)
    start_decoding = torch.cuda.Event(enable_timing=True)
    end_decoding = torch.cuda.Event(enable_timing=True)
    end_encoding = torch.cuda.Event(enable_timing=True)
    start_encoding.record()
    image_tensor = transform(image)  # shape: [3, H, W]

# 4. Optionally, add batch dimension

    x = image_tensor.unsqueeze(0).cuda()*2 - 1
    y,_,info = vq_model.encode(x)
    token_idx = info[2]

    start_decoding.record()
    rate = estimate_rate(
        model=gpt_model, tokens=token_idx, max_new_tokens=latent_size**2,
        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        )
    end_encoding.record()

   
      
    samples = vq_model.decode(y) # output value is between [-1, 1]
    # decoder_time = time.time() - t2
    # print(f"decoder takes about {decoder_time:.2f} seconds.")
  
   


    end_decoding.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    # Save and display images:
    encode_time = start_encoding.elapsed_time(end_encoding)
    decode_time = start_decoding.elapsed_time(end_decoding)
    os.makedirs(args.save_path, exist_ok=True)
    image_save_path = os.path.join(args.save_path, args.image_path.split('/')[-1])
    bpp = rate/(image_tensor.shape[-1]*image_tensor.shape[-2])
    import csv
    csv_file = os.path.join(args.save_path, 'rate.csv')
    write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["fn", "bpp", "encode_time", "decode_time"])  # write header
            writer.writerow([args.image_path.split('/')[-1], bpp, encode_time, decode_time])
    save_image(samples, image_save_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"image is saved to {image_save_path} with {rate/(image_tensor.shape[-1]*image_tensor.shape[-2])}bpp")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--image-path", type=str) 
    parser.add_argument("--save-path", type=str) 

    args = parser.parse_args()
    main(args)