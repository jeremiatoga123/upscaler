import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def upscale_image(input_file, output_file, target_scale=4, model_name='RealESRGAN_x4plus'):
    try:
        model_path = f'{model_name}.pth'
        
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            print("Download models from:")
            print("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
            print("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth")
            return False
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        if 'anime' in model_name:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True if device == 'cuda' else False,
            device=device
        )
        
        print(f"Processing: {input_file}")
        
        img = cv2.imread(input_file, cv2.IMREAD_COLOR)
        original_height, original_width = img.shape[:2]
        print(f"Original size: {original_width}x{original_height}")
        
        num_passes = 1
        if target_scale > 4:
            num_passes = 2 if target_scale <= 8 else 3 if target_scale <= 12 else 4
        
        current_img = img
        current_scale = 1
        
        for pass_num in range(num_passes):
            print(f"Pass {pass_num + 1}/{num_passes}...")
            
            try:
                output, _ = upsampler.enhance(current_img, outscale=4)
                current_img = output
                current_scale *= 4
                
                if current_scale >= target_scale:
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("✗ Out of memory! Trying with tiling...")
                    upsampler.tile = 400
                    output, _ = upsampler.enhance(current_img, outscale=4)
                    current_img = output
                    current_scale *= 4
                else:
                    raise e
        
        if current_scale > target_scale:
            target_width = int(original_width * target_scale)
            target_height = int(original_height * target_scale)
            print(f"Resizing to exact {target_scale}x scale: {target_width}x{target_height}")
            current_img = cv2.resize(current_img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        print(f"Final size: {current_img.shape[1]}x{current_img.shape[0]}")
        
        cv2.imwrite(output_file, current_img)
        print(f"✓ Saved to: {output_file}")
        
        return True
        
    except ImportError:
        print("✗ Real-ESRGAN not installed!")
        print("Install with: pip install realesrgan")
        print("Also need: pip install basicsr")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python realesrgan_upscale.py <input_image>")
        print("Example: python realesrgan_upscale.py image.jpg")
        return
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"✗ Input file not found: {input_file}")
        return
    
    print("=== Real-ESRGAN Multi-Scale Upscale Script ===")
    print(f"Input file: {input_file}")
    print("\nAvailable scale options:")
    print("1. 4x upscale")
    print("2. 8x upscale")
    print("3. 12x upscale")
    print("4. 16x upscale")
    
    scale_choice = input("\nEnter scale choice (1-4): ").strip()
    scale_map = {'1': 4, '2': 8, '3': 12, '4': 16}
    
    if scale_choice not in scale_map:
        print("Invalid choice! Using default 4x scale.")
        target_scale = 4
    else:
        target_scale = scale_map[scale_choice]
    
    print(f"\nSelected scale: {target_scale}x")
    
    print("\nModel options:")
    print("1. RealESRGAN_x4plus (for photos)")
    print("2. RealESRGAN_x4plus_anime_6B (for illustrations/anime)")
    
    model_choice = input("Enter model choice (1 or 2, default: 1): ").strip()
    if model_choice == '2':
        model_name = 'RealESRGAN_x4plus_anime_6B'
    else:
        model_name = 'RealESRGAN_x4plus'
    
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_realesrgan_{target_scale}x.png"
    
    print(f"\nOutput file: {output_file}")
    
    success = upscale_image(input_file, output_file, target_scale, model_name)
    
    if success:
        print("\n✓ Upscaling completed successfully!")
        
        original_size = os.path.getsize(input_file) / (1024 * 1024)
        upscaled_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\nFile sizes:")
        print(f"Original: {original_size:.2f} MB")
        print(f"Upscaled: {upscaled_size:.2f} MB")

if __name__ == "__main__":
    main()