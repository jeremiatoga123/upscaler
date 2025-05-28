# Real-ESRGAN Multi-Scale Upscaler

Memperbesar gambar menggunakan Real-ESRGAN dengan berbagai pilihan skala (4x, 8x, 12x, 16x)

## Instalasi

### Langkah 1: Install Python
Install Python 3.10 dari file exe yang ada di folder ini.

### Langkah 2: Install PyTorch dengan CUDA Support
Pertama, install PyTorch dengan CUDA 11.8:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### Langkah 3: Install Package Lainnya
Kemudian install dependencies lainnya:
```bash
pip install -r requirements.txt
```

### Langkah 4: Download File Model
Download file model dan simpan di folder yang sama dengan script:

- **Untuk Foto**: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
- **Untuk Anime/Ilustrasi**: [RealESRGAN_x4plus_anime_6B.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)

## Cara Penggunaan

Jalankan script dengan file gambar Anda:
```bash
python upscaler.py gambar.jpg
```

Script akan menanyakan pilihan:
1. **Faktor skala**: 4x, 8x, 12x, atau 16x
2. **Tipe model**: Foto atau Anime/Ilustrasi

## Contoh
```bash
python upscaler.py foto.jpg
```
Output: `foto_realesrgan_4x.png`

## Persyaratan
- Python 3.10
- GPU NVIDIA (disarankan) atau CPU
- Minimal 4GB VRAM untuk pemrosesan GPU
- File model (.pth) di direktori yang sama

## Catatan
- Format output selalu PNG untuk kualitas terbaik
- Skala lebih tinggi membutuhkan waktu proses dan memori lebih banyak
- Jika kehabisan memori GPU, script akan otomatis beralih ke pemrosesan tile