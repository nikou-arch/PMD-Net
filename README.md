### Train
```
python train.py --rate=10 --batch_size=32
```

The training dataset path is "../dataset/BSD400". Model and log will be saved in the trained_models folder.
### Test
```
python test.py --rate=10
```

The reconstructed_images and the corresponding PSNR and SSIM metrics will be saved in the reconstructed_images folder.
