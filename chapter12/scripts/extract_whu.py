#!/usr/bin/env python3
"""Extract WHU zip directly to organized structure. Uses python zipfile."""
import zipfile
import os
from pathlib import Path

# 基于脚本所在目录自动定位
BASE = Path(__file__).resolve().parent.parent  # chapter12/
ZIP_PATH = str(BASE / 'whu_download' / 'WHU_aerial_0.3m.zip')
OUTPUT_DIR = str(BASE / 'whu_building')

def main():
    # Clean any partial extraction
    for d in ['train', 'val', 'test']:
        p = os.path.join(OUTPUT_DIR, d)
        if os.path.exists(p):
            import shutil
            shutil.rmtree(p)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Opening zip: {ZIP_PATH}", flush=True)
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        names = z.namelist()
        print(f"Total entries: {len(names)}", flush=True)
        
        # Find internal prefix
        internal_prefix = None
        for n in names:
            if n.endswith('/') and '/' not in n[:-1]:
                internal_prefix = n
                break
        print(f"Internal prefix: {internal_prefix}", flush=True)
        
        # Mapping: WHU split → target split
        splits = {'train': 'train', 'test': 'test', 'val': 'val'}
        
        for whu_split, target_split in splits.items():
            src_img_prefix = f"{internal_prefix}{whu_split}/image/"
            src_lbl_prefix = f"{internal_prefix}{whu_split}/label/"
            
            img_files = sorted([n for n in names if n.startswith(src_img_prefix) and not n.endswith('/')])
            lbl_files = sorted([n for n in names if n.startswith(src_lbl_prefix) and not n.endswith('/')])
            
            # Create output dirs
            img_out = os.path.join(OUTPUT_DIR, target_split, 'image')
            lbl_out = os.path.join(OUTPUT_DIR, target_split, 'label')
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)
            
            print(f"\n[{target_split}] {len(img_files)} images, {len(lbl_files)} labels", flush=True)
            
            # Extract directly to correct path (no renaming needed)
            for i, (img_n, lbl_n) in enumerate(zip(img_files, lbl_files)):
                img_name = os.path.basename(img_n)
                lbl_name = os.path.basename(lbl_n)
                
                # Extract to correct output path directly
                img_data = z.read(img_n)
                with open(os.path.join(img_out, img_name), 'wb') as f:
                    f.write(img_data)
                
                lbl_data = z.read(lbl_n)
                with open(os.path.join(lbl_out, lbl_name), 'wb') as f:
                    f.write(lbl_data)
                
                if (i + 1) % 500 == 0:
                    print(f"  [{target_split}] {i+1}/{len(img_files)} ...", flush=True)
    
    # Stats
    print("\n" + "=" * 50, flush=True)
    for split_name in ['train', 'val', 'test']:
        img_dir = os.path.join(OUTPUT_DIR, split_name, 'image')
        lbl_dir = os.path.join(OUTPUT_DIR, split_name, 'label')
        n_img = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
        n_lbl = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0
        print(f"  {split_name}/: {n_img} images, {n_lbl} labels", flush=True)
    
    print(f"\nDone! Data root: {OUTPUT_DIR}", flush=True)

if __name__ == '__main__':
    main()
