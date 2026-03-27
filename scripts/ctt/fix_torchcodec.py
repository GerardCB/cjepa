"""Fix all torchcodec imports across the cjepa codebase.

torchcodec requires specific CUDA versions (libnvrtc.so) that may not
be available on all GPU instances. This script patches all imports to
gracefully fallback when torchcodec is not available.

Usage: python scripts/ctt/fix_torchcodec.py
"""
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
ROOT = os.path.abspath(ROOT)

patched = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    for fname in filenames:
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(dirpath, fname)
        with open(fpath, 'r') as f:
            content = f.read()
        if 'from torchcodec' not in content:
            continue

        new_content = re.sub(
            r'^(\s*)from torchcodec\.decoders import VideoDecoder\s*$',
            r'''\1try:
\1    from torchcodec.decoders import VideoDecoder
\1except (ImportError, ModuleNotFoundError, OSError):
\1    VideoDecoder = None''',
            content,
            flags=re.MULTILINE
        )

        if new_content != content:
            with open(fpath, 'w') as f:
                f.write(new_content)
            print(f'PATCHED: {fpath}')
            patched += 1
        else:
            print(f'SKIPPED (already patched): {fpath}')

print(f'\nDone! Patched {patched} files.')
