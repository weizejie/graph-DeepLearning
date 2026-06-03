# -*- coding: utf-8 -*-
import sys, os, base64
sys.stdout.reconfigure(encoding='utf-8')

fp = r'c:\Users\weize jie\Desktop\test\graph DeepLearning\text for analysis\????????.md'

# Verify target
data = open(fp, 'rb').read()
print(f"Target file: {len(data)} bytes, non-ASCII: {len([b for b in data if b > 127])}")

# Base64 encoded content (Python 3 ascii string - survives Write tool)
b64_content = "77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/77u/..."