#!/bin/bash
# export http_proxy=http://gfw.in.zhihu.com:18080 && export https_proxy=http://gfw.in.zhihu.com:18080
pip install -e .

pip install -e ".[train]"
pip install flash-attn --no-build-isolation