# Third-party notices

## Vendored code

**PointNet** (`sem_navae/models/pointnet.py`), adapted from
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch):

```
MIT License

Copyright (c) 2019 benny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Models and libraries used at run time (not redistributed)

| Component | Used for | License |
|-----------|----------|---------|
| [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) weights `CIDAS/clipseg-rd64-refined`, via [transformers](https://github.com/huggingface/transformers) | Open-vocabulary segmentation in the selector (downloaded on first use) | Apache-2.0 (transformers); see the model card for the weights |
| [elevation_mapping](https://github.com/ANYbotics/elevation_mapping), [grid_map](https://github.com/ANYbotics/grid_map) | Elevation map for the collision filter | BSD-3-Clause |
| [PyTorch](https://pytorch.org) | Models and training | BSD-3-Clause |
| ROS 1 Noetic packages | Robot integration | BSD-3-Clause and others |
