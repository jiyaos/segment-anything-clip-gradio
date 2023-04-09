## Segment Anything with CLIP Web UI

This is a gradio web ui for sam with clip. You can extract the specific object from the sam model with some text prompts. Also, it will show the best text prompt with score.
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [Contrastive Language-Image Pre-Training (CLIP)](https://github.com/openai/CLIP)


## Usaage 

1. Download the sam model from [here](https://github.com/facebookresearch/segment-anything#model-checkpoints)

2. Install dependencies:
```python
    pip install torch opencv-python Pillow
    pip install git+https://github.com/openai/CLIP.git
    pip install git+https://github.com/facebookresearch/segment-anything.git
```
3. Run the app.py
