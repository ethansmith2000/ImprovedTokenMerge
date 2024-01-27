# ImprovedTokenMerge
![compare3.png](compare3.png)
![GEuoFn1bMAABQqD](https://github.com/ethansmith2000/ImprovedTokenMerge/assets/98723285/82e03423-81e6-47da-afa4-9c1b2c1c4aeb)

heavily inspired by https://github.com/dbolya/tomesd by @dbolya, a big thanks to the original authors.

This project aims to adress some of the shortcomings of Token Merging for Stable Diffusion. Namely consistenly faster inference without quality loss.
I found with the original that you would have to use a high merging ratio to get really any speedups at all, and by then quality was tarnished. Benchmarks here: https://github.com/dbolya/tomesd/issues/19#issuecomment-1507593483



I propose two changes to the original to solve this.
1. Merging Method
   - the original calculates a similarity matrix of the input tokens and merges those with highest similarity
   - an issue here is that similarity calculation is O(n2) time, for ViT where token merging was proposed, you only had to do this a few times so it was quite efficient
   - here it needs to be done at every step, and the computation ends up being nearly as costly as attention itself
   - We can leverage a simple obsevation that nearby tokens tend to be similar to each other.
   - therefore we can merge tokens via downsampling which is very cheap and seems to be a good approximation
   - this can be analogized to grid-based subsampling of an image when using a nearest-neighbor downsample method, this is similar to what DiNAT (dilated neigborhood attention) does except for the fact we are still making use of global context
2. Merge Targets
   - the original merges the input tokens to attention, and then "unmerges" the resulting tokens to the original size
   - this operation seems to be quite lossy
   - instead i propose simply downsampling keys/values of the attention operation. both the QK calculation and QK * V can still drastically be reduced from the typical O(n2) scaling of attention, without needing to unmerge anything
   - queries are left fully intact, they just attend more sparsely to the image
   - attention for images, especially at larger resolutions, seems to be very sparse in general (QK matrix is low rank) so it does not appear that we lose too much from this

putting this altogether we can get tangible speedups of ~1.5x at typical sizes like 768-1024 and up to 3x and beyond at 1536 to 2048 range, in combination with flash attention


# Setup 🛠
```
pip install -r requirements.txt
```

# Inference 🚀
See the provided notebook, or gradio demo which you can run with python app.py


