## Usage

To run the face comparison tool, use the following command:


```
python compare.py images/your_image.png
```

This will compare the face in `images/your_image.png` with the face in `images/og_img.png` and print the similarity metrics between the two faces.

The cosine similarities between the two faces are stored in `cosine_similarities.txt` in numerical order of the image numbers.

## Requirements

The face comparison tool requires the following dependencies:

- `facenet-pytorch`
- `torch`
- `torchvision`
- `Pillow`

You can install the dependencies using pip:

```
pip install -r requirements.txt
```


