from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
import torch
from torchvision import utils
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
import argparse

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transforms.ToTensor()(image)
    image = TF.resize(image, (256, 256), interpolation=TF.InterpolationMode.BICUBIC).unsqueeze(0)
    return image

def extract_face(imgs, batch_boxes, mtcnn):
    image_size = imgs.shape[-1]
    faces = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        if batch_boxes[i] is not None:
            box = batch_boxes[i][0]
            margin = [
                mtcnn.margin * (box[2] - box[0]) / (160 - mtcnn.margin),
                mtcnn.margin * (box[3] - box[1]) / (160 - mtcnn.margin),
            ]

            box = [
                int(max(box[0] - margin[0] / 2, 0)),
                int(max(box[1] - margin[1] / 2, 0)),
                int(min(box[2] + margin[0] / 2, image_size)),
                int(min(box[3] + margin[1] / 2, image_size)),
            ]
            crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
        else:
            # crop_face = img[None, :, :, :]
            return None

        faces.append(F.interpolate(crop_face, size=160, mode='bicubic'))
    new_faces = torch.cat(faces)

    return (new_faces - 127.5) / 128.0

def get_faces(x, mtcnn):
    img = (x + 1.0) * 0.5 * 255.0
    img = img.permute(0, 2, 3, 1)
    with torch.no_grad():
        batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
        # Select faces
        batch_boxes, batch_probs, batch_points = mtcnn.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=mtcnn.selection_method
        )

    img = img.permute(0, 3, 1, 2)
    faces = extract_face(img, batch_boxes, mtcnn)
    return faces



def get_embedding(image, mtcnn, resnet, device, id):

    # Detect face and get face crop
    face = get_faces(image, mtcnn=mtcnn)
    
    if face is None:
        print(f"No face detected in the image")
        return None
    
    # Move face tensor to the same device as the model
    face = face.to(device)
    
    # Get embedding
    embedding = resnet(face)
    return embedding.detach()

def calculate_l1_distance(emb1, emb2):
    # Calculate L1 distance
    dist = F.l1_loss(emb1, emb2, reduction='none').mean(dim=1)
    return dist.item()

def calculate_cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1, emb2, dim=1).item()

def calculate_arcface_loss(emb1, emb2, s=64.0, m=0.5):
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    
    cos_similarity = F.cosine_similarity(emb1_norm, emb2_norm, dim=1)
    
    theta = torch.acos(torch.clamp(cos_similarity, -1.0 + 1e-7, 1.0 - 1e-7))
    
    arcface_logits = s * torch.cos(theta + m)
    
    arcface_loss = -torch.log(torch.exp(arcface_logits) / (torch.exp(arcface_logits) + 1))
    
    return arcface_loss.item()

def compare_embeddings(embedding1, embedding2, model_name):
    l1_distance = calculate_l1_distance(embedding1, embedding2)
    cosine_similarity = calculate_cosine_similarity(embedding1, embedding2)
    arcface_loss = calculate_arcface_loss(embedding1, embedding2)

    print(f"{model_name} Results:")
    print(f"L1 distance: {l1_distance:.4f}")
    print(f"Cosine similarity: {cosine_similarity:.4f}")
    print(f"ArcFace loss: {arcface_loss:.4f}")
    print()

def main(image1_path, image2_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    # Initialize MTCNN
    mtcnn = MTCNN(image_size=160, margin=0, device=device)

    # Initialize InceptionResnetV1 for both VGG-Face and CASIA-WebFace
    resnet_vggface = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet_casia = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

    # Load images
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # Get embeddings for VGG-Face
    embedding1_vggface = get_embedding(image1, mtcnn, resnet_vggface, device, "1")
    embedding2_vggface = get_embedding(image2, mtcnn, resnet_vggface, device, "2")

    # Get embeddings for CASIA-WebFace
    embedding1_casia = get_embedding(image1, mtcnn, resnet_casia, device, "1")
    embedding2_casia = get_embedding(image2, mtcnn, resnet_casia, device, "2")

    if embedding2_vggface is None or embedding2_casia is None:
        return None

    # Compare embeddings for VGG-Face
    compare_embeddings(embedding1_vggface, embedding2_vggface, "VGG-Face")

    # Compare embeddings for CASIA-WebFace
    compare_embeddings(embedding1_casia, embedding2_casia, "CASIA-WebFace")

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Compare two face images.')
        parser.add_argument('image', type=str, help='Path to the second image')
        args = parser.parse_args()
        image1 = "images/og_img.png"
        main(image1, args.image)

