import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from types import MethodType
from facenet_pytorch import MTCNN
# Desired dimensions
desired_width = 370
desired_height = 320
### helper function
def encode(img):
    res = resnet(torch.Tensor(img))
    return res

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces
### load model
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(
  image_size=(370,320), keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

### get encoded features for all saved images
saved_pictures =  r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\dataset\train\adults'
all_people_faces = {}
mtcnn = MTCNN()

count_no_face = 0
count_multiple_faces = 0

# Create a directory to save the cropped faces
cropped_faces_dir = os.path.join(saved_pictures, 'detect_faces')
no_faces_dir = os.path.join(saved_pictures, 'no_detect_faces')
if not os.path.exists(cropped_faces_dir):
    os.makedirs(cropped_faces_dir)
if not os.path.exists(no_faces_dir):
    os.makedirs(no_faces_dir)

# Iterate over images in the folder
for filename in os.listdir(saved_pictures):
    if filename.endswith('.jpg'):
        # Read the image
        img_path = os.path.join(saved_pictures, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            # Detect faces
            batch_boxes, _ = mtcnn.detect(img)
            if batch_boxes is not None and len(batch_boxes) == 1:
                # If only one face detected, crop it and save
                x1, y1, x2, y2 = map(int, batch_boxes[0])
                face_img = img[y1:y2, x1:x2]
                # Inside the loop where you process each image
                if face_img is not None and face_img.size != 0:  # Check if face_img is not None and not empty
                    # Perform resizing and other operations
                    face_img = cv2.resize(face_img, (desired_width, desired_height))
                    # Further processing or saving of the face image
                else:
                    # Handle the case where no face is detected
                    print(f"No face detected in {filename}. Original image saved.")
                    # Optionally resize the original image before saving
                    # img = cv2.resize(img, (desired_width, desired_height))
                    cv2.imwrite(os.path.join(no_faces_dir, filename), img)
                    count_no_face += 1
                # Check if the face image is not empty before saving
                if face_img.size != 0:
                    face_filename = f"{os.path.splitext(filename)[0]}_face.jpg"
                    face_path = os.path.join(cropped_faces_dir, face_filename)
                    cv2.imwrite(face_path, face_img)
                    print(f"Face cropped and saved as: {face_path}")
                else:
                    print(f"No face detected in {filename}. Original image saved.")
                    cv2.imwrite(os.path.join(no_faces_dir, filename), img)
                    count_no_face += 1
            else:
                # If no face or multiple faces detected, save the original image
                original_img_path = os.path.join(no_faces_dir, filename)
                #img = cv2.resize(img, (desired_width, desired_height))
                cv2.imwrite(original_img_path, img)
                print(f"No or multiple faces detected in {filename}. Original image saved as: {original_img_path}")
                if batch_boxes is None:
                    count_no_face += 1
                else:
                    count_multiple_faces += 1
        else:
            print(f"Error loading image: {img_path}.")

print("Number of images with no faces detected:", count_no_face)
print("Number of images with multiple faces detected:", count_multiple_faces)


'''
#detect specific image
count_not_detect = 0
# Iterate over images in the folder
for filename in os.listdir(saved_pictures):
    if filename.endswith('.jpg'):
        # Read the image
        img_path = os.path.join(saved_pictures, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            # Detect faces
            cropped_images = mtcnn(img)
            if cropped_images is not None:
                # Ensure images are in the correct format before saving
                if isinstance(cropped_images, list):
                    # Save each detected face as a separate image
                    for i, face in enumerate(cropped_images):
                        face_filename = f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
                        face_path = os.path.join(saved_pictures, face_filename)
                        # Convert tensor to numpy array and save as image
                        face_np = face.permute(1, 2, 0).detach().numpy()
                        # Scale values to [0, 255] range
                        face_np = (face_np * 255).astype('uint8')
                        # Convert color space to match original image and save
                        cv2.imwrite(face_path, cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))
                        print(f"Face {i + 1} saved as: {face_path}")
                else:
                    # Save the single detected face as a separate image
                    face_filename = f"{os.path.splitext(filename)[0]}_face.jpg"
                    face_path = os.path.join(saved_pictures, face_filename)
                    # Convert tensor to numpy array and save as image
                    face_np = cropped_images.permute(1, 2, 0).detach().numpy()
                    # Scale values to [0, 255] range
                    face_np = (face_np * 255).astype('uint8')
                    # Convert color space to match original image and save
                    cv2.imwrite(face_path, cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))
                    print(f"Face saved as: {face_path}")
            else:
                print(f"No faces detected in {img_path}.")
                count_not_detect = count_not_detect + 1
        else:
            print(f"Error loading image: {img_path}.")
print("not detect ",count_not_detect)
'''
### load model
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(
  image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

def detect(cam=0, thres=0.7):
    vdo = cv2.VideoCapture(cam)
    while vdo.grab():
        _, img0 = vdo.retrieve()
        batch_boxes, cropped_images = mtcnn.detect_box(img0)

        if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                x, y, x2, y2 = [int(x) for x in box]
                img_embedding = encode(cropped.unsqueeze(0))
                detect_dict = {}
                for k, v in all_people_faces.items():
                    detect_dict[k] = (v - img_embedding).norm().item()
                min_key = min(detect_dict, key=detect_dict.get)

                if detect_dict[min_key] >= thres:
                    min_key = 'Undetected'
                
                cv2.rectangle(img0, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                  img0, min_key, (x + 5, y + 10), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                
        ### display
        cv2.imshow("output", img0)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
'''
if __name__ == "__main__":
    detect(0)
'''