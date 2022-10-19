#to extract frame, crop face and getting landmark points
from facenet_pytorch import MTCNN, InceptionResnetV1
from skimage import io
import os
import face_alignment
import subprocess
from PIL import Image
from shutil import copyfile
# from mtcnn import MTCNN
import cv2
import numpy

def extract_frames(input_path, name, outpath):
    cmd = "ffmpeg -i "+ input_path+name+".mov -vf \"select=not(mod(n\,5))\" -vsync vfr -q:v 2 "+outpath+name+'/'+name+"_%03d.png"
    subprocess.run([cmd], shell=True, executable = '/bin/bash')
    
def findvideos(inpfile, outpath):
    for dirpath, dirnames, filenames in os.walk('/home/shivali.chansoriya/data/sample/'):
        # print('searching for person', i)
        for filename in [f for f in filenames if f.endswith('.mov')]:
            filename = os.path.splitext(filename)[0]
            with open(inpfile) as trainlist:
                for person in trainlist:
                    person = person.strip('\n')
                    if filename == person:
                        # print(dirpath)
                        if not os.path.isdir(outpath+person+'/'):
                            os.mkdir(outpath+person+'/')                            
                        extract_frames(dirpath+'/',person, outpath)
                        # face_landmarks_crop(outpath)

        
def mtcnn_caller(path, image_folder,image):
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=256,margin=0)
    img = Image.open(path+image_folder+image)
    print('__taken input__', path+image_folder+image)
    faces,_= mtcnn.detect(img)
    if faces is not None:
        if img is not None:
            # Get cropped and prewhitened image tensor
            try:
                fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
                input = cv2.imread(path+image_folder+image)
                image = image.strip('.png')
                print('Input__', input)
                preds = fa.get_landmarks(input)
                image = image.strip('.png')
                os.mkdir(path+'spoof/'+image+'/')
                numpy.save(path+'spoof/'+image+'/'+image, preds)
                
                img_cropped = mtcnn(img, save_path=path+'spoof/'+image+'/'+image+'.png')
                
            except IndexError as error:
                print(error)

# def face_landmarks_crop(input_path):
#     for files in os.listdir(input_path)
#     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
#     input = io.imread('/home/shivalichansoriya/data/Paper/')
#     preds = fa.get_landmarks(input)
    
if __name__ =='__main__':

    inpfile = 'trainlist.txt'                        
    path= '/home/shivali.chansoriya/data/sample/train/'
    if not os.path.isdir(path):
        os.mkdir(path)

    findvideos(inpfile, path)
    image_folders = os.listdir(path)
    print(image_folders)
    for image_folder in image_folders:
        print(image_folder)
        if image_folder != 'spoof' and image_folder != 'live':
            images = os.listdir(path+image_folder)
            print(images)
            for image in images:
                print(image)
                print(path+'/'+image_folder+'/'+image)
                mtcnn_caller(path, image_folder+'/',image)