import os
import cv2
import numpy as np
import face_recognition as fr

def getEncodedFaces(): #read img, encode, format
    encoded = {}

    for dirpath, dirnames, fnames in os.walk("./faces"): #walk directory and load img
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file(os.path.join("faces", f)) #load img
                encoding = fr.face_encodings(face)[0] #gives img in encoded format
                faceName = f.split(".")[0] #removes .png/.jpg
                encoded[faceName] = encoding #associate file name with encoded format

    return encoded


def classifyFace(im):
    faces = getEncodedFaces()
    facesEncoded = list(faces.values()) #gives us encoded face values
    knownFaceNames = list(faces.keys()) #all keys in dict = known names

    img = cv2.imread(im, 1)

    faceLocations = fr.face_locations(img) #pass formated img to face_locations to give location of faces
    unknownFaceEncodings = fr.face_encodings(img, faceLocations)#get encodings of faces to match with known faces

    faceNames = [] #puts located matched faces in array
    for faceEncoding in unknownFaceEncodings:
        matches = fr.compare_faces(facesEncoded, faceEncoding) #compare two faces & gives us array back indicating true or false match
        name = "Unknown"

        faceDistances = fr.face_distance(facesEncoded, faceEncoding) #gets distance between diff faces to later see which face is closes to known face
        bestMatchIndex = np.argmin(faceDistances) #uses min value from faceDistances index to give face with shortest possible distance
        if matches[bestMatchIndex]: #checks if we found match by comparing true/false of face as well as min euclidian distance
            name = knownFaceNames[bestMatchIndex]

        faceNames.append(name) #tells us how many faces we have in img even if we do not know the face


    for (top, right, bottom, left), name in zip(faceLocations, faceNames): #draw rect on coordinates of faces
        cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (0, 0, 255), 2)
        cv2.rectangle(img, (left-20, bottom-15), (right+20, bottom+20), (0, 0, 255), cv2.FILLED) #draw 5px high rect to write name
        font = cv2.FONT_HERSHEY_DUPLEX #text on filled rect
        cv2.putText(img, name, (left-20, bottom+15), font, 1.0, (255, 255, 255), 2)

    while True: #draw all onto screen
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return faceNames

if __name__ == "__main__":
    print(classifyFace("testImage.jpg"))