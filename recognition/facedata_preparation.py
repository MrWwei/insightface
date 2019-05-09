# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:51:18 2019

@author: yusheng xu
"""

import os
import shutil
import numpy
import cv2
import face_alignment
from xml.dom import minidom

from resizeimage import resizeimage
from PIL import Image

# Filter all folds
def filterAllFolds(input_fold_path, output_fold_path, min_files_num):
            
    input_allfiles_names= os.listdir(input_fold_path) #得到根文件夹下的所有文件名称
    valid_ID_infile_pathnames=[]
    valid_ID_outfile_pathnames=[] 
    
    for fold_name in input_allfiles_names: 
        
         if not os.path.isdir(fold_name): #判断是否是文件夹，是文件夹才打开， 即对应一个人的ID文件夹
             
             if not fold_name=="unknow": 
                
                 input_subfold_path = input_fold_path+fold_name
                 output_subfold_path= output_fold_path+fold_name
                 
                 if os.path.isdir(input_subfold_path): #判断是否是文件，是文件才打开a
                     
                     # Test
                     #print(input_subfold_path)
                                     
                     fold_files= os.listdir(input_subfold_path) #得到子文件夹下的所有文件名称
                     
                     fold_files_num = numpy.size(fold_files)
                     
                     # Test
                     #print(fold_files_num)
    
                     if fold_files_num >  min_files_num: # 筛选样本数目
                         
                         valid_infiles_pathname = []
                         valid_outfiles_pathname = []
                          
                         for idx, file_name in enumerate(fold_files):
                             
                             if os.path.splitext(file_name)[-1][1:] == "xml":
                                 
                                 file_name_nontype = os.path.splitext(file_name)[0]
                                 
                                 input_file_pathname = input_subfold_path+"/"+file_name_nontype
                                 output_file_pathname = output_subfold_path+"/"+file_name_nontype                               
                             
                                 valid_infiles_pathname.append(input_file_pathname)
                                 valid_outfiles_pathname.append(output_file_pathname)
                     
                         valid_ID_infile_pathnames.append([input_subfold_path,valid_infiles_pathname])
                         valid_ID_outfile_pathnames.append([output_subfold_path,valid_outfiles_pathname])
    
    return valid_ID_infile_pathnames, valid_ID_outfile_pathnames

# Convert residentID to faceID
def convertResidentIDtoFaceID(list_file_residentID):
    list_file_faceID = []
    list_residentID_nonduplicated= list(set(list_file_residentID))
    list_residentID_nonduplicated.sort(key=list_file_residentID.index)
                         
    for resident_ID in list_file_residentID:
        face_ID=0
        for resident_ID_nonduplicated in list_residentID_nonduplicated:
            face_ID=face_ID+1;
            if resident_ID == resident_ID_nonduplicated:
                list_file_faceID.append(face_ID)
                
    return list_file_faceID

# Parse XML file
def parseXMLfile(input_file_pathname):
     doc = minidom.parse(input_file_pathname)
     rootNode = doc.documentElement
     faceNode = rootNode.getElementsByTagName("face")[0]
     
     # Get all the attribute needed
     residentId = faceNode.getElementsByTagName("residentId")[0]
     faceAge = faceNode.getElementsByTagName("age")[0]
     faceGender = faceNode.getElementsByTagName("gender")[0]
     
     # Get the bounding box
     bboxNode = faceNode.getElementsByTagName("boundingBox")[0]
     left = bboxNode.getElementsByTagName("left")[0]
     right = bboxNode.getElementsByTagName("right")[0]
     top = bboxNode.getElementsByTagName("top")[0]
     bottom = bboxNode.getElementsByTagName("bottom")[0]
         
     return [residentId.childNodes[0].nodeValue, faceAge.childNodes[0].nodeValue, faceGender.childNodes[0].nodeValue, left, right, top, bottom]

# Parse XML file
def parseXMLfile2(input_file_pathname):
     doc = minidom.parse(input_file_pathname)
     rootNode = doc.documentElement
     faceNode = rootNode.getElementsByTagName("face")[0]
     
     # Get all the attribute needed
     residentId = faceNode.getElementsByTagName("residentId")[0]
     faceAge = faceNode.getElementsByTagName("age")[0]
     faceGender = faceNode.getElementsByTagName("gender")[0]
     
     # Get the bounding box
     bboxNode = faceNode.getElementsByTagName("boundingBox")[0]
     left = bboxNode.getElementsByTagName("left")[0]
     right = bboxNode.getElementsByTagName("right")[0]
     top = bboxNode.getElementsByTagName("top")[0]
     bottom = bboxNode.getElementsByTagName("bottom")[0]
         
     return [residentId.childNodes[0].nodeValue, faceAge.childNodes[0].nodeValue, faceGender.childNodes[0].nodeValue, left, right, top, bottom]

# Load image
def loadImageFile(input_img_pathname):
    # Read image
    image_mat = cv2.imread(input_img_pathname,cv2.IMREAD_UNCHANGED) 
    return image_mat 

# Save image
# resize image
def saveImageFile(output_img_pathname, output_subfolds_path, output_img_mat):   
    if not os.path.isdir(output_subfolds_path): 
         os.mkdir(output_subfolds_path)
     
    cv2.imwrite(output_img_pathname, output_img_mat, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    with open(output_img_pathname, 'r+b') as f:
        with Image.open(f) as image:
            size = image.size
            print(size)
            cover = resizeimage.resize_cover(image, [112, 112])
            cover.save(output_img_pathname, image.format)
    return

# Face image detection
def cropFaceBBox(input_face_mat, edge, left, right, top, bottom):  
    # Crop image
    #if (top - edge) < 1:
    #    top=edge  
    #if (left - edge) <1:
    #    left=edge  
    output_face_mat = input_face_mat[top:bottom, left:right]       
    return output_face_mat 

# Face landmarks extraction
def extractFaceLandmarks(input_face_mat, face_marker):
    #
    face_ornot=False
    rows=0
    cols=0
    face_points =[]
    
    # Get landmarks
    face_marks = face_marker.get_landmarks(input_face_mat)
    
    if not face_marks == None:
        face_ornot=True;
        face_points =face_marks[-1]
        face_points = numpy.array(face_points)
        pt_min_row =1000000000
        pt_max_row = 0
        pt_min_col =1000000000
        pt_max_col = 0
     
        for pt_preds in face_points:
            if (pt_preds[0]) > pt_max_col:
                pt_max_col=pt_preds[0]
            if (pt_preds[0]) < pt_min_col:
                pt_min_col=pt_preds[0]
            if (pt_preds[1]) > pt_max_row:
                pt_max_row=pt_preds[1]
            if (pt_preds[1]) < pt_min_row:
                pt_min_row=pt_preds[1]       
        # Test 
        #print(pt_min_col, pt_max_col, pt_min_row, pt_max_row)
        rows=pt_max_row-pt_min_row
        cols=pt_max_col-pt_min_col
        
    return face_ornot, face_points, rows, cols 

# Face transformation
def estimatePointsTransformation(source_points,target_points):
    # Affine estimation                                 
    transform_matrix = cv2.estimateAffinePartial2D(source_points,target_points,True)[0]
    return transform_matrix

# Face alignment
def alignFaceByMatrix(transform_matrix, input_img_mat, out_rows, out_cols):
    # Affine transformation                                       
    output_img_mat = cv2.warpAffine(input_img_mat, transform_matrix, (out_rows, out_cols))
    return output_img_mat

# Traverse all valid the folds
def traverseValidFolds(input_IDs_pathname, output_IDs_pathname, input_standard_pathname, face_marker):
    
    # Output variables
    list_file_Idx = []
    list_file_name = []
    list_file_residentID = []
    list_file_faceID = []
    list_file_gender = []
    list_file_age = []
    
    # Process standard face
    standard_face_mat = cv2.imread(input_standard_pathname) 
    [standard_face_ornot, standard_face_marks, standard_rows, standard_cols] = extractFaceLandmarks(standard_face_mat, face_landmarker)

    # Process all valid fold
    for  idx_ID, ID_data in enumerate(input_IDs_pathname):
        
        [ID_folds, ID_allfiles] = ID_data
                        
        for idx_img, ID_file in enumerate(ID_allfiles):
        
            # Read image file
            input_img_pathname =ID_file +".png"
            output_subfolds_path = output_IDs_pathname[idx_ID][0]
            
            # Test 
            print(output_IDs_pathname[idx_ID][1][idx_img])
            
            output_img_pathname =output_IDs_pathname[idx_ID][1][idx_img] +".png"
            output_xml_pathname =output_IDs_pathname[idx_ID][1][idx_img] +".xml"  
            
            face_all_mat = loadImageFile(input_img_pathname)
        
            # Read XML file
            input_xml_pathname =ID_file +".xml"
            [residentId, faceAge, faceGender, left, right, top, bottom] = parseXMLfile(input_xml_pathname)
            
            if len(left.childNodes)>0 and len(top.childNodes)>0 and len(right.childNodes)>0 and len(bottom.childNodes)>0:
                left=int(left.childNodes[0].nodeValue)
                right=int(right.childNodes[0].nodeValue)
                top=int(top.childNodes[0].nodeValue)
                bottom=int(bottom.childNodes[0].nodeValue)
                
                # Crop image file
                face_crop_mat=cropFaceBBox(face_all_mat, 0, left, right, top, bottom)
                
                # Extract face landmarks
                [test_face_ornot, test_face_marks, test_rows, test_cols] = extractFaceLandmarks(face_crop_mat, face_marker)
            
                if test_face_ornot:
                
                   # Face transformation
                   trans_matrix = estimatePointsTransformation(test_face_marks, standard_face_marks)
                
                   # Face alignment
                   face_crop_align_mat = alignFaceByMatrix(trans_matrix, face_crop_mat, standard_rows, standard_cols)
                
                   # Save aligned and cropped face
                   print(output_subfolds_path)
                   saveImageFile(output_img_pathname, output_subfolds_path, face_crop_align_mat)
                
                   # Copy XML file
                   shutil.copyfile(input_xml_pathname, output_xml_pathname)
                
                   # Save into the lists
                   list_file_Idx.append(idx_ID-1)
                   list_file_residentID.append(residentId)
                   list_file_gender.append(faceGender)
                   list_file_age.append(faceAge)
                   list_file_name.append(output_img_pathname)
    
    list_file_faceID = convertResidentIDtoFaceID(list_file_residentID)
          
    return [list_file_Idx, list_file_faceID, list_file_residentID, list_file_gender, list_file_age, list_file_name]

#rename files
def rename(algndir):
    for path, dirs, files in os.walk(algndir, followlinks=True):
        dirs.sort()
        files.sort()
        i = 0
        for fname in files:

            suffix = os.path.splitext(fname)[1].lower()
            if suffix == '.xml':
                i -= 1
            strr = path.split("/")

            s = "%05d" % i
            dst = strr[-1] + "_" + s + suffix
            src = path + "/" + fname
            dst = path + "/" + dst

            # rename() function will
            # rename all the files
            os.rename(src, dst)
            i += 1


### Parameters and settings

# Input paths
input_fold_path = "/home/heisai/Pictures/test/" #文件夹目录
input_standardface_pathname="/home/heisai/Pictures/standard.png" # 标准脸目录

# Output paths
output_path = "/home/heisai/disk/HeisAI_data/output/"
output_fold_path="/home/heisai/disk/HeisAI_data/new_data/"
output_all_name = "train.lst"
# output_train_name="train_info.txt"
# output_valid_name="valid_info.txt"

# Output parameters
check_thred=0.75;
min_num_faces=20;


### Main

# Create face landmarker
face_landmarker = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

# Select valid face IDs
valid_input_IDs, valid_output_IDs=filterAllFolds(input_fold_path, output_fold_path, min_num_faces)

# Traverse all valid face IDs
list_file_Idx, list_file_faceID, list_file_residentID, list_file_gender, list_file_age, list_file_name = traverseValidFolds(valid_input_IDs, valid_output_IDs, input_standardface_pathname, face_landmarker)

#rename(output_fold_path)
#Random group
numpy.random.shuffle(list_file_Idx)
list_train_idx, list_valid_idx = list_file_Idx[:round(check_thred*len(list_file_Idx))], list_file_Idx[round(check_thred*len(list_file_Idx)):]

# Generate img-label file
output_all_file= open (output_path+"/"+output_all_name, "w")
#output_train_file= open (output_path+"/"+output_train_name, "w")
#output_valid_file= open (output_path+"/"+output_valid_name, "w")
i = 0
for index in range(len(list_file_faceID)):
    #print(index)
    #print(list_file_faceID[index])
    output_all_file.write(str(1) + "\t" + str(list_file_name[index]) + "\t" + str(list_file_faceID[index]-1) + "\t" + str(list_file_age[index]) + "\t" + str(list_file_gender[index]) + "\n")
    #i+=1
# for train_idx in range(len(list_train_idx)):
#
#     index1=list_train_idx[train_idx]
#     output_train_file.write(str(list_file_name[index1])+" "+str(list_file_faceID[index1])+" "+str(list_file_age[index1])+" "+str(list_file_gender[index1])+"\n")
#
# for valid_idx in range(len(list_valid_idx)):
#
#     index2=list_valid_idx[valid_idx]
#     output_valid_file.write(str(list_file_name[index2])+" "+str(list_file_faceID[index2])+" "+str(list_file_age[index2])+" "+str(list_file_gender[index2])+"\n")


