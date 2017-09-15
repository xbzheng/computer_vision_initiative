"""
work in progress as 08/18/2017
"""

# This Python 2 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os, sys

from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature

from skimage.segmentation import clear_border

from skimage import data

from scipy import ndimage as ndi
import matplotlib.pyplot as plt

import dicom
import scipy.misc
import numpy as np

from numpy.lib.stride_tricks import as_strided as ast
from itertools import product


def read_dicom(DicomPath, Resolution = []):
    
        # Read the slices from the dicom file
        dcmData = [dicom.read_file(DicomPath + filename) for filename in os.listdir(DicomPath)]
        
        # Sort the dicom slices in their respective order
        dcmData.sort(key=lambda x: int(x.InstanceNumber))
        
        # Get Slice Thickness
        if ("SliceLocation" in dcmData[0]):
            SliceThickness = np.abs(dcmData[0].SliceLocation  - dcmData[1].SliceLocation)
        elif("ImagePositionPatient" in dcmData[0]):
            SliceThickness = np.abs(dcmData[0].ImagePositionPatient[2] - dcmData[1].ImagePositionPatient[2])
        elif ("SliceThickness" in dcmData[0]):
            SliceThickness = dcmData[0].SliceThickness
        else:
            SliceThickness = np.nan
            
        # Get the pixel values for all the slices
        slices = np.stack([s.pixel_array for s in dcmData])
        
        slices[slices == slices.min()] = 0
        
        # Scale the Slice Data to HU
        slices = ((slices.astype(np.float64) * dcmData[0].RescaleSlope).astype(np.int16) + dcmData[0].RescaleIntercept).astype(np.int16) 
        
        # Resize if requested
        if( len(Resolution) == 3):
            slices, Resolution = resample(slices, np.array([SliceThickness] + dcmData[0].PixelSpacing,np.float64), Resolution)
        else:
            Resolution = ['','','']
            
        # Get Header Info
        dcmData = dcmData[0]
        
        dcmData = [dcmData.PatientID, dcmData.Rows, dcmData.Columns, len(dcmData) , 
                     dcmData.PixelSpacing[0], dcmData.PixelSpacing[1], SliceThickness,
                     Resolution[0], Resolution[1], Resolution[2],
                     dcmData.RescaleSlope, dcmData.RescaleIntercept, dcmData.BitsStored,
                     dcmData.HighBit, dcmData.PixelRepresentation, dcmData.ImageOrientationPatient,
                     dcmData.SpecificCharacterSet, dcmData.SOPClassUID, dcmData.SOPInstanceUID,
                     dcmData.StudyInstanceUID, DicomPath]
        
        return slices, dcmData
    
def read_dicom_hdr(DicomPath):
       
        #print DicomPath
        
        # Read the slices from the dicom file
        dcmData = [dicom.read_file(DicomPath + filename,stop_before_pixels=True) for filename in os.listdir(DicomPath)]
        
        # Sort the dicom slices in their respective order
        dcmData.sort(key=lambda x: int(x.InstanceNumber))

        # Get Slice Thickness
        if ("SliceLocation" in dcmData[0]):
            SliceThickness = np.abs(dcmData[0].SliceLocation  - dcmData[1].SliceLocation)
        elif("ImagePositionPatient" in dcmData[0]):
            SliceThickness = np.abs(dcmData[0].ImagePositionPatient[2] - dcmData[1].ImagePositionPatient[2])
        elif ("SliceThickness" in dcmData[0]):
            SliceThickness = dcmData[0].SliceThickness
        else:
            SliceThickness = np.nan
            
        dcmData = dcmData[0]
        
        return [dcmData.PatientID, dcmData.Rows, dcmData.Columns, len(dcmData) , 
                         dcmData.PixelSpacing[0], dcmData.PixelSpacing[1], SliceThickness, 
                         '', '', '', 
                         dcmData.RescaleSlope, dcmData.RescaleIntercept, dcmData.BitsStored,
                         dcmData.HighBit, dcmData.PixelRepresentation, dcmData.ImageOrientationPatient,
                         dcmData.SpecificCharacterSet, dcmData.SOPClassUID, dcmData.SOPInstanceUID,
                         dcmData.StudyInstanceUID, DicomPath]

def resample(image, curRes, newRes=[1,1,1]):
    
    resize_factor = curRes / newRes
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    newRes = curRes / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, newRes


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image.astype(np.int8)


    

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a

    from http://www.johnvinyard.com/blog/?p=268
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)


    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError('ws cannot be larger than a in any dimension. a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)

def LoadLungData(DicomPath, labelsPath, nSize = [1,229,229], nRes = [0.6,0.6,0.6]):

    # Get Dicom Folders
	folders = os.listdir(DicomPath)
	folders = folders[1:3]
    
	ImgInfo = []
	ImgData = []
	ImgLabel = []

	# Stride 
	stride = nSize

	# Load in the label data
	Labels = pd.read_csv(labelsPath)
	N = len(folders)
	# Get subset of folders to explore
	for i, folder in enumerate(folders):
		lab = Labels[(Labels['id']==folder)]['cancer'].values
		if  (folder.startswith('.')) or (len(lab)==0):
			print ("{0} of {1}) Skipping: {2}.\t(Label Not Found)".format(i,N, folder))
			continue
		#print i, N

		print ("{0} of {1}) Loading Data: {2}".format(i,N, folder))
        # print ('{} of {}) Loading Data: {}'.format(i,N, folder))
		Img, dcmHdr = (read_dicom(DicomPath + folder + '/',nRes ))
		del dcmHdr
		#print Img.shape

		print ('\tSegmenting')
		# Segment Lung
		segmented_lung = segment_lung_mask(Img, True)
		print ('\tDone segmenting')
		# Crop Lung
		props = regionprops(segmented_lung, intensity_image=Img, cache=True)
		min_row = props[0].bbox[1]
		max_row = props[0].bbox[4]

		min_col = props[0].bbox[2]
		max_col = props[0].bbox[5]

		min_slice = props[0].bbox[0]
		max_slice = props[0].bbox[3]

		# Segment the Mask and Image
		segmented_lung = segmented_lung[min_slice:max_slice, min_row:max_row,min_col:max_col].astype(np.int8)
		Img = (Img[min_slice:max_slice, min_row:max_row,min_col:max_col]*segmented_lung).astype(np.int16)
		del segmented_lung, min_row, max_row, min_col, max_col, min_slice, max_slice, props
		#print Img.shape, type(Img[0,0,0])

		if(len(nSize)>0):
			print ('\tSizing Image')

			if( Img.shape[1]<nSize[1] ):
				padding = nSize[1]-Img.shape[1]
				Img = np.pad(Img, ((0,0),(0,padding),(0,0)),'constant',constant_values=((0,0),(0,0),(0,0)) )

			if( Img.shape[2]<nSize[2] ):
				padding = nSize[2]-Img.shape[2]
				Img = np.pad(Img, ((0,0),(0,0),(0,padding)),'constant',constant_values=((0,0),(0,0),(0,0)) )

			Img = sliding_window(Img,ws = nSize ,ss=stride, flatten = True)
			#print Img.shape, type(Img[0,0,0]) 

		# Store Results
		#ImgInfo.append(dcmHdr)
		ImgData.append(Img)
		ImgLabel.extend(lab*np.ones((Img.shape[0]),))
		print ("\tMemory Usage (Bytes) %0.2f"%(sys.getsizeof(ImgData)))


	# convert to pandas for display and summary stats
	ImgInfo = pd.DataFrame(ImgInfo , columns=['PatientID', 'Rows', 'Columns', 'Slices' , 
	                    'xRes','yRes','zRes', 'nxRes','nyRes','nzRes','Rescale Slope','Rescale Intercept','Bit Depth',
	                    'HighBit', 'PixelRepresentation','ImageOrientation',
	                    'SpecificCharacterSet', 'SOPClassUID', 'SOPInstanceUID',
	                    'StudyInstanceUID', 'DicomPath'])
	
	# Turn into numpy arrays
	ImgData = np.vstack(ImgData).astype(np.float32)
	ImgLabel = np.array(ImgLabel).astype(np.int8)

	# # Normalize data
	# print "\tNormalizing Data"
	# ImgMin = ImgData.min()
	# ImgMax = ImgData.max()
	# ImgData = (ImgData-ImgMin)/(ImgMax-ImgMin)

	return ImgData, ImgLabel, ImgInfo
