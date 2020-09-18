from torch.utils.data import DataLoader,Dataset
import os
import cv2 as cv
from torchvision.transforms import Compose, ToTensor
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import json


class CaptchaData(Dataset):
	def __init__(self, DirPath,):
		super(Dataset, self).__init__()
		self.Samples=self.getSamples(DirPath)
		self.transfer=transforms.Compose([transforms.ToTensor(),])

	def getSamples(self,dir):
		Samples=[]
		imgPaths=os.listdir(dir)
		for imgPath in imgPaths:
			
			if(imgPath.find(".jpg")>0):
				#ImgPaths.append(os.path.join(dir,imgPath))
				#LabelPaths.append(os.path.join(dir,imgPath.replace("jpg","json")))
				Samples.append([os.path.join(dir,imgPath),os.path.join(dir,imgPath.replace("jpg","json"))])
		return Samples

	def __len__(self):
		return len(self.Samples)
	

	def parase(self,labelPath,size):
		JsonStruct=json.load(open(labelPath))
		Shapes=JsonStruct['shapes']
		bg=np.zeros(size).astype('uint8')
		cc=[]
		for Shape in Shapes:
		#Polos=JsonStruct['shapes'][0]['points']
			H=JsonStruct['imageHeight']
			W=JsonStruct['imageWidth']
			#img=cv.imread("CSU3F689A_0000000144_20190827_160159_866.jpg")
			#img=cv.resize(img,(256,256))
			Polos=Shape['points']
			Contours=[]
			for points in Polos:
				Contours.append((int(points[0]/W*128),int(points[1]/H*128)))
				#cv.circle(img,(int(points[0]/W*256),int(points[1]/H*256)),3,(255,255,255),2)
			cc.append(np.array(Contours))
		cc=np.array(cc)
		#Contours=np.array(Contours)
		for i in range(cc.shape[0]):
			cv.drawContours(bg,cc,i,1,-1)
		#cv.imwrite("draw.jpg",bg)
		return bg.astype('float64')

	def __getitem__(self, index):
		imgPath=self.Samples[index][0]
		labelPath=self.Samples[index][1]
		img=Image.open(imgPath).resize((128,128))
		label=self.parase(labelPath,img.size)

		img=self.transfer(img)
		return img,label

#TrainData=CaptchaData("D:\\UnetData\\Img\\")
#DataLoader(TrainData,1,4,drop_last=False)
#for img,label in TrainData:
#	print(img,label)