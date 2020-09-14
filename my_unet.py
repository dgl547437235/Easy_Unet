from dataload import CaptchaData
import torch
import torch.nn as nn
from Model import U_Net
from torch.utils.data import DataLoader,Dataset
import torchvision.utils as vutils
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p

model=U_Net(3,1)
model.load_state_dict(torch.load("unet.pt"))
model=model.cuda()
TrainData=CaptchaData("D:\\UnetData\\img\\")
dataload=DataLoader(TrainData,38,4,drop_last=False)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)#定义优化器
criterion = nn.BCEWithLogitsLoss()
for epoch in range(3000):
	for img,label in dataload:
		img=img.cuda()
		label=label.cuda()
		pred=model(img)
		optimizer.zero_grad()
		loss=calc_loss(pred,label.view(-1,1,256,256))
		loss.backward()
		optimizer.step()

	if(epoch%10==0):
		print(loss)
		pred=pred*255
		vutils.save_image(img.data,"img.jpg",)
		vutils.save_image(pred.data,"Result.jpg",)
		vutils.save_image(label.view(-1,1,256,256).data,"Label.jpg",)
		torch.save(model.state_dict(),"unet.pt")
		x=torch.rand(1,3,256,256,device="cuda")
		torch.onnx._export(model, x,"unet.onnx",export_params=True)

