import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class myVggnet(nn.Module):
	def __init__(self, vggnet):
		super(myVggnet, self).__init__()
		self.vggnet = vggnet

	# def forward(self, img):
	# 	x = img.unsqueeze(0)

	# 	x = self.vggnet.features(x)
	# 	x = x.view(x.size(0), -1)
	# 	fc = self.vggnet.classifier[0](x)
	# 	return fc

	def forward(self, img):
		x = img.unsqueeze(0)

		x = self.vggnet.features(x)
		fc = x.mean(3).mean(2).squeeze()
		return fc

