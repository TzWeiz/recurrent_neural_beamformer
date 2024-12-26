import torch.nn as nn
import torch

class Conv2dBlock(nn.Module):
	def __init__(self, in_channels, n_kernels, kernel_size, padding=1, dilation=1, use_bn=True, activation=nn.ReLU):
		# pass
		super(Conv2dBlock, self).__init__()
		
		self.model = torch.nn.Sequential(
			nn.Conv2d(in_channels, n_kernels, kernel_size, padding=padding, dilation=dilation),
			nn.BatchNorm2d(n_kernels, affine=True),
			activation(inplace=True)
		)
		self.in_channels = in_channels
		self.out_channels = n_kernels

	def forward(self, x):
		return self.model.forward(x)

class Conv3dBlock(nn.Module):
	def __init__(self, in_channels, n_kernels, kernel_size, padding=1, dilation=1, use_bn=True, activation=nn.ReLU):
		# pass
		super(Conv3dBlock, self).__init__()
		
		self.model = torch.nn.Sequential(
			nn.Conv3d(in_channels, n_kernels, kernel_size, padding=padding, dilation=dilation),
			nn.BatchNorm3d(n_kernels, affine=True),
			activation(inplace=True)
		)
		self.in_channels = in_channels
		self.out_channels = n_kernels

	def forward(self, x):
		return self.model.forward(x)



class ResConn(nn.Module):
	def __init__(self, model):
		super(ResConn, self).__init__()
		self.model = model
		self.in_channels = self.model.in_channels
		self.out_channels = self.model.out_channels
		if not self.in_channels == self.out_channels: 
			self.nin = nn.Conv2d(self.in_channels, self.out_channels, 1)
		
	def forward(self, x):
		Identity_x = x.clone()
		if not self.in_channels == self.out_channels:
			Identity_x = self.nin(Identity_x)
		
		x = self.model.forward(x)
		x = Identity_x + x
		return x

class ReZero(nn.Module):
	def __init__(self, model):
		super(ReZero, self).__init__()
		self.model = model
		self.in_channels = self.model.in_channels
		self.out_channels = self.model.out_channels
		if not self.in_channels == self.out_channels: 
			self.nin = nn.Conv2d(self.in_channels, self.out_channels, 1)
		self.alpha = None

	def forward(self, x):
		Identity_x = x.clone()
		

		if not self.in_channels == self.out_channels:
			Identity_x = self.nin(Identity_x)
		
		x = self.model.forward(x)
		if self.alpha is None:
			self.alpha = torch.nn.Parameter(torch.zeros_like(x)[0:1,:,0:1])
		x = self.alpha * x
		x = Identity_x + x
		return x
