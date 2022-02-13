from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import torch

class LOGGER():
	def __init__(self):
		self.m_tensor_writer = None
		self.m_io_writer = None

	def f_add_writer(self, args):
		myhost = os.uname()[1]
		file_time = datetime.datetime.now().strftime('%m_%d_%H_%M')
		data_name = args.data_name
		output_file = myhost+"_"+file_time+str(data_name)

		# print("output_file", output_file)
	
		output_dir = "../log/"+args.data_name+"_"+args.model_name+"/"
		print("output_dir", output_dir)
		
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		output_file = os.path.join(output_dir, output_file)
		print("output_file", output_file)
		self.m_io_writer = open(output_file, "w")

		if not args.parallel:
			tensor_file_name = myhost+"_"+file_time
			print("tensor_file_name", tensor_file_name)
		
			self.m_tensor_writer = SummaryWriter("../tensorboard_hcdmg1/"+args.model_name+"/"+tensor_file_name)

		self.f_add_output2IO("="*10+"parameters"+"="*10)
		for attr, value in sorted(args.__dict__.items()):
			self.m_io_writer.write("{}={}\n".format(attr.upper(), value))
		self.f_add_output2IO("="*10+"parameters"+"="*10)

	def f_add_output2IO(self, msg):
		print(msg)
		self.m_io_writer.write(msg+"\n")
		# print("here")
		self.m_io_writer.flush()

	def f_add_scalar2tensorboard(self, scalar_name, scalar, index):
		self.m_tensor_writer.add_scalar('./data/'+scalar_name, scalar, index)

	def f_add_histogram2tensorboard(self, name):
		print("----"*10)

	def f_close_writer(self):
		self.f_close_IOwriter()
		self.f_close_tensorwriter()

	def f_close_IOwriter(self):
		self.m_io_writer.close()

	def f_close_tensorwriter(self):
		self.m_tensor_writer.close()


