from utils import *
import imports 

class MRRes_Model():

	def __init__(self,datadir,epoch=10,batch_size=48):
		self.epoch 		= epoch
		self.batch_size = batch_size
		self._model()
		self._loadData()
		self._loadData(datadir)

		#Contain Accuracies for each model
		self.abnormalAccuracys = []	
		self.aclAccuracys	   = []
		self.meniscusAccuracys = []

		self.abnormal_acc = 0
		self.acl_acc      = 0
		self.meniscus_acc = 0

	def _model(self):
		Resnet = keras.applications.Xception(include_top=False, weights='imagenet', input_tensor=None, pooling='avg')
		for layer in Resnet.layers:
	  		layer.trainable= False

	  	Extra = Sequential()
		Extra.add(Dense(256, activation ='relu',kernel_constraint=keras.constraints.MaxNorm(max_value=2, axis=0)))
		Extra.add(Dense(256, activation ='relu',kernel_constraint=keras.constraints.MaxNorm(max_value=2, axis=0)))
		Extra.add(Dense(256, activation ='relu',kernel_constraint=keras.constraints.MaxNorm(max_value=2, axis=0)))
		Extra.add(Dense(1, activation ='sigmoid'))

		self.MRRes_Abnormal_axial = Sequential([
		    Resnet,
		    Extra ])

		self.MRRes_Abnormal_coronal = Sequential([
		    Resnet,
		    Extra ])

		self.MRRes_Abnormal_sagittal = Sequential([
		    Resnet,
		    Extra ])


		self.MRRes_Acl_axial = Sequential([
		    Resnet,
		    Extra ])


		self.MRRes_Acl_coronal = Sequential([
		    Resnet,
		    Extra ])


		self.MRRes_Acl_sagittal = Sequential([
		    Resnet,
		    Extra ])


		self.MRRes_Meniscus_axial = Sequential([
		    Resnet,
		    Extra ])

		self.MRRes_Meniscus_coronal = Sequential([
		    Resnet,
		    Extra ])

		self.MRRes_Meniscus_sagittal = Sequential([
		    Resnet,
		    Extra ])

		#sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

		self.MRRes_Abnormal_axial.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
		self.MRRes_Abnormal_coronal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
		self.MRRes_Abnormal_sagittal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])


		self.MRRes_Acl_axial.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
		self.MRRes_Acl_coronal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
		self.MRRes_Acl_sagittal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

		self.MRRes_Meniscus_axial.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
		self.MRRes_Meniscus_coronal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
		self.MRRes_Meniscus_sagittal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

	def _loadData(self,datadir):
		train_axial_dir	   	       = os.path.join(datadir,'train/axial') 
		train_coronal_dir  		   = os.path.join(datadir,'train/coronal')
		train_sagittal_dir 		   = os.path.join(datadir,'train/sagittal')

		self.train_abnormal_labels 	   = os.path.join(datadir,'train-abnormal.csv')
		self.train_acl_labels	  	   = os.path.join(datadir,'train-acl.csv')
		self.train_meniscus_labels 	   = os.path.join(datadir,'train-meniscus.csv')

		val_axial_dir 	 		   = os.path.join(datadir,'valid/axial')
		val_coronal_dir  		   = os.path.join(datadir,'valid/coronal')
		val_sagittal_dir 	       = os.path.join(datadir,'valid/sagittal')

		self.val_abnormal_labels 	   = os.path.join(datadir,'valid-abnormal.csv')
		self.val_acl_labels      	   = os.path.join(datadir,'valid-acl.csv')
		self.val_meniscus_labels 	   = os.path.join(datadir,'valid-meniscus.csv')


		self.train_axial_data 	  	   = os.listdir(train_axial_dir)
		self.train_coronal_data    	   = os.listdir(train_coronal_dir)
		self.train_sagittal_data  	   = os.listdir(train_sagittal_dir)
		
		self.train_abnormal_labels 	   = load_labels(self.train_abnormal_labels)
		self.train_acl_labels 	  	   = load_labels(self.train_acl_labels)
		self.train_meniscus_labels 	   = load_labels(self.train_meniscus_labels)

		self.val_abnormal_labels 	   = load_labels(self.val_abnormal_labels)
		self.val_meniscus_labels 	   = load_labels(self.val_meniscus_labels)
		self.val_acl_labels	    	   = load_labels(self.val_acl_labels)


		self.axial_abnormal        = data_gen(self.train_axial_data , self.train_abnormal_labels,train_axial_dir,self.axialt_length)
		self.coronal_abnormal  	   = data_gen(train_coronal_data,train_abnormal_labels,train_coronal_dir,coronalt_length)
		self.sagittal_abnormal 	   = data_gen(train_sagittal_data,train_abnormal_labels,train_sagittal_dir,saggitalt_length)

		self.axial_abnormal_val    = data_gen(val_axial_data , val_abnormal_labels,val_axial_dir,axialv_length)
		self.coronal_abnormal_val  = data_gen(val_coronal_data , val_abnormal_labels,val_coronal_dir,coronalv_length)
		self.sagittal_abnormal_val = data_gen(val_sagittal_data , val_abnormal_labels,val_sagittal_dir,saggitalv_length)

		##########################################################
		##########################################################

		self.axial_acl    	  	   = data_gen(train_axial_data , train_acl_labels,train_axial_dir,axialt_length)
		self.coronal_acl  	  	   = data_gen(train_coronal_data,train_acl_labels,train_coronal_dir,coronalt_length)
		self.sagittal_acl 	  	   = data_gen(train_sagittal_data,train_acl_labels,train_sagittal_dir,saggitalt_length)

		self.axial_acl_val         = data_gen(val_axial_data , val_acl_labels,val_axial_dir,axialv_length)
		self.coronal_acl_val       = data_gen(val_coronal_data , val_acl_labels,val_axial_dir,coronalv_length)
		self.sagittal_acl_val      = data_gen(val_sagittal_data , val_acl_labels,val_axial_dir,saggitalv_length)

		##########################################################
		##########################################################

		self.axial_meniscus    	   = data_gen(train_axial_data , train_meniscus_labels,train_axial_dir,axialt_length)
		self.coronal_meniscus  	   = data_gen(train_coronal_data,train_meniscus_labels,train_coronal_dir,coronalt_length)
		self.sagittal_meniscus 	   = data_gen(train_sagittal_data,train_meniscus_labels,train_sagittal_dir,saggitalt_length)

		self.axial_meniscus_val    = data_gen(val_axial_data , val_meniscus_labels,val_axial_dir,axialv_length)
		self.coronal_meniscus_val  = data_gen(val_coronal_data , val_meniscus_labels,val_axial_dir,coronalv_length)
		self.sagittal_meniscus_val = data_gen(val_sagittal_data , val_meniscus_labels,val_axial_dir,saggitalv_length)

		self.axialt_length    	   = len(train_axial_data)/batch_size + len(train_axial_data)%batch_size
		self.coronalt_length  	   = len(train_coronal_data)/batch_size + len(train_coronal_data)%batch_size
		self.saggitalt_length 	   = len(train_sagittal_data)/batch_size + len(train_sagittal_data)%batch_size


		self.axialv_length    	   = len(val_axial_data)/batch_size + len(val_axial_data)%batch_size
		self.coronalv_length  	   = len(val_coronal_data)/batch_size + len(val_coronal_data)%batch_size
		self.saggitalv_length 	   = len(val_sagittal_data)/batch_size + len(val_sagittal_data)%batch_size

	def train_abnormal(self):
		axial_abnormal_output  = self.MRRes_Abnormal_axial.fit_generator(axial_abnormal, epochs =epoch,steps_per_epoch=axialt_length,max_queue_size=axialt_length)
		axial_abnormal_valacc  = self.MRRes_Abnormal_axial.evaluate_generator(axial_abnormal_val,steps=axialv_length,verbose=1,max_queue_size=50,use_multiprocessing=False)
		self.abnormalAccuracys.append(axial_abnormal_valacc[1])

		print('\n===========================================================\n')
		coronal_abnormal_output = self.MRRes_Abnormal_axial.fit_generator(coronal_abnormal, epochs =epoch,steps_per_epoch= coronalt_length)
		coronal_abnormal_valacc = MRRes_Abnormal_axial.evaluate_generator(coronal_abnormal_val,steps=coronalv_length,verbose=1,max_queue_size=50,use_multiprocessing=False)
		self.abnormalAccuracys.append(coronal_abnormal_valacc[1])
		
		print('\n===========================================================\n')

		sagittal_abnormal_out   = MRRes_Abnormal_axial.fit_generator(sagittal_abnormal, epochs =epoch,steps_per_epoch= coronalt_length)
		sagittal_abnormal_valacc= MRRes_Abnormal_axial.evaluate(sagittal_abnormal_val,steps=saggitalv_length ,verbose=1,max_queue_size=50,use_multiprocessing=False)
		abnormalAccuracys.append(sagittal_abnormal_valacc[1])
		print('\n===========================================================\n')

	def train_acl(self):
		axial_acl_output = MRRes_Abnormal_axial.fit_generator(axial_acl, epochs =epoch,steps_per_epoch= axialt_length)
		axial_acl_valacc = MRRes_Abnormal_axial.evaluate_generator(axial_acl_val,steps=axialv_length,verbose=1,max_queue_size=50,use_multiprocessing=False)
		self.aclAccuracys.append(axial_acl_valacc[1]) 

		print('\n===========================================================\n')

		coronal_acl_out = MRRes_Abnormal_axial.fit_generator(coronal_acl, epochs =epoch,steps_per_epoch= coronalt_length)
		coronal_acl_valacc = MRRes_Abnormal_axial.evaluate_generator(coronal_acl_val,steps=coronalv_length,verbose=1,max_queue_size=50,use_multiprocessing=False)
		self.aclAccuracys.append(coronal_acl_valacc[1])

		print('\n===========================================================\n')

		sagittal_acl_out = MRRes_Abnormal_axial.fit_generator(sagittal_acl, epochs =epoch,steps_per_epoch= saggitalt_length)
		sagittal_acl_valacc = MRRes_Abnormal_axial.evaluate_generator(sagittal_acl_val,steps=saggitalv_length ,verbose=1,max_queue_size=50,use_multiprocessing=False)
		self.aclAccuracys.append(sagittal_acl_valacc[1])

	def train_Meniscus(self):
		axial_meniscus_out = MRRes_Abnormal_axial.fit_generator(axial_meniscus, epochs =epoch,steps_per_epoch= axialt_length)
		axial_meniscus_valacc = MRRes_Abnormal_axial.evaluate_generator(axial_meniscus_val,steps=axialv_length,verbose=1,max_queue_size=50,use_multiprocessing=False)
		self.meniscusAccuracys.append(axial_meniscus_valacc[1])

		print('\n===========================================================\n')

		coronal_meniscus_out = MRRes_Abnormal_axial.fit_generator(coronal_meniscus, epochs =epoch,steps_per_epoch= axialt_length)
		coronal_meniscus_valacc = MRRes_Abnormal_axial.evaluate_generator(coronal_meniscus_val,steps=coronalv_length,verbose=1,max_queue_size=50,use_multiprocessing=False)
		self.meniscusAccuracys.append(coronal_meniscus_valacc[1])

		print('\n===========================================================\n')

		sagittal_meniscus_out = MRRes_Abnormal_axial.fit_generator(sagittal_meniscus, epochs =epoch,steps_per_epoch= axialt_length)
		sagittal_meniscus_valacc = MRRes_Abnormal_axial.evaluate_generator(sagittal_meniscus_val,steps=saggitalv_length,verbose=1,max_queue_size=50,use_multiprocessing=False)
		self.meniscusAccuracys.append(sagittal_meniscus_valacc[1])

	def calculate_acc(self):
		abnormal_acc  = combine_accuracies(self.abnormalAccuracys)
		acl_acc 	  = combine_accuracies(self.aclAccuracys)
		meniscus_acc  = combine_accuracies(self.meniscusAccuracys)