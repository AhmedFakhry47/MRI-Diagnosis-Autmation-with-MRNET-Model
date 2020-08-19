import imports 

def initiate_model():
	Resnet = keras.applications.Xception(include_top=False, weights='imagenet', input_tensor=None, pooling='avg')
	for layer in Resnet.layers:
  		layer.trainable= False

  	Extra = Sequential()
	Extra.add(Dense(256, activation ='relu',kernel_constraint=keras.constraints.MaxNorm(max_value=2, axis=0)))
	Extra.add(Dense(256, activation ='relu',kernel_constraint=keras.constraints.MaxNorm(max_value=2, axis=0)))
	Extra.add(Dense(256, activation ='relu',kernel_constraint=keras.constraints.MaxNorm(max_value=2, axis=0)))
	Extra.add(Dense(1, activation ='sigmoid'))

	MRRes_Abnormal_axial = Sequential([
	    Resnet,
	    Extra ])

	MRRes_Abnormal_coronal = Sequential([
	    Resnet,
	    Extra ])

	MRRes_Abnormal_sagittal = Sequential([
	    Resnet,
	    Extra ])


	MRRes_Acl_axial = Sequential([
	    Resnet,
	    Extra ])


	MRRes_Acl_coronal = Sequential([
	    Resnet,
	    Extra ])


	MRRes_Acl_sagittal = Sequential([
	    Resnet,
	    Extra ])


	MRRes_Meniscus_axial = Sequential([
	    Resnet,
	    Extra ])

	MRRes_Meniscus_coronal = Sequential([
	    Resnet,
	    Extra ])

	MRRes_Meniscus_sagittal = Sequential([
	    Resnet,
	    Extra ])

	sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

	MRRes_Abnormal_axial.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
	MRRes_Abnormal_coronal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
	MRRes_Abnormal_sagittal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])


	MRRes_Acl_axial.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
	MRRes_Acl_coronal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
	MRRes_Acl_sagittal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

	MRRes_Meniscus_axial.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
	MRRes_Meniscus_coronal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
	MRRes_Meniscus_sagittal.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])