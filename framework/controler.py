from load import im_loaders
from load import loaders
from load import transforms
from load import writers
from load import preselectors

from models import models

from preprocess import preprocess

from train import augmentations
from train import trainers

from submit import submitters

from framework.datatools import Data
from framework.config import LOCS

import cPickle as pickle

# File locations
selector_train_file = LOCS.cache_dir + 'sel_train_{}.p'
selector_val_file = LOCS.cache_dir + 'sel_val_{}.p'
x_train_file = LOCS.cache_dir + 'X_train_{}.npy'
x_val_file = LOCS.cache_dir + 'X_validate_{}.npy'
y_train_file = LOCS.cache_dir + 'y_train_{}.npy'
ids_val_file = LOCS.cache_dir + 'ids_validate_{}.npy'

weights_file = LOCS.cache_dir + 'weights_{}.hdf5'
weights_best_file = LOCS.cache_dir + 'weights_best_{}.hdf5'
weights_systole_file = LOCS.cache_dir + 'weights_systole_{}.hdf5'
weights_diastole_file = LOCS.cache_dir + 'weights_diastole_{}.hdf5'
weights_systole_best_file = LOCS.cache_dir + 'weights_systole_best_{}.hdf5'
weights_diastole_best_file = LOCS.cache_dir + 'weights_diastole_best_{}.hdf5'

val_loss_file = LOCS.cache_dir + 'val_loss_{}.txt'

submission_file = LOCS.submissions_dir + 'submission_{}.csv'

def transform_wrapper(f_transform,**kwargs):
	def inner(im,meta):
		return f_transform(im,meta,**kwargs)
	return inner

def preprocess_wrapper(f_preprocess,**kwargs):
	def inner(im):
		return f_preprocess(im,**kwargs)
	return inner

def augmentations_wrapper(f_aug,**kwargs):
	def inner(im):
		return f_aug(im,**kwargs)
	return inner


def run_preselect(name,workflow):
	sub_workflow = workflow.get('load')

	train = Data(LOCS.train_dir)
	validate = Data(LOCS.validate_dir)

	# Get functions from workflow
	f_preselect = getattr(preselectors,sub_workflow.get('preselect',''),None)

	# Run 
	if f_preselect:
		f_preselect(
			train,
			selector_train_file.format(name),
			n_cons=sub_workflow.get('n_preselect',8))
		f_preselect(
			validate,
			selector_val_file.format(name),
			n_cons=sub_workflow.get('n_preselect',8))

def run_write(name,workflow):
	sub_workflow = workflow.get('load')

	train = Data(LOCS.train_dir)
	validate = Data(LOCS.validate_dir)

	# Get functions from workflow
	f_transforms = [
				transform_wrapper(getattr(transforms,f_transform),**kwargs) 
				for f_transform,kwargs in sub_workflow.get('transforms',[])
				]
	f_im_loader = getattr(im_loaders,sub_workflow.get('im_loader'))
	f_writer = getattr(writers,sub_workflow.get('writer'))

	# Run
	ids_train, samples_train = f_im_loader(
		train,
		f_transforms,
		selector=pickle.load(open(selector_train_file.format(name),'rb')) if sub_workflow.get('preselect') else None
		)
	ids_val, samples_val = f_im_loader(
		validate,
		f_transforms,
		selector=pickle.load(open(selector_val_file.format(name),'rb')) if sub_workflow.get('preselect') else None
		)

	f_writer(
		ids_train, 
		samples_train, 
		x_train_file.format(name), 
		studies_to_results = writers.map_studies_results(LOCS.train_csv),
		output_y = y_train_file.format(name))
	f_writer(
		ids_val, 
		samples_val, 
		x_val_file.format(name),
		output_ids = ids_val_file.format(name))

def run_load(name,workflow):
	sub_workflow = workflow.get('load')

	train = Data(LOCS.train_dir)
	validate = Data(LOCS.validate_dir)

	# Get functions from workflow
	f_loader = getattr(loaders,sub_workflow.get('loader'))

	#Run
	X_train,y_train = f_loader(
		x_train_file.format(name),
		y_file = y_train_file.format(name))
	X_val, ids_val = f_loader(
		x_train_file.format(name),
		ids_file = ids_val_file.format(name))

	return X_train,y_train,X_val,ids_val

def run_train(name,workflow,**params):
	train_workflow = workflow['train']

	X,y,_,_ = run_load(name,workflow)

	# Get functions from workflow
	print 'Getting model...'
	if workflow.get('model'):
		model = getattr(models,workflow.get('model'))(workflow.get('model_inputs'))
	else:
		model_systole = getattr(models,workflow.get('models')[0])(workflow.get('model_inputs'))
		model_diastole = getattr(models,workflow.get('models')[1])(workflow.get('model_inputs'))
	f_preprocess = [
		preprocess_wrapper(getattr(preprocess,f_preprocess),**kwargs) 
		for f_preprocess,kwargs in workflow.get('preprocess',[])]
	f_trainer = getattr(trainers,train_workflow.get('trainer'))
	f_augmentations = [
		augmentations_wrapper(getattr(augmentations,f_aug),**kwargs)
		for f_aug,kwargs in train_workflow.get('augmentations')]

	trainer_kwargs = train_workflow.get('params')
	trainer_kwargs.update(params)

	# Run
	if workflow.get('model'):
		f_trainer(
			model,
			X,
			y,
			f_preprocess,
			f_augmentations,
			weights_file.format(name),
			weights_best_file.format(name),
			val_loss_file.format(name),
			**trainer_kwargs)
	else:
		f_trainer(
			model_systole,
			model_diastole,
			X,
			y,
			f_preprocess,
			f_augmentations,
			weights_systole_file.format(name),
			weights_diastole_file.format(name),
			weights_systole_best_file.format(name),
			weights_diastole_best_file.format(name),
			val_loss_file.format(name),
			**trainer_kwargs)

def run_submit(name,workflow):
	submit_workflow = workflow['submit']

	_,_,X,ids = run_load(name,workflow)

	# Get functions from workflow
	if workflow.get('model'):
		model = getattr(models,workflow.get('model'))(workflow.get('model_inputs'))
		model.load_weights(weights_best_file.format(name))
	else:
		model_systole = getattr(models,workflow.get('models')[0])(workflow.get('model_inputs'))
		model_diastole = getattr(models,workflow.get('models')[1])(workflow.get('model_inputs'))
		model_systole.load_weights(weights_systole_best_file.format(name))
		model_diastole.load_weights(weights_diastole_best_file.format(name))
	f_preprocess = [
		preprocess_wrapper(getattr(preprocess,f_preprocess),**kwargs) 
		for f_preprocess,kwargs in workflow.get('preprocess',[])]
	f_submitter = getattr(submitters,submit_workflow.get('submitter'))

	submitter_kwargs = submit_workflow.get('params')

	# Run
	if workflow.get('model'):
		f_submitter(
			model,
			val_loss_file.format(name),
			X,
			ids,
			f_preprocess,
			LOCS.sample_submission_validate,
			submission_file.format(name),
			**submitter_kwargs
			)
	else:
		f_submitter(
			model_systole,
			model_diastole,
			val_loss_file.format(name),
			X,
			ids,
			f_preprocess,
			LOCS.sample_submission_validate,
			submission_file.format(name),
			**submitter_kwargs
			)