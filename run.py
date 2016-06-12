import argparse
from framework.workflows import workflows
from framework import controler

def run(args):

	if args.command == 'preselect':
		controler.run_preselect(args.workflow,workflows[args.workflow])

	if args.command == 'load':
		controler.run_write(args.workflow,workflows[args.workflow])

	elif args.command == 'train':
		params = {}
		if args.nb_iter: params['nb_iter']= args.nb_iter
		controler.run_train(args.workflow,workflows[args.workflow],**params)

	elif args.command == 'submit':
		controler.run_submit(args.workflow,workflows[args.workflow])

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run and train models')

	parser.add_argument('command',choices=['preselect','load','train','submit'])
	parser.add_argument('workflow',choices=workflows.keys())
	parser.add_argument('--nb-iter', type=int)

	args = parser.parse_args()
	run(args)