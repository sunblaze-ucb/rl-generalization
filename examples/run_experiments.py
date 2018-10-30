import argparse
import collections
import itertools
import glob
import json
import logging
import os
import shlex
import subprocess

import natsort
import numpy as np
import yaml

logger = logging.getLogger(__name__)


TrainedModel = collections.namedtuple('TrainedModel', [
    'base_directory', 'model', 'environment', 'model_info', 'parameters',
])


def model_train(args, model, environment, dry_run, results_dir_base, skip_existing_train=False):
    """Run model training script."""
    cmd_str = ""
    needs_train = True
    output_directory = os.path.join(results_dir_base, model['name'].replace(' ', '-'), environment)
    try:
        os.makedirs(output_directory)
    except OSError:
        logger.warning("Experiment directory '{}' already exists. Assuming trained model.".format(output_directory))
        if skip_existing_train:
            logger.warning("Skip existing train flag enabled, returning.")
            return
        needs_train = False

    if needs_train:
        process = model['train']['command'].strip().format(
            environment=environment,
            output=output_directory,
            **model['hyperparameters']
        )
        process = shlex.split(process)
        cmd_str = " ".join(process)
        if dry_run:
            logger.debug("[DRY RUN] Running: '{}'".format(cmd_str))
        else:
            logger.debug("Running: '{}'".format(cmd_str))
            try:
                subprocess.run(process, check=True)
            except subprocess.CalledProcessError:
                logger.error("Failed to train model '{}' on environment '{}'.".format(model['name'], environment))
                return

    # Check if output directory exists and return it.
    if dry_run or model['train'].get('output_no_check', False):
        output_model = os.path.join(output_directory, model['train']['output'])
    else:
        output_model = glob.glob(os.path.join(output_directory, model['train']['output']))
        if not output_model:
            logger.error("Unable to find trained model output file.")
            return

        # Fixes issue of 'normalize' file inside checkpoint folder
        output_model = [f for f in output_model if 'normalize' not in f]
        if not output_model:
            logger.error("Unable to find trained model output file.")
            return
        output_model = natsort.natsorted(output_model, reverse=True)[0]

    # Get all parameters used during training.
    parameters = []
    if 'parameters' in model['train']:
        for parameters_filename in glob.glob(os.path.join(output_directory, model['train']['parameters'])):
            with open(parameters_filename) as parameters_file:
                for line in parameters_file:
                    try:
                        parameters.append(json.loads(line))
                    except ValueError:
                        continue

    return TrainedModel(
        base_directory=output_directory,
        model=output_model,
        model_info=model,
        environment=environment,
        parameters=parameters,
    ), cmd_str


def model_evaluate(args, model, environment, trained_model, dry_run):
    """Run model evaluation script."""
    cmd_str = ""
    needs_evaluate = True
    output_directory = os.path.join(trained_model.base_directory, 'evaluations', environment)

    # make sure file exists before running the rest
    if not os.path.isfile(trained_model.model):
        logger.error("Checkpoint does not exist: '{}', stopping eval.".format(
            trained_model.model
        ))
        return

    try:
        os.makedirs(output_directory)
    except OSError:
        if not args.force_evaluate:
            logger.warning("Experiment directory '{}' already exists. Assuming evaluation done.".format(
                output_directory
            ))
            needs_evaluate = False

    if needs_evaluate:
        process = model['evaluate']['command'].strip().format(
            environment=environment,
            output=output_directory,
            model=trained_model.model,
            #**model['hyperparameters']
        )
        process = shlex.split(process)
        cmd_str = " ".join(process)
        if dry_run:
            logger.debug("[DRY RUN] Running: '{}'".format(cmd_str))
        else:
            logger.debug("Running: '{}'".format(cmd_str))

            try:
                subprocess.run(process, check=True)
            except subprocess.CalledProcessError:
                logger.error("Failed to evaluate model '{}' on environment '{}'.".format(model['name'], environment))
                return

    # We can't evaluate the reward statistics on a dry run, so return a dummy dict
    if dry_run:
        return {}, cmd_str

    else:
        # Get evaluation results.
        evaluation = glob.glob(os.path.join(output_directory, model['evaluate']['output']))
        if not evaluation:
            logger.error("Unable to find evaluation output file.")
            return
    
        evaluation = natsort.natsorted(evaluation, reverse=True)[0]
        with open(evaluation) as evaluation_file:
            episodes = []
            for line in evaluation_file:
                try:
                    episodes.append(json.loads(line))
                except ValueError:
                    pass
    
            rewards = np.asarray([data['reward'] for data in episodes])
    
        return {
            'model': trained_model.model_info,
            'trained_on': trained_model.environment,
            'trained_parameters': trained_model.parameters,
            'evaluated_on': environment,
            'episodes': episodes,
            'rewards': {
                'count': len(rewards),
                'mean': float(np.mean(rewards)),
                'median': float(np.median(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
            }
        }, cmd_str


def random_evaluate(args, model, environment, results_dir, dry_run):
    """Similar to model_evaluate(), but doesn't take trained model as input.

    Whereas the log dir format for regular (non-random) models is:
        results_dir/model-name/training-env-name/progress.csv (logs, etc)
        results_dir/model-name/training-env-name/checkpoints/* (saved model files)
        results_dir/model-name/training-env-name/evaluations/testing-env-name/results.json (testing results)

    The log dir format for random models is simply:
        results_dir/Random/evaluations/testing-env-name/log.txt (testing log, etc.)
        results_dir/Random/evaluations/testing-env-name/results.json (testing results)
    """
    cmd_str = ""
    needs_evaluate = True
    output_directory = os.path.join(results_dir, model['name'], 'evaluations', environment)
    try:
        os.makedirs(output_directory)
    except OSError:
        if not args.force_evaluate:
            logger.warning("Experiment directory '{}' already exists. Assuming evaluation done.".format(
                output_directory
            ))
            needs_evaluate = False

    if needs_evaluate:
        process = model['evaluate']['command'].strip().format(
            environment=environment,
            output=output_directory,
        )
        process = shlex.split(process)
        cmd_str = " ".join(process)
        if dry_run:
            logger.debug("[DRY RUN] Running: '{}'".format(cmd_str))
        else:
            logger.debug("Running: '{}'".format(cmd_str))

            try:
                subprocess.run(process, check=True)
            except subprocess.CalledProcessError:
                logger.error("Failed to evaluate model '{}' on environment '{}'.".format(model['name'], environment))
                return

    # We can't evaluate the reward statistics on a dry run, so return a dummy dict
    if dry_run:
        return {}, cmd_str

    else:
        # Get evaluation results.
        evaluation = glob.glob(os.path.join(output_directory, model['evaluate']['output']))
        if not evaluation:
            logger.error("Unable to find evaluation output file.")
            return
    
        evaluation = natsort.natsorted(evaluation, reverse=True)[0]
        with open(evaluation) as evaluation_file:
            episodes = []
            for line in evaluation_file:
                try:
                    episodes.append(json.loads(line))
                except ValueError:
                    pass
    
            rewards = np.asarray([data['reward'] for data in episodes])
    
        return {
            'model': model['name'],
            'trained_on': 'N/A',
            'trained_parameters': 'N/A',
            'evaluated_on': environment,
            'episodes': episodes,
            'rewards': {
                'count': len(rewards),
                'mean': float(np.mean(rewards)),
                'median': float(np.median(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
            }
        }, cmd_str


def record_result(results_dir, result):
    """Record evaluation result."""
    with open(os.path.join(results_dir, 'results.json'), 'a') as results_file:
        results_file.write(json.dumps(result))
        results_file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('experiments', type=str,
                        help="Experiments definition file (.yml)")
    parser.add_argument('output', type=str,
                        help="Output directory for all experiments")
    parser.add_argument('--filter-models', type=str, nargs='+',
                        help="Only run experiments on specified models")
    parser.add_argument('--filter-envs', type=str, nargs='+',
                        help="Only run experiments on specified environments")
    parser.add_argument('--force-evaluate', action='store_true',
                        help="Run experiment even if directory already exists")
    # NOTE: This creates dummy files. Make sure to run on empty output
    # directory and delete afterwards
    parser.add_argument('--dry-run', action='store_true',
                        help="Print the expr. commands but don't run them")
    parser.add_argument('--dry-run-file', default='run_experiments_cmds',
                        help="Where to store the commands (saves both .txt and .json)")
    # NOTE: If used with --dry-run on an experiments.yml with explicit
    # testing on train envs, script will fail
    parser.add_argument('--eval-train', action='store_true',
                        help="Implicitly evaluate on the training environment")
    parser.add_argument('--skip-existing-train', action='store_true',
                        help="Skip models dirs with existing checkpoint files but no evaluate dir")
    parser.add_argument('--skip-existing-eval', action='store_true',
                        help="Skip models dirs with existing evaluate dir")
    args = parser.parse_args()

    # Configure logger.
    logging.basicConfig(
        format='%(asctime)-15s %(levelname)s: %(message)s',
        level=logging.DEBUG,
    )

    # Store then write out the commands to be run (dry run only)
    # Dict should look like:
    # commands = {
    #   'PPO': [{
    #     'train': 'run xyz_command',
    #     'test':  ['run a_command', 'run b_command', ...]
    #   }],
    #   'TRPO': {
    #   ...
    if args.dry_run:
        commands = {}

    # Load experiments definitions.
    logger.info("Loading experiment plan '{}'.".format(args.experiments))
    with open(args.experiments) as experiments_file:
        config = yaml.load(experiments_file)

    logger.info("Models:")
    for model in config['models']:
        logger.info('  * {}'.format(model['name']))

    logger.info("Loaded {} environment sets.".format(len(config['environments'])))


    for model in config['models']:
        if args.filter_models and model['name'] not in args.filter_models:
            continue
        random_model = model['name'] == 'Random'

        if random_model:
            # Hyperparameters don't make sense for Random
            assert 'hyperparameters' not in model
            hp_dicts = [{}]
        else:
            # Check in advance to see if any model hyperparameters are specified as lists
            # If so, then have an outer-loop that does a full sweep based on the list(s)
            hp_original_copy = model['hyperparameters'].copy()
            hps_with_list = list(filter(lambda kv: isinstance(kv[1], list), model['hyperparameters'].items()))
            if len(hps_with_list) > 0:
                logger.info("Found hyperparameter(s) specified as list, performing hyperparameter sweep.")
                # Generate Cartesian product of all hyperparameters specified as lists
                hp_names, hp_vals = zip(*hps_with_list)
                hp_vals_combos = list(itertools.product(*hp_vals))
                logger.info("Testing following combinations for {}: {}.".format(hp_names, hp_vals_combos))
                # Create a different hyperparam. dict. for each combination
                hp_dicts = []
                for combo in hp_vals_combos:
                    base_dict = model['hyperparameters'].copy()
                    # Overwrite (list) values with specific combination
                    base_dict.update(dict(zip(hp_names, combo)))
                    hp_dicts.append(base_dict)
            else:
                hp_dicts = [model['hyperparameters']]
            # Dump all the dicts for the log
            logger.info("Full hyperparameter set(s) [total %d]:" % len(hp_dicts))
            for hpd in hp_dicts:
                logger.info(hpd)

        # Loop over each hyperparameter configuration,
        # and write-out to a different results DIR for each
        output_dir_base = str(args.output)
        for hp_dict in hp_dicts:

            output_dir = output_dir_base
            if not random_model:
                # Code to modify the provided log dir (args.output) with hyperparameter suffix
                model['hyperparameters'] = hp_dict
                # Only append to output DIR if we're doing a hyperparameter sweep
                if len(hp_dicts) > 1:
                    # Sort by key so that we get consistent log dirs
                    hp_dict_sorted = [(k, hp_dict[k]) for k in sorted(hp_dict)]
                    suffix = '_'.join(['{0}-{1}'.format(k, v) for k, v in hp_dict_sorted])
                    output_dir = output_dir_base + '___' + suffix
                    logger.info("Hyperparameter sweep dict: {}".format(hp_dict))
                    logger.info("Logging to: {}".format(output_dir))

            logger.info("Evaluating model '{}'.".format(model['name']))
            if args.dry_run:
                # Algorithm -> list of train/test combos to try
                if model['name'] not in commands:
                    commands[model['name']] = []

            for environment in config['environments']:
                if args.filter_envs and environment['train'] not in args.filter_envs:
                    continue

                if args.skip_existing_train:
                    output_directory = os.path.join(output_dir_base, model['name'].replace(' ', '-'), environment['train'])
                    if os.path.isdir(output_directory):
                        logger.warning("Experiment directory '{}' already exists, and --skip-existing-train flag enabled. Skipping environment.".format(output_directory))
                        continue

                if random_model:
                    # 'Random' is a special case where we don't actually train anything
                    logger.info("'Random' model specified, skipping training.")
                else:
                    logger.info("Training on '{}'.".format(environment['train']))
                    ret = model_train(args, model, environment['train'], args.dry_run, output_dir, args.skip_existing_train)
                    if not ret:
                        continue
                    else:
                        trained_model, cmd = ret

                if args.dry_run:
                   # Store the command for training on the train env
                   env_exprs = {}
                   if not random_model:
                       env_exprs['train'] = cmd
                   env_exprs['test'] = []   

                # Evaluate on the train environment
                # In the current experiments.yml, this is done explicitly instead
                if args.eval_train:
                    logger.info("Evaluating on '{}'.".format(environment['train']))
                    ret = model_evaluate(args, model, environment['train'], trained_model, args.dry_run)
                    if not ret:
                        continue
                    else:
                        results_train, cmd = ret
    
                    if args.dry_run:
                        # Store the command for evaluating on the train env
                        env_exprs['test'].append(cmd)
                    else:
                        # Skip if we just want to see the commands
                        record_result(output_dir, results_train)

                test_environments = environment['test']
                if not isinstance(test_environments, list):
                    test_environments = [test_environments]
    
                for test_environment in test_environments:

                    # If a directory for this test scenario exists, skip it
                    if args.skip_existing_eval:
                        eval_directory = os.path.join(output_dir_base,
                            model['name'].replace(' ', '-'),
                            environment['train'], 'evaluations',
                            test_environment)
                        if os.path.isdir(eval_directory):
                            logger.warning("Evaluation directory '{}' already exists, and --skip-existing-eval flag enabled. Skipping environment.".format(eval_directory))
                            continue
 
                    logger.info("Evaluating on '{}'.".format(test_environment))
                    if random_model:
                        results_test, cmd = random_evaluate(args, model, test_environment, output_dir_base, args.dry_run)
                    else:
                        ret = model_evaluate(args, model, test_environment, trained_model, args.dry_run)
                        if not ret:
                            continue
                        else:
                            results_test, cmd = ret
                    if args.dry_run:
                        # Store the command for evaluating on this test env
                        env_exprs['test'].append(cmd)
                    else:
                        # Skip if we just want to see the commands
                        record_result(output_dir, results_test)
    
                if args.dry_run:
                    commands[model['name']].append(env_exprs)

    # Dump the commands
    if args.dry_run:
        logger.info("Writing commands to {}.txt/json".format(args.dry_run_file))
        with open(args.dry_run_file + '.json', 'w') as fout:
            json.dump(commands, fout)
        with open(args.dry_run_file + '.txt', 'w') as fout:
            for model_name, env_list in commands.items():
                # fout.write('# {}\n'.format(model_name))
                for env_dict in env_list:
                    # Random models won't have train commands specified
                    if 'train' in env_dict:
                        train_cmd = env_dict['train']
                        fout.write('{}\n'.format(train_cmd))
                    test_cmds = env_dict['test']
                    for test_cmd in test_cmds:
                        fout.write('{}\n'.format(test_cmd))

