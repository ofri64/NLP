import os
import argparse
import numpy as np

from config import *

from DataProcessor import DataProcessor
from POSTaggers import SimpleTagger, MTLOneFeatureTagger, MTLAllFeaturesTagger
from KerasCallbacks import CloudCallback

LOG_FILE = 'experiments.log'


def datasets_paths(language):
    language_train_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', language, 'train.conllu')
    language_test_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', language, 'test.conllu')
    return os.path.abspath(language_train_path), os.path.abspath(language_test_path)


def pickle_path(name):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'pickles', name + '.pickle')
    return os.path.abspath(path)


def model_path(subpath):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'models', subpath)
    return os.path.abspath(path)


def log_experiment(experiment_name, acc, unseen_acc):
    with open(LOG_FILE, "a+") as logfile:
        logfile.write('-----------\n')
        logfile.write('Experiment: {0}\nAccuracy: {1}\nUnseen Accuracy: {2}\n'
                      .format(experiment_name, acc, unseen_acc))
        logfile.write('-----------\n\n')


def run_experiment(processor, tagger, train_path, test_path, load_processor_from=None, load_tagger_from=None,
                   features=None, name='experiment', remote=False, remote_stop=False, _all=False):
    cb = CloudCallback(remote=remote, slack_url=slack_url, stop_url=stop_url if remote_stop else '')

    if not _all:
        if type(features) is str:
            features = [features]
        elif not features:
            features = []

    try:
        # Initiate processor
        if load_processor_from:
            processor.load(load_processor_from)
        else:
            processor.process(train_path)
            if _all:
                features = processor.get_features()
            processor.save()

        word_dict = processor.get_word2idx_dict()
        tag_dict = processor.get_tag2idx_dict()

        feature_dicts = {k: v for k, v in processor.get_features2idx_dicts().items() if k in features}
        feature_dicts = feature_dicts.values()

        # Process training set
        x_train, y_train, y_train_features = processor.preprocess_sample(train_path)

        y_train_features = {k: v for k, v in y_train_features.items() if k in features}
        y_train_features = y_train_features.values()

        x_train = processor.transform_to_one_hot(x_train, len(word_dict))
        y_train = processor.transform_to_one_hot(y_train, len(tag_dict))
        y_train_features = [processor.transform_to_one_hot(y_train_feature, len(feature_dict)) for
                            y_train_feature, feature_dict in zip(y_train_features, feature_dicts)]

        # Process test set
        x_test, y_test, y_test_features = processor.preprocess_sample(test_path)

        y_test_features = {k: v for k, v in y_test_features.items() if k in features}
        y_test_features = y_test_features.values()

        x_test = processor.transform_to_one_hot(x_test, len(word_dict))
        y_test = processor.transform_to_one_hot(y_test, len(tag_dict))
        y_test_features = [processor.transform_to_one_hot(y_test_feature, len(feature_dict)) for
                           y_test_feature, feature_dict in zip(y_test_features, feature_dicts)]

        # Fit model
        if load_tagger_from:
            tagger.load_model_params(load_tagger_from)
        else:
            tagger.fit(x_train, [y_train] + y_train_features, callbacks=[cb])

        # Evaluate results
        cb.send_update('Evaluation has just started.')
        metrics_output = tagger.evaluate_sample(x_test, [y_test] + y_test_features)
        print_str = ""
        acc = None
        for metric, value in metrics_output:
            if "acc" in metric:
                if acc is None:  # only update the first acc
                    acc = value
                print_str += "{0}: {1} ".format(metric, value)

        cb.send_update('Results for: *{0}*'.format(name))
        cb.send_update('*Regular evaluation has ended!* ' + print_str)

        # Evaluate unseen results
        unseen_acc = tagger.evaluate_sample_conditioned(x_test, [y_test] + y_test_features, 'unseen')
        cb.send_update('*Unseen Evaluation has ended!* Accuracy: `{0}`'.format(unseen_acc))

        return acc, unseen_acc

    except Exception as e:
        cb.send_update(repr(e))

        import traceback
        traceback.print_exc()
    finally:
        cb.stop_instance()


def main(language, feature, n_epochs, times, remote, features, remote_stop, _all, hidden_size, load_processor_from, load_tagger_from):
    train_path, test_path = datasets_paths(language)

    if features:
        import pandas as pd
        dp = DataProcessor()
        dp.process(train_path)
        available_features = dp.get_features()
        print('\r\nAvailable features')
        print('------------------')
        print(pd.DataFrame(available_features))
        print('\r\n')
        return

    if _all:
        experiment_name = '{0}_all'.format(language)
        processor = DataProcessor(name=experiment_name)
        tagger = MTLAllFeaturesTagger(processor, n_epochs=n_epochs, hidden_size=hidden_size)
    elif feature:
        experiment_name = '{0}_{1}'.format(language, feature)
        processor = DataProcessor(name=experiment_name)
        tagger = MTLOneFeatureTagger(processor, n_epochs=n_epochs, feature=feature, hidden_size=hidden_size)
    else:
        experiment_name = language
        processor = DataProcessor(name=experiment_name)
        tagger = SimpleTagger(processor, n_epochs=n_epochs, hidden_size=hidden_size)

    if times == 1:
        if _all:
            acc, unseen_acc = run_experiment(processor, tagger, train_path, test_path,
                                             name=experiment_name, remote=remote, remote_stop=remote_stop, _all=True, load_processor_from=load_processor_from, load_tagger_from=load_tagger_from)
        elif feature:
            acc, unseen_acc = run_experiment(processor, tagger, train_path, test_path, features=[feature],
                                             name=experiment_name, remote=remote, remote_stop=remote_stop, load_processor_from=load_processor_from, load_tagger_from=load_tagger_from)
        else:
            acc, unseen_acc = run_experiment(processor, tagger, train_path, test_path, name=experiment_name,
                                             remote=remote, remote_stop=remote_stop, load_processor_from=load_processor_from, load_tagger_from=load_tagger_from)
        log_experiment(experiment_name, acc, unseen_acc)
    else:
        all_acc, all_unseen_acc = 0, 0
        for i in range(times):
            experiment_i_name = experiment_name + '_' + str(i + 1)
            if _all:
                acc, unseen_acc = run_experiment(processor, tagger, train_path, test_path,
                                                 name=experiment_i_name, remote=remote, remote_stop=remote_stop,
                                                 _all=True, load_processor_from=load_processor_from, load_tagger_from=load_tagger_from)
            elif feature:
                acc, unseen_acc = run_experiment(processor, tagger, train_path, test_path, features=[feature],
                                                 name=experiment_i_name, remote=remote, remote_stop=remote_stop, load_processor_from=load_processor_from, load_tagger_from=load_tagger_from)
            else:
                acc, unseen_acc = run_experiment(processor, tagger, train_path, test_path,
                                                 name=experiment_i_name, remote=remote, remote_stop=remote_stop, load_processor_from=load_processor_from, load_tagger_from=load_tagger_from)

            all_acc += acc
            all_unseen_acc += unseen_acc

        avg_acc = np.divide(all_acc, times)
        avg__unseen_acc = np.divide(all_unseen_acc, times)
        log_experiment(experiment_name, avg_acc, avg__unseen_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('language', help='language to experiment on')
    parser.add_argument('-f', '--feature', help='feature to experiment on')
    parser.add_argument('-e', '--n_epochs', help='number of epochs to run', type=int, default=10)
    parser.add_argument('-x', '--times', help='number of times to average the experiment', type=int, default=1)
    parser.add_argument('-r', '--remote', help='run remotely on the configured gcloud vm', action='store_true')
    parser.add_argument('-fs', '--features', help='get all features of the given language', action='store_true')
    parser.add_argument('-s', '--stop', help='stop the instance upon finish', action='store_true')
    parser.add_argument('-a', '--all', help='', action='store_true')
    parser.add_argument('-hs', '--hidden_size', help='number of epochs to run', type=int, default=100)
    parser.add_argument('-lp', '--load_processor_from', help="path to load trained processor", default=None)
    parser.add_argument('-lm', '--load_model_from', help="path to load a trained tagger model", default=None)

    args = parser.parse_args()

    main(args.language, args.feature, args.n_epochs, args.times, args.remote, args.features, args.stop, args.all,
         args.hidden_size,args.load_processor_from, args.load_model_from)
