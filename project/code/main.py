import os
import argparse
import numpy as np

from config import *

from DataProcessor import DataProcessor
from POSTaggers import SimpleTagger, MTLOneFeatureTagger, MTLAllFeaturesTagger
from KerasCallbacks import CloudCallback

LOG_FILE = '../logs/experiments.log'

cb = None


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


def log_experiment(experiment_name, acc, unseen_acc, ambig_acc, hidden_size, embedding_size):
    print('\nacc:\t\t\t{0}\nunseen acc:\t\t{1}\nambiguous acc:\t{2}'.format(acc, unseen_acc, ambig_acc))
    with open(LOG_FILE, "a+") as logfile:
        logfile.write('-----------\n')
        logfile.write('Experiment:\t\t\t{0}\n'
                      'Accuracy:\t\t\t{1}\n'
                      'Unseen Accuracy:\t{2}\n'
                      'Ambiguous Accuracy:\t{3}\n'
                      'Hidden Size:\t\t{4}\n'
                      'Embedding Size:\t\t{5}\n'
                      .format(experiment_name, acc, unseen_acc, ambig_acc, hidden_size, embedding_size))
        logfile.write('-----------\n\n')


def run_experiment(processor, tagger, train_path, test_path, load_processor=None, load_tagger=None,
                   features=None, name=None, remote=False, remote_stop=False, _all=False):
    cb = CloudCallback(remote=remote, slack_url=slack_url, stop_url=stop_url if remote_stop else '')

    if not _all:
        if type(features) is str:
            features = [features]
        elif not features:
            features = []
    try:
        # Initiate processor
        load_processor_path = pickle_path(name)

        if load_processor:  # load a trained processor from file
            processor.load(load_processor_path)

        else:  # train a processor using the training data
            processor.process(train_path)
            processor.save(file_path=load_processor_path)

        if _all:
            features = processor.get_features()

        word_dict = processor.get_word2idx_dict()
        tag_dict = processor.get_tag2idx_dict()

        feature_dicts = {k: v for k, v in processor.get_features2idx_dicts().items() if k in features}
        feature_dicts = feature_dicts.values()

        if load_tagger:
            load_tagger_path = model_path(name + ".h5")
            tagger.load_model_params(load_tagger_path)

        else:  # need to preprocess training data and train model

            # Process training set
            x_train, y_train, y_train_features = processor.preprocess_sample(train_path)

            y_train_features = {k: v for k, v in y_train_features.items() if k in features}
            y_train_features = y_train_features.values()

            x_train = processor.transform_to_one_hot(x_train, len(word_dict))
            y_train = processor.transform_to_one_hot(y_train, len(tag_dict))
            y_train_features = [processor.transform_to_one_hot(y_train_feature, len(feature_dict)) for
                                y_train_feature, feature_dict in zip(y_train_features, feature_dicts)]

            tagger.fit(x_train, [y_train] + y_train_features, callbacks=[cb], name=name)

        # Process test set
        x_test, y_test, y_test_features = processor.preprocess_sample(test_path)

        y_test_features = {k: v for k, v in y_test_features.items() if k in features}
        y_test_features = y_test_features.values()

        x_test = processor.transform_to_one_hot(x_test, len(word_dict))
        y_test = processor.transform_to_one_hot(y_test, len(tag_dict))
        y_test_features = [processor.transform_to_one_hot(y_test_feature, len(feature_dict)) for
                           y_test_feature, feature_dict in zip(y_test_features, feature_dicts)]

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

        # Evaluate ambiguous results
        ambig_acc = tagger.evaluate_sample_conditioned(x_test, [y_test] + y_test_features, 'ambiguous')
        cb.send_update('*Ambiguous Evaluation has ended!* Accuracy: `{0}`'.format(ambig_acc))

        return acc, unseen_acc, ambig_acc

    except Exception as e:
        cb.send_update(repr(e))

        import traceback
        traceback.print_exc()
    finally:
        cb.stop_instance()


def evaluate(processor, tagger, test_path, feature=None, _all=False):
    word_dict = processor.get_word2idx_dict()
    tag_dict = processor.get_tag2idx_dict()

    features = []
    if _all:
        features = processor.get_features()
    elif feature:
        features = [feature]

    # Process test set
    x_test, y_test, y_test_features = processor.preprocess_sample(test_path)

    x_test = processor.transform_to_one_hot(x_test, len(word_dict))
    y_test = processor.transform_to_one_hot(y_test, len(tag_dict))

    if features:
        feature_dicts = {k: v for k, v in processor.get_features2idx_dicts().items() if k in features}
        feature_dicts = feature_dicts.values()

        y_test_features = {k: v for k, v in y_test_features.items() if k in features}
        y_test_features = y_test_features.values()

        y_test_features = [processor.transform_to_one_hot(y_test_feature, len(feature_dict)) for
                           y_test_feature, feature_dict in zip(y_test_features, feature_dicts)]

    metrics_output = tagger.evaluate_sample(x_test, [y_test] + y_test_features if features else [y_test])
    acc = list(metrics_output)[1][1]
    unseen_acc = tagger.evaluate_sample_conditioned(x_test, [y_test] + y_test_features if features else [y_test], 'unseen')
    return acc, unseen_acc


def main(language, feature, n_epochs, times, remote, features, remote_stop, _all, embedding_size, hidden_size,
         load_processor, load_tagger, to_evaluate):
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
        tagger = MTLAllFeaturesTagger(processor, n_epochs=n_epochs, hidden_size=hidden_size, embed_size=embedding_size)
    elif feature:
        experiment_name = '{0}_{1}'.format(language, feature)
        processor = DataProcessor(name=experiment_name)
        tagger = MTLOneFeatureTagger(processor, n_epochs=n_epochs, feature=feature, hidden_size=hidden_size,
                                     embed_size=embedding_size)
    else:
        experiment_name = language
        processor = DataProcessor(name=experiment_name)
        tagger = SimpleTagger(processor, n_epochs=n_epochs, hidden_size=hidden_size, embed_size=embedding_size)

    all_acc, all_unseen_acc, all_ambig_acc = 0, 0, 0

    if to_evaluate:
        cb = CloudCallback(remote=remote, slack_url=slack_url)

        load_processor_path = pickle_path(experiment_name)
        processor.load(load_processor_path)

        load_tagger_path = model_path(experiment_name + ".h5")
        tagger.load_model_params(load_tagger_path)

        train_path, test_path = datasets_paths(to_evaluate)
        test_data_name = '/'.join(test_path.split('/')[-2:])

        eval_name = 'Evaluating {0} on {1}'.format(experiment_name, test_data_name)
        print(eval_name)
        acc, unseen_acc = evaluate(processor, tagger, test_path, feature, _all)

        log_experiment(eval_name, acc, unseen_acc, 0, 0, 0)
        return

    for i in range(times):
        # experiment_i_name = experiment_name + '_' + str(i + 1)

        features = [feature] if feature else None
        acc, unseen_acc, ambig_acc = run_experiment(processor, tagger, train_path, test_path, features=features,
                                                    name=experiment_name, remote=remote, remote_stop=remote_stop,
                                                    _all=_all, load_processor=load_processor, load_tagger=load_tagger)

        all_acc += acc
        all_unseen_acc += unseen_acc
        all_ambig_acc += ambig_acc

    # Average over all accuracies and log
    avg_acc = np.divide(all_acc, times)
    avg_unseen_acc = np.divide(all_unseen_acc, times)
    avg_ambig_acc = np.divide(all_ambig_acc, times)
    log_experiment(experiment_name, avg_acc, avg_unseen_acc, avg_ambig_acc, hidden_size, embedding_size)


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
    parser.add_argument('-em', '--embedding_size', help='size of embedding layer', type=int, default=50)
    parser.add_argument('-hs', '--hidden_size', help='number of epochs to run', type=int, default=100)
    parser.add_argument('-lp', '--load_processor', help="load train processor flag", action='store_true')
    parser.add_argument('-lm', '--load_model', help="load a trained tagger model flag", action='store_true')
    parser.add_argument('-ev', '--evaluate', help="evaluate a pretrained model")

    args = parser.parse_args()

    main(args.language, args.feature, args.n_epochs, args.times, args.remote, args.features, args.stop, args.all,
         args.embedding_size, args.hidden_size, args.load_processor, args.load_model, args.evaluate)
