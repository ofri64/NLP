import argparse
from subprocess import call

from config import *


def main(language, feature, n_epochs, times, remote, features, remote_update, remote_stop,
         _all, embedding_size, hidden_size, load_processor_from, load_tagger_from, evaluate):
    if not remote:
        import main as _main_
        return _main_.main(language, feature, n_epochs, times, remote, features, False, _all, embedding_size, hidden_size, load_processor_from, load_tagger_from, evaluate)

    call('gcloud compute instances start --zone={zone} {vm}'.format(zone=gcloud_zone, vm=gcloud_vm), shell=True)

    if remote_update:
        print('Updating code.')
        call('gcloud compute ssh --zone={zone} {vm} --command "rm -rfv {remote_dir}/*"'.format(
            zone=gcloud_zone,
            vm=gcloud_vm,
            remote_dir=gcloud_code_dir
        ), shell=True)
        call('gcloud compute scp --recurse --compress --zone={zone} {local_dir} {username}@{vm}:~/nlprun/'.format(
            zone=gcloud_zone,
            local_dir=local_code_dir,
            username=gcloud_username,
            vm=gcloud_vm
        ), shell=True)
        print('done.')

    print('Starting experiment remotely.')
    call('gcloud compute ssh --zone={zone} {vm} --command "nohup /home/shared/anaconda3/bin/python {remote_dir}/main.py {language} -r -e {epochs}{feature}{features}{stop} > run.out 2> run.err < /dev/null &" &'.format(
        zone=gcloud_zone,
        vm=gcloud_vm,
        remote_dir=gcloud_code_dir,
        language=language,
        epochs=n_epochs,
        feature=' -f %s' % feature if feature else '',
        features=' -a' if features else '',
        stop=' -s' if remote_stop else ''
    ), shell=True)
    print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('language', help='language to experiment on')
    parser.add_argument('-f', '--feature', help='feature to experiment on')
    parser.add_argument('-e', '--n_epochs', help='number of epochs to run', type=int, default=10)
    parser.add_argument('-r', '--remote', help='run remotely on the configured gcloud vm', action='store_true')
    parser.add_argument('-fs', '--features', help='get all features of the given language', action='store_true')
    parser.add_argument('-u', '--update', help='update the remote code', action='store_true')
    parser.add_argument('-s', '--stop', help='stop the instance upon finish', action='store_true')
    parser.add_argument('-a', '--all', help='to run with all available features', action='store_true')
    parser.add_argument('-em', '--embedding_size', help='size of embedding layer', type=int, default=50)
    parser.add_argument('-hs', '--hidden_size', help='size of hidden layer in LSTM layers', type=int, default=100)
    parser.add_argument('-x', '--times', help='number of experiments to run', type=int, default=1)
    parser.add_argument('-lp', '--load_processor', help="load train processor flag", action='store_true')
    parser.add_argument('-lm', '--load_model', help="load a trained tagger model flag", action='store_true')
    parser.add_argument('-ev', '--evaluate', help="evaluate a pretrained model")

    args = parser.parse_args()

    main(args.language, args.feature, args.n_epochs, args.times, args.remote, args.features, args.update, args.stop,
         args.all, args.embedding_size, args.hidden_size, args.load_processor, args.load_model, args.evaluate)
