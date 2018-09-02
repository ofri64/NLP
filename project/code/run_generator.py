from GeneratorProcessor import GeneratorDataProcessor, dataset_path
from GeneratorPOSTagger import GeneratorPOSTagger
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Usage {0}: Please insert file name to process".format(sys.argv[0]))

    dataset_name = sys.argv[1]
    train_set = dataset_path(dataset_name, "train")
    generator_dp = GeneratorDataProcessor()
    # generator_dp.create_vocabs(train_set)
    # generator_dp.save()
    generator_dp.load()
    genModel = GeneratorPOSTagger(dataset_name, generator_dp, steps_per_epoch=220, n_epochs=10, multi_gpus=False)
    # genModel.fit(train_set)
    dev_set = dataset_path(dataset_name, "test")
    accuracy = genModel.predict(dev_set, num_batch_steps=21)
    print(accuracy)
