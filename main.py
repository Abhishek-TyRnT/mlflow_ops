import tensorflow as tf
import argparse
from src.data.data import get_train, get_test, get_val, get_classes, search_files
from src.model.inceptionet import InceptionNet
from src.train import Train

def get_args():
    parser = argparse.ArgumentParser(description = "args for training the model")
    parser.add_argument('-dd','--data-dir', type = str, required= True, help = "path to the data dir")
    parser.add_argument('-m','--model-name', type = str, default = 'inceptionet', help = "name of the model")
    parser.add_argument("-s",'--seed', type = int, default = 400, help = "seed value for randomisation")
    parser.add_argument('-bs', '--batch-size', type = int, default = 128, help = 'batch size for the training data')
    parser.add_argument('-e', '--epochs', type = int, default = 100, help = 'number of epochs for training')
    parser.add_argument('-lr', '--learning-rate', type = float, default = 1e-3, help = 'learning rate for training')
    parser.add_argument('-opt','--optimizer', type = str, default = "sgd", help = "optimizer for training")

    return parser.parse_args()

def main(args):
    data_dir = args.data_dir
    batch_size = args.batch_size
    seed_value = args.seed
    input_shape = (224, 224, 3)
    optimizer_name = args.optimizer
    epochs = args.epochs

    classes = get_classes(data_dir)
    train_dataset = get_train(args.data_dir,
                             classes, 
                             input_shape, 
                             True, 
                             batch_size, 
                             seed_value
                            )

    val_dataset = get_val(
                        args.data_dir,
                        classes,
                        input_shape,
                        True,
                        batch_size,
                        seed_value
    )

    no_steps_per_epoch = len(search_files(f"{data_dir}/seg_train", ".jpg"))//batch_size
    val_steps_per_epoch    = len(search_files(f"{data_dir}/seg_val", ".jpg"))//batch_size

    
    if optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate = args.learning_rate)
    
    elif optimizer_name == "rms_prop":
        optimizer = tf.keras.optimizers.RMSProp(learning_rate = args.learning_rate)
    
    else:
        raise ValueError(f"optimizer is {optimizer_name} is unknown")
    

    loss_func = tf.keras.losses.CategoricalCrossentropy()


    inp = tf.keras.layers.Input(shape = input_shape)

    if args.model_name == "inceptionet":
        model = InceptionNet(len(classes))
    
    else:
        raise ValueError(f"model name {args.model_name} is unknown")
    

    _ = model(inp)

    train = Train(
        model, 
        loss_func, 
        optimizer
    )

    model = train.fit(train_dataset, val_dataset, epochs, no_steps_per_epoch, val_steps_per_epoch)
    # for image, _cls in train_dataset.take(1):
    #     y = model(image)
    #     print(y)

if __name__ == "__main__":
    args = get_args()
    main(args)


    




    
    
