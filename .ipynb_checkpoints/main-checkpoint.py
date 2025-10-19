import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
   
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, y_test = load_data()
    means = xtrain.mean(axis=(0, 1, 2))  # shape (3,)
    stds = xtrain.std(axis=(0, 1, 2))    # shape (3,)
    
    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test:
        from sklearn.model_selection import train_test_split
        xtrain, x_val, ytrain, y_val = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42, stratify=ytrain)
    else:
        x_val, y_val = xtest, y_test    
    ### WRITE YOUR CODE HERE


    ### WRITE YOUR CODE HERE to do any other data processing
        
    xtrain = normalize_fn(xtrain, means, stds)
    x_val = normalize_fn(x_val, means, stds)
    xtest = normalize_fn(xtest, means, stds)

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        xtrain_flat = xtrain.reshape(xtrain.shape[0], -1)
        x_val_flat = x_val.reshape(x_val.shape[0], -1)
        xtest_flat = xtest.reshape(xtest.shape[0], -1)
    
        model = MLP(input_size=xtrain_flat.shape[1], n_classes=n_classes)
        xtrain_input, xval_input, xtest_input = xtrain_flat, x_val_flat, xtest_flat
            
    elif args.nn_type == "cnn":
        xtrain_input = np.transpose(xtrain, (0, 3, 1, 2))  # (N, H, W, C) → (N, C, H, W)
        xval_input = np.transpose(x_val, (0, 3, 1, 2))
        xtest_input = np.transpose(xtest, (0, 3, 1, 2))
        model = CNN(input_channels=3, n_classes=n_classes)
    
    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain_input, ytrain)

    # Predict on unseen data
    
    if args.test:
        preds = method_obj.predict(xtest_input)
        true_labels = y_test
    else:
        preds = method_obj.predict(xval_input)
        true_labels = y_val

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, true_labels)
    macrof1 = macrof1_fn(preds, true_labels)
    if args.test:
        
        print(f"Test set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    else:
        
        print(f"Validation set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
