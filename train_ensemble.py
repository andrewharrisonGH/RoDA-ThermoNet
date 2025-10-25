#!/usr/bin/env python3
# Author: Andrew Harrison | 1580584
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import argparse


def parse_cmd_args() -> argparse.Namespace:
    """
    Parses command line arguments required for training the model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--direct_features', dest='direct_features', type=str, required=True,
                        help='Features for the direct training set (N, D, H, W, C, float32).')
    parser.add_argument('-i', '--inverse_features', dest='inverse_features', type=str, required=True,
                        help='Features for the inverse training set (N, D, H, W, C, float32).')
    parser.add_argument('-y', '--direct_targets', dest='direct_targets', type=str, required=True,
                        help='Targets for the direct training set.')
    parser.add_argument('-t', '--inverse_targets', dest='inverse_targets', type=str, required=True,
                        help='Targets for the inverse training set.')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, required=True,
                        help='Number of training epochs.')
    parser.add_argument('--prefix', dest='prefix', type=str, required=True,
                        help='Prefix for the output file to store trained model.')
    parser.add_argument('--member', required=True, type=int,
                        help='Index (1-based) of ensemble member to train')
    parser.add_argument('-k', '--k', dest='k', type=int, default=10,
                        help='Number of ensemble members.')
    args = parser.parse_args()
    return args


def build_model(model_type: str = 'regression', conv_layer_sizes: tuple = (16, 16, 16), dense_layer_size: int = 16, dropout_rate: float = 0.5, input_shape: tuple = (16, 16, 16, 14)) -> tf.keras.Model:
    """
    Builds the 3D Convolutional Neural Network model for regression or classification.
    """
    # Check model type
    if model_type not in ['regression', 'classification']:
        print('Requested model type {0} is invalid'.format(model_type))
        sys.exit(1)
        
    # Instantiate 3D convnet
    model = models.Sequential()
    # First convolution layer
    model.add(layers.Conv3D(filters=conv_layer_sizes[0], kernel_size=(3, 3, 3), input_shape=input_shape))
    model.add(layers.Activation(activation='relu'))
    # Remaining convolution layers
    for c in conv_layer_sizes[1:]:
        model.add(layers.Conv3D(filters=c, kernel_size=(3, 3, 3)))
        model.add(layers.Activation(activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(units=dense_layer_size, activation='relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    
    # Last layer dependent on model type
    if model_type == 'regression':
        # Regression output
        model.add(layers.Dense(units=1))
    else:
        # Classification output
        model.add(layers.Dense(units=3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=0.0001),
                      metrics=['accuracy'])
    
    return model


def make_dataset_memmap(X_direct: np.ndarray, X_inverse: np.ndarray, y_direct: np.ndarray, y_inverse: np.ndarray,
                         direct_idx: np.ndarray, inverse_idx: np.ndarray, batch_size: int = 8, shuffle: bool = True) -> tf.data.Dataset:
    """
    Creates a vectorized tf.data input pipeline using memory-mapped NumPy arrays.
    Assumes features are pre-processed (N, D, H, W, C) and float32.
    """

    # Make paired indices
    pairs = np.stack([direct_idx, inverse_idx], axis=1)
    ds = tf.data.Dataset.from_tensor_slices(pairs)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), reshuffle_each_iteration=True)

    def _load_batch(batch_pairs):
        # Indices for direct and inverse samples
        d_idx = batch_pairs[:, 0]
        i_idx = batch_pairs[:, 1]

        # Vectorized load for direct: ONLY SLICING
        X_d = X_direct[d_idx]
        y_d = y_direct[d_idx].astype(np.float32)

        # Vectorized load for inverse: ONLY SLICING
        X_i = X_inverse[i_idx]
        y_i = y_inverse[i_idx].astype(np.float32)

        # Concatenate to form a full batch (direct + inverse)
        X = np.concatenate([X_d, X_i], axis=0)
        y = np.concatenate([y_d, y_i], axis=0)

        return X, y

    def _tf_load_batch(batch_pairs):
        # Load data using NumPy function for efficient I/O
        X, y = tf.numpy_function(_load_batch, [batch_pairs], [tf.float32, tf.float32])
        
        # Explicitly set the shapes for TensorFlow graph consistency
        X.set_shape([None, 16, 16, 16, 14])
        # Reshape y to (None, 1) for explicit regression output consistency 
        y = tf.expand_dims(y, axis=-1)
        y.set_shape([None, 1]) 
        
        return X, y

    # Batch indices first, then map to load data
    ds = ds.batch(batch_size // 2, drop_remainder=False)  # half batch of pairs -> full batch
    ds = ds.map(_tf_load_batch, num_parallel_calls=8)  # Parallelize data loading
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def train_member(member_idx: int, args: argparse.Namespace, X_direct: np.ndarray, X_inverse: np.ndarray, y_direct: np.ndarray, y_inverse: np.ndarray,
                  conv_layer_sizes: tuple = (16, 24, 32), dense_layer_size: int = 24) -> str:
    """
    Trains a single ensemble member on a k-fold split and returns the path to the best model.
    """
    
    batch_size = 8
    n_samples = len(y_direct)
    val_size = n_samples // args.k

    # Determine k-fold indices
    val_direct_idx = np.arange(member_idx * val_size, (member_idx + 1) * val_size)
    val_inverse_idx = val_direct_idx.copy()
    train_direct_idx = np.setdiff1d(np.arange(n_samples), val_direct_idx)
    train_inverse_idx = train_direct_idx.copy()

    # Log data sizes
    print(f"Member {member_idx + 1}: Training with {len(train_direct_idx) * 2} samples, Validating with {len(val_direct_idx) * 2} samples.")

    # Build datasets
    train_ds = make_dataset_memmap(X_direct, X_inverse, y_direct, y_inverse,
                                   train_direct_idx, train_inverse_idx,
                                   batch_size, shuffle=True)
    val_ds = make_dataset_memmap(X_direct, X_inverse, y_direct, y_inverse,
                                 val_direct_idx, val_inverse_idx,
                                 batch_size, shuffle=False)

    # Build and compile model
    model = build_model(model_type='regression', conv_layer_sizes=conv_layer_sizes, 
                        dense_layer_size=dense_layer_size, dropout_rate=0.5)
    model.compile(loss='mse', optimizer=optimizers.Adam(
        learning_rate=0.001, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=False
        ), 
        metrics=['mae']
    )

    model_path = f"{args.prefix}_member_{member_idx + 1}.h5"
    checkpoint = callbacks.ModelCheckpoint(model_path, monitor='val_loss',
                                           save_best_only=True, mode='min')
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Train model
    model.fit(train_ds, validation_data=val_ds,
              epochs=args.epochs,
              callbacks=[checkpoint, early_stopping],
              verbose=1)
    
    return model_path


def main() -> None:
    """
    Entry point for loading data, setting up GPU, and training the ensemble member.
    """
    args = parse_cmd_args()

    # Memory-mapped loading of pre-processed features and targets
    X_direct = np.load(args.direct_features, mmap_mode='r')
    X_inverse = np.load(args.inverse_features, mmap_mode='r')
    y_direct = np.loadtxt(args.direct_targets, dtype=np.float32)
    y_inverse = np.loadtxt(args.inverse_targets, dtype=np.float32)

    member_idx = args.member - 1
    print(f"Training ensemble member {args.member}/{args.k}")
    print(f"Features loaded successfully in (N, D, H, W, C) shape.")

    # Set up memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all visible GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled memory growth for GPUs.")
        except RuntimeError as e:
            print(e)
            
    train_member(member_idx, args, X_direct, X_inverse, y_direct, y_inverse)

if __name__ == '__main__':
    main()
