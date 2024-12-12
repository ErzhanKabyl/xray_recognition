from tensorflow import keras

def train_model(model, train_generator, test_generator, epochs=5):
    checkpoint = keras.callbacks.ModelCheckpoint(
        'model/xray_classification_{epoch:02d}_{val_accuracy:.3f}.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[checkpoint, reduce_lr]
    )
    return history


