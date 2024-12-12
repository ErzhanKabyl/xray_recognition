from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir, test_dir, target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,  # Увеличиваем вращение
        width_shift_range=0.2,  # Увеличиваем сдвиг по ширине
        height_shift_range=0.2,  # Увеличиваем сдвиг по высоте
        zoom_range=0.2,  # Увеличиваем зум
        horizontal_flip=True,
        shear_range=0.2,  # Добавляем эффект сдвига
        brightness_range=[0.8, 1.2]  # Регулируем яркость
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse'
    )

    return train_generator, test_generator
