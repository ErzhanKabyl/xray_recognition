from src.data_preprocessing import get_data_generators
from src.model_builder import create_model
from src.training import train_model
from src.visualization import plot_training_history

if __name__ == "__main__":
    train_generator, test_generator = get_data_generators('dataset/train', 'dataset/test')
    model = create_model((224, 224, 3), num_classes=2)
    history = train_model(model, train_generator, test_generator, epochs=10)
    plot_training_history(history)
