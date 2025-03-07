from model_pipeline import train_model
import numpy as np

def test_model_training():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = train_model(X, y)

    # Vérifie que le modèle est bien un KNeighborsClassifier
    assert hasattr(model, "n_neighbors"), "Le modèle ne possède pas l'attribut 'n_neighbors'."
    
    # Vérifie que le nombre de voisins est bien celui par défaut (5)
    assert model.n_neighbors == 5, f"Le modèle devrait avoir n_neighbors=5, mais a {model.n_neighbors}."
