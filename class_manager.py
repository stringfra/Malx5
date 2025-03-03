import os
from typing import Dict

def load_classes(model_path: str) -> Dict[int, str]:
    """Carica le classi associate al modello da varie possibili posizioni"""
    # Cerca il file delle classi in diverse posizioni
    possible_class_files = [
        os.path.splitext(model_path)[0] + ".txt",  # stesso nome del modello
        os.path.join(os.path.dirname(model_path), "classes.txt"),
        os.path.join(os.path.dirname(model_path), os.path.basename(model_path).replace(".onnx", ".txt"))
    ]
    
    print("Cercando file delle classi in:")
    for file_path in possible_class_files:
        print(f"- {file_path}")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    classes = {i: name.strip() for i, name in enumerate(f.readlines())}
                print(f"Caricate {len(classes)} classi da {file_path}")
                print(f"Prime 5 classi: {list(classes.items())[:5]}")
                return classes
            except Exception as e:
                print(f"Errore nel caricamento delle classi da {file_path}: {e}")
    
    # Se arriviamo qui, nessun file di classe è stato caricato
    print("ATTENZIONE: Nessun file delle classi trovato, uso classi COCO predefinite")
    return {
        0: 'person', 1: 'bicycle', 2: 'car'
        # aggiungi altre classi COCO se necessario
    }