#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Screen Object Detector con supporto ONNX
------------------------------------------------

Un'applicazione avanzata per il rilevamento di oggetti sullo schermo
che utilizza modelli ONNX e supporta sia GPU NVIDIA che AMD.

Caratteristiche:
- Rilevamento oggetti in tempo reale
- Supporto modelli ONNX
- Compatibilità GPU NVIDIA/AMD/CPU
- Assistenza al puntamento configurabile
"""

import os
import sys
import tkinter as tk
import argparse
import pyautogui
import warnings
import traceback
import logging
from pathlib import Path

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("detector.log", mode='w')
    ]
)
logger = logging.getLogger("detector")

# Ignora avvisi specifici
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
warnings.filterwarnings("ignore", category=FutureWarning)

def check_requirements():
    """Verifica i requisiti dell'applicazione"""
    try:
        import numpy
        import cv2
        import torch
        import pandas
        import onnxruntime
        import mss
        import keyboard
        
        logger.info("Tutti i requisiti soddisfatti")
        return True
    except ImportError as e:
        logger.error(f"Requisito mancante: {e}")
        return False

def find_default_model():
    """Cerca un modello ONNX predefinito"""
    search_paths = [
        ".",
        "models",
        "data",
        "assets",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    ]
    
    for path in search_paths:
        if not os.path.exists(path):
            continue
        
        for file in os.listdir(path):
            if file.endswith(".onnx"):
                full_path = os.path.join(path, file)
                logger.info(f"Trovato modello predefinito: {full_path}")
                return full_path
    
    return None

def ensure_onnx_model(model_path: str, device: str = "cpu") -> str:
    """Converte un modello .pt in .onnx se necessario"""
    if model_path.endswith('.pt') or model_path.endswith('.pth'):
        # Determina il percorso del file .onnx
        onnx_path = os.path.splitext(model_path)[0] + '.onnx'
        
        # Se il file .onnx non esiste già, convertilo
        if not os.path.exists(onnx_path):
            logger.info(f"Convertendo modello PyTorch {model_path} in formato ONNX...")
            try:
                from utils import convert_yolov8_to_onnx, convert_torch_to_onnx
                
                # Prova prima come modello YOLOv8
                try:
                    import ultralytics
                    onnx_path = convert_yolov8_to_onnx(model_path, onnx_path)
                except (ImportError, Exception) as e:
                    logger.warning(f"Impossibile convertire come modello YOLOv8: {e}")
                    # Fallback alla conversione generica PyTorch
                    onnx_path = convert_torch_to_onnx(model_path, onnx_path)
                
                logger.info(f"Conversione completata: {onnx_path}")
            except Exception as e:
                logger.error(f"Errore nella conversione del modello: {e}")
                return model_path  # Ritorna il modello originale in caso di errore
        else:
            logger.info(f"Trovato modello ONNX esistente: {onnx_path}")
        
        return onnx_path
    
    return model_path  # Ritorna il percorso originale se non è un file .pt


def main():
    """Funzione principale per l'avvio dell'applicazione"""
    parser = argparse.ArgumentParser(description="Advanced Screen Object Detector")
    
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Percorso del modello ONNX")
    parser.add_argument("--device", "-d", type=str, default="auto",
                        choices=["auto", "cuda", "directml", "cpu"],
                        help="Dispositivo di inferenza (auto, cuda, directml, cpu)")
    parser.add_argument("--conf", "-c", type=float, default=0.45,
                        help="Soglia di confidenza per le rilevazioni")
    parser.add_argument("--size", "-s", type=int, default=400,
                        help="Dimensione dell'area di rilevamento")
    parser.add_argument("--log-level", "-l", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Livello di logging")
    
    args = parser.parse_args()
    
    # Imposta livello di logging
    logger.setLevel(getattr(logging, args.log_level))
    
    # Verifica requisiti
    logger.info("Verifica requisiti in corso...")
    if not check_requirements():
        logger.error("Requisiti mancanti. Per favore installa tutti i pacchetti richiesti.")
        return 1
    
    # Trova un modello predefinito se non specificato
    model_path = args.model
    if not model_path:
        model_path = find_default_model()
        if not model_path:
            logger.error("Nessun modello trovato. Specifica un modello con --model.")
            return 1

    # Verifica esistenza modello
    if not os.path.exists(model_path):
        logger.error(f"Modello non trovato: {model_path}")
        return 1

    # Converti in ONNX se necessario
    model_path = ensure_onnx_model(model_path, args.device)
    
    # Importa i moduli solo dopo aver verificato i requisiti
    try:
        from detector import ONNXObjectDetector
        from mouse_controller import MouseController
        from app import DetectorApp
    except ImportError as e:
        logger.error(f"Errore nell'importazione dei moduli: {e}")
        traceback.print_exc()
        return 1
    
    # Disabilita failsafe di pyautogui
    pyautogui.FAILSAFE = False
    
    # Inizializzazione rilevatore
    try:
        logger.info(f"Inizializzazione rilevatore con modello: {model_path}")
        detector = ONNXObjectDetector(
            model_path=model_path,
            conf_threshold=args.conf,
            box_size=args.size,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Errore nell'inizializzazione del rilevatore: {e}")
        traceback.print_exc()
        return 1
    
    # Inizializzazione controller
    controller = MouseController()
    
    # Inizializzazione interfaccia
    try:
        root = tk.Tk()
        root.title("Advanced Screen Object Detector")
        
        # Imposta icona se disponibile
        icon_paths = [
            "icon.ico", 
            "icon.png",
            os.path.join("assets", "icon.ico"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")
        ]
        
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                try:
                    root.iconbitmap(icon_path) if icon_path.endswith(".ico") else None
                    break
                except Exception:
                    pass
        
        # Crea l'applicazione
        app = DetectorApp(root, detector, controller)
        
        logger.info("Avvio dell'applicazione")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Errore nell'avvio dell'applicazione: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())