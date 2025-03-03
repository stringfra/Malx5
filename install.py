#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script di installazione per Advanced Screen Object Detector
----------------------------------------------------------

Installa tutte le dipendenze necessarie per l'applicazione.
Supporta sia GPU NVIDIA che AMD.
"""

import os
import sys
import platform
import subprocess
import argparse
import logging
from pathlib import Path

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("installer")

# Requisiti di base (comuni)
BASE_REQUIREMENTS = [
    "numpy>=1.20.0",
    "opencv-python>=4.5.0",
    "pandas>=1.3.0",
    "pillow>=8.0.0",
    "mss>=6.1.0",
    "pyautogui>=0.9.50",
    "keyboard>=0.13.5",
    "onnxruntime>=1.8.0",
    "pywin32>=300; platform_system=='Windows'",
]

# Requisiti aggiuntivi per CUDA (NVIDIA)
CUDA_REQUIREMENTS = [
    "torch>=1.10.0",
    "torchvision>=0.11.0",
    "onnxruntime-gpu>=1.8.0",
]

# Requisiti aggiuntivi per DirectML (AMD)
DIRECTML_REQUIREMENTS = [
    "onnxruntime-directml>=1.8.0",
]

# Requisiti per conversione modelli
CONVERSION_REQUIREMENTS = [
    "ultralytics>=8.0.0",
    "onnx>=1.10.0",
    "onnxsim>=0.4.0",
]

def check_python_version():
    """Verifica che la versione di Python sia compatibile"""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        logger.error(f"Python 3.8 o superiore richiesto. Versione attuale: {major}.{minor}")
        return False
    logger.info(f"Versione Python compatibile: {major}.{minor}")
    return True

def run_command(command):
    """Esegue un comando e restituisce l'output"""
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Errore nell'esecuzione del comando: {e}")
        return None

def check_cuda():
    """Verifica se CUDA è disponibile"""
    # Controlla se nvcc è disponibile (CUDA compiler)
    nvcc_version = run_command("nvcc --version")
    if nvcc_version:
        logger.info(f"CUDA rilevato: {nvcc_version.split('release')[-1].strip()}")
        return True
        
    # Alternatively check nvidia-smi
    nvidia_smi = run_command("nvidia-smi")
    if nvidia_smi:
        logger.info("CUDA rilevato tramite nvidia-smi")
        return True
        
    logger.info("CUDA non rilevato")
    return False

def check_directml():
    """Verifica se DirectML è supportato (GPU AMD su Windows)"""
    if platform.system() != "Windows":
        logger.info("DirectML supportato solo su Windows")
        return False
        
    # Controlla la presenza di GPU AMD/Intel
    try:
        import wmi
        c = wmi.WMI()
        for gpu in c.Win32_VideoController():
            if "AMD" in gpu.Name or "Radeon" in gpu.Name:
                logger.info(f"GPU AMD rilevata: {gpu.Name}")
                return True
            elif "Intel" in gpu.Name and "Graphics" in gpu.Name:
                logger.info(f"GPU Intel rilevata: {gpu.Name}")
                return True
        logger.info("Nessuna GPU AMD o Intel rilevata per DirectML")
        return False
    except ImportError:
        logger.warning("Pacchetto wmi non disponibile, impossibile verificare GPU AMD/Intel")
        # Prova con dxdiag
        dxdiag = run_command("dxdiag /t dxdiag_output.txt")
        if os.path.exists("dxdiag_output.txt"):
            with open("dxdiag_output.txt", "r") as f:
                content = f.read()
                if "AMD" in content or "Radeon" in content:
                    logger.info("GPU AMD rilevata tramite dxdiag")
                    return True
            # Pulisci il file temporaneo
            os.remove("dxdiag_output.txt")
        return False

def install_requirements(requirements, use_pip=True):
    """Installa i requisiti tramite pip o conda"""
    if use_pip:
        cmd = [sys.executable, "-m", "pip", "install"]
        cmd.extend(requirements)
        cmd_str = " ".join(cmd)
    else:
        # Conda installation
        cmd_str = f"conda install -y {' '.join(requirements)}"
    
    logger.info(f"Esecuzione comando: {cmd_str}")
    result = subprocess.run(cmd_str, shell=True)
    
    if result.returncode != 0:
        logger.error(f"Errore nell'installazione dei pacchetti")
        return False
    return True

def create_virtual_env(env_name, python_version=None):
    """Crea un ambiente virtuale"""
    python_spec = f"python={python_version}" if python_version else "python"
    cmd = f"conda create -y -n {env_name} {python_spec}"
    
    logger.info(f"Creazione ambiente virtuale: {cmd}")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        logger.error(f"Errore nella creazione dell'ambiente virtuale")
        return False
    return True

def download_sample_model():
    """Scarica un modello di esempio"""
    try:
        import urllib.request
        
        # Crea directory modelli se non esiste
        os.makedirs("models", exist_ok=True)
        
        # URL di un modello YOLOv8n pre-addestrato
        model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        model_path = os.path.join("models", "yolov8n.pt")
        
        # Scarica solo se non esiste già
        if not os.path.exists(model_path):
            logger.info(f"Download modello di esempio da {model_url}")
            urllib.request.urlretrieve(model_url, model_path)
            logger.info(f"Modello salvato in {model_path}")
            return True
        else:
            logger.info(f"Modello già esistente in {model_path}")
            return True
    except Exception as e:
        logger.error(f"Errore nel download del modello: {e}")
        return False

def convert_sample_model():
    """Converte il modello di esempio in formato ONNX"""
    try:
        from ultralytics import YOLO
        
        model_path = os.path.join("models", "yolov8n.pt")
        onnx_path = os.path.join("models", "yolov8n.onnx")
        
        if not os.path.exists(model_path):
            logger.error(f"Modello non trovato: {model_path}")
            return False
            
        if os.path.exists(onnx_path):
            logger.info(f"Modello ONNX già esistente in {onnx_path}")
            return True
            
        logger.info(f"Conversione modello {model_path} in ONNX")
        model = YOLO(model_path)
        success = model.export(format="onnx", simplify=True)
        
        if success:
            logger.info(f"Conversione completata: {onnx_path}")
            return True
        else:
            logger.error("Errore nella conversione del modello")
            return False
            
    except Exception as e:
        logger.error(f"Errore nella conversione del modello: {e}")
        return False

def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(description="Installer per Advanced Screen Object Detector")
    
    parser.add_argument("--env", type=str, default="detector",
                        help="Nome dell'ambiente virtuale da creare (solo con conda)")
    parser.add_argument("--python", type=str, default="3.9",
                        help="Versione di Python da usare nell'ambiente virtuale")
    parser.add_argument("--pip", action="store_true", default=True,
                        help="Usa pip invece di conda per l'installazione")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Non installare supporto CUDA anche se disponibile")
    parser.add_argument("--no-directml", action="store_true",
                        help="Non installare supporto DirectML anche se disponibile")
    parser.add_argument("--no-conversion", action="store_true",
                        help="Non installare strumenti di conversione modelli")
    parser.add_argument("--no-download", action="store_true",
                        help="Non scaricare modelli di esempio")
    
    args = parser.parse_args()
    
    # Verifica versione Python
    if not check_python_version():
        return 1
    
    # Determina quali componenti installare
    use_cuda = not args.no_cuda and check_cuda()
    use_directml = not args.no_directml and check_directml()
    
    # Raccogli tutti i requisiti da installare
    requirements = BASE_REQUIREMENTS.copy()
    
    if use_cuda:
        requirements.extend(CUDA_REQUIREMENTS)
        logger.info("Installazione supporto CUDA (NVIDIA GPU)")
    
    if use_directml:
        requirements.extend(DIRECTML_REQUIREMENTS)
        logger.info("Installazione supporto DirectML (AMD GPU)")
    
    if not args.no_conversion:
        requirements.extend(CONVERSION_REQUIREMENTS)
        logger.info("Installazione strumenti conversione modelli")
    
    # Esegui installazione
    logger.info(f"Installazione pacchetti con {'pip' if args.pip else 'conda'}")
    if not install_requirements(requirements, use_pip=args.pip):
        logger.error("Installazione fallita")
        return 1
    
    # Scarica e converti modello di esempio
    if not args.no_download and not args.no_conversion:
        logger.info("Download e conversione modello di esempio")
        if download_sample_model():
            convert_sample_model()
    
    logger.info("Installazione completata con successo!")
    logger.info("Per avviare l'applicazione eseguire: python main.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())