import os
import sys
import logging
import numpy as np
import time
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any

# Configurazione logger
logger = logging.getLogger("detector.utils")

def is_nvidia_gpu_available() -> bool:
    """Verifica se è disponibile una GPU NVIDIA con CUDA"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def is_amd_gpu_available() -> bool:
    """Verifica se è disponibile una GPU AMD con DirectML"""
    try:
        import onnxruntime as ort
        return 'DmlExecutionProvider' in ort.get_available_providers()
    except ImportError:
        return False

def get_available_devices() -> List[str]:
    """Restituisce l'elenco dei dispositivi disponibili"""
    devices = ["cpu"]
    
    if is_nvidia_gpu_available():
        devices.insert(0, "cuda")
    
    if is_amd_gpu_available():
        devices.insert(0 if "cuda" not in devices else 1, "directml")
    
    devices.insert(0, "auto")
    return devices

def convert_torch_to_onnx(model_path: str, output_path: Optional[str] = None, 
                         input_size: Tuple[int, int] = (640, 640), 
                         simplify: bool = True) -> str:
    """
    Converte un modello PyTorch in formato ONNX
    
    Args:
        model_path: Percorso del modello PyTorch (.pt)
        output_path: Percorso di output per il modello ONNX (.onnx)
        input_size: Dimensioni di input (altezza, larghezza)
        simplify: Se True, semplifica il modello ONNX
        
    Returns:
        Percorso del modello ONNX generato
    """
    try:
        import torch
        from torch import nn
        
        # Se output_path non è specificato, usa lo stesso nome del modello input
        if output_path is None:
            output_path = os.path.splitext(model_path)[0] + ".onnx"
        
        logger.info(f"Conversione modello da {model_path} a {output_path}")
        
        # Carica il modello
        model = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Gestisci diversi formati (peso o modello)
        if isinstance(model, dict):
            if 'model' in model:
                model = model['model']  # Estrai il modello da un dizionario
            elif 'state_dict' in model:
                # Questo potrebbe richiedere la definizione della classe del modello
                logger.error("Il modello è un dizionario di stato, è necessaria la definizione della classe")
                raise ValueError("Formato modello non supportato (state_dict senza definizione della classe)")
        
        # Assicurati che sia in modalità eval
        model.eval()
        
        # Crea input dummy
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
        
        # Esporta in ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            verbose=False,
            opset_version=12,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # dynamic batch, h, w
                         'output': {0: 'batch', 1: 'anchors'}}
        )
        
        # Semplifica il modello se richiesto
        # if simplify:
        #     try:
        #         import onnx
                
        #         from onnxsim import simplify as onnx_simplify
                
        #         # Carica modello ONNX
        #         onnx_model = onnx.load(output_path)
                
        #         # Semplifica
        #         simplified_model, check = onnx_simplify(onnx_model)
                
        #         if check:
        #             # Salva il modello semplificato
        #             onnx.save(simplified_model, output_path)
        #             logger.info(f"Modello ONNX semplificato salvato in {output_path}")
        #         else:
        #             logger.warning("La semplificazione del modello ONNX non è riuscita")
        #     except ImportError:
        #         logger.warning("Pacchetti onnx o onnx-simplifier non disponibili, saltando semplificazione")
        
        # logger.info(f"Conversione completata: {output_path}")
        # return output_path
        
    except Exception as e:
        logger.error(f"Errore nella conversione del modello: {e}")
        raise

def optimize_onnx_model(model_path: str, output_path: Optional[str] = None,
                      target_platform: str = "cpu") -> str:
    """
    Ottimizza un modello ONNX per una specifica piattaforma
    
    Args:
        model_path: Percorso del modello ONNX
        output_path: Percorso di output per il modello ottimizzato
        target_platform: Piattaforma target ('cpu', 'cuda', 'directml')
        
    Returns:
        Percorso del modello ottimizzato
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer
        
        # Se output_path non è specificato, aggiungi un suffisso al nome del file originale
        if output_path is None:
            output_path = os.path.splitext(model_path)[0] + f"_opt_{target_platform}.onnx"
        
        logger.info(f"Ottimizzazione modello {model_path} per {target_platform}")
        
        # Carica il modello
        model = onnx.load(model_path)
        
        # Configurazione ottimizzazione
        opt_options = optimizer.OptimizationOptions()
        
        # Impostazioni specifiche per piattaforma
        if target_platform == "cpu":
            # Ottimizzazioni per CPU
            opt_options.enable_gelu = True
            opt_options.enable_layer_norm = True
            opt_options.enable_attention = True
            opt_options.enable_skip_layer_norm = True
            opt_options.enable_bias_gelu = True
            opt_options.enable_bias_skip_layer_norm = True
            opt_options.enable_gelu_approximation = False
        elif target_platform == "cuda":
            # Ottimizzazioni per CUDA
            opt_options.enable_gelu = True
            opt_options.enable_layer_norm = True
            opt_options.enable_attention = True
            opt_options.enable_skip_layer_norm = True
            opt_options.enable_bias_gelu = True
            opt_options.enable_bias_skip_layer_norm = True
            opt_options.enable_gelu_approximation = True
        elif target_platform == "directml":
            # Ottimizzazioni per DirectML (AMD)
            opt_options.enable_gelu = True
            opt_options.enable_layer_norm = True
            opt_options.enable_attention = True
            opt_options.enable_skip_layer_norm = True
            opt_options.enable_bias_gelu = True
            opt_options.enable_bias_skip_layer_norm = True
            opt_options.enable_gelu_approximation = True
        
        # Esegui ottimizzazione
        optimized_model = optimizer.optimize_model(
            model_path,
            model_type="yolo",
            num_heads=8,
            hidden_size=768,
            optimization_options=opt_options
        )
        
        # Salva il modello ottimizzato
        optimized_model.save_model_to_file(output_path)
        
        logger.info(f"Ottimizzazione completata: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Errore nell'ottimizzazione del modello: {e}")
        raise

def benchmark_model(model_path: str, device: str = "cpu", num_runs: int = 50,
                   input_size: Tuple[int, int] = (640, 640)) -> Dict[str, float]:
    """
    Esegue un benchmark delle prestazioni di un modello ONNX
    
    Args:
        model_path: Percorso del modello ONNX
        device: Dispositivo di inferenza (cpu, cuda, directml)
        num_runs: Numero di esecuzioni per il benchmark
        input_size: Dimensioni di input (altezza, larghezza)
        
    Returns:
        Dizionario con i risultati del benchmark
    """
    try:
        import onnxruntime as ort
        
        logger.info(f"Benchmark del modello {model_path} su {device}")
        
        # Configura provider in base al dispositivo
        if device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == "directml":
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Crea una sessione di inferenza
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        
        # Ottieni nome input
        input_name = session.get_inputs()[0].name
        
        # Crea un input dummy
        dummy_input = np.random.rand(1, 3, input_size[0], input_size[1]).astype(np.float32)
        
        # Esegui warm-up
        logger.info("Esecuzione warm-up...")
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # Esegui benchmark
        logger.info(f"Esecuzione benchmark con {num_runs} iterazioni...")
        latencies = []
        
        for i in range(num_runs):
            start_time = time.time()
            session.run(None, {input_name: dummy_input})
            latency = time.time() - start_time
            latencies.append(latency * 1000)  # Converti in ms
        
        # Calcola statistiche
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        fps = 1000 / avg_latency
        
        results = {
            "device": device,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "p95_latency_ms": p95_latency,
            "fps": fps
        }
        
        logger.info(f"Risultati benchmark: Latenza media={avg_latency:.2f}ms, FPS={fps:.2f}")
        return results
        
    except Exception as e:
        logger.error(f"Errore nel benchmark del modello: {e}")
        raise

def is_model_yolov8(model_path: str) -> bool:
    """
    Determina se un modello ONNX è di tipo YOLOv8
    
    Args:
        model_path: Percorso del modello ONNX
        
    Returns:
        True se il modello è YOLOv8, False altrimenti
    """
    try:
        import onnx
        
        # Carica il modello
        model = onnx.load(model_path)
        
        # Analizza gli output
        output_names = [output.name for output in model.graph.output]
        
        # YOLOv8 ha tipicamente output specifici
        if len(output_names) >= 3:
            return True
        
        # Controlla i nomi dei nodi per identificatori YOLOv8
        yolov8_node_patterns = [
            "model.22", "model.23", "model.24",  # YOLOv8 detection heads
            "Conv_245", "Conv_246", "Conv_247"   # Altri pattern comuni in YOLOv8
        ]
        
        for node in model.graph.node:
            for pattern in yolov8_node_patterns:
                if any(pattern in output for output in node.output):
                    return True
        
        return False
        
    except Exception as e:
        logger.error(f"Errore nell'analisi del modello: {e}")
        return False  # In caso di errore, assume non YOLOv8

def detect_model_input_size(model_path: str) -> Tuple[int, int]:
    """
    Rileva le dimensioni di input di un modello ONNX
    
    Args:
        model_path: Percorso del modello ONNX
        
    Returns:
        Tuple con (altezza, larghezza) dell'input
    """
    try:
        import onnx
        
        # Carica il modello
        model = onnx.load(model_path)
        
        # Cerca l'input primario
        input_shape = None
        for input in model.graph.input:
            # L'input delle immagini ha tipicamente 4 dimensioni (N, C, H, W)
            if len(input.type.tensor_type.shape.dim) == 4:
                dims = input.type.tensor_type.shape.dim
                
                # Estrai le dimensioni di altezza e larghezza
                height_dim = dims[2]
                width_dim = dims[3]
                
                # Verifica se sono dimensioni dinamiche
                if height_dim.dim_param:  # Dimensione simbolica/dinamica
                    height = -1  # Indica dimensione dinamica
                else:
                    height = int(height_dim.dim_value)
                
                if width_dim.dim_param:  # Dimensione simbolica/dinamica
                    width = -1  # Indica dimensione dinamica
                else:
                    width = int(width_dim.dim_value)
                
                input_shape = (height, width)
                break
        
        # Se non è stato possibile determinare la dimensione, usa un valore predefinito
        if input_shape is None:
            logger.warning("Impossibile determinare le dimensioni di input, uso (640, 640)")
            return (640, 640)
            
        # Se una o entrambe le dimensioni sono dinamiche, usa un valore predefinito
        if input_shape[0] <= 0 or input_shape[1] <= 0:
            logger.info("Rilevate dimensioni dinamiche, uso (640, 640)")
            return (640, 640)
            
        logger.info(f"Dimensioni input rilevate: {input_shape}")
        return input_shape
        
    except Exception as e:
        logger.error(f"Errore nel rilevamento dimensioni input: {e}")
        return (640, 640)  # Fallback su dimensioni comuni

def load_class_names(model_path: str) -> Dict[int, str]:
    """
    Carica i nomi delle classi da file associati al modello
    
    Args:
        model_path: Percorso del modello
        
    Returns:
        Dizionario con indici e nomi delle classi
    """
    # Possibili posizioni dei file delle classi
    possible_files = [
        os.path.splitext(model_path)[0] + ".txt",
        os.path.join(os.path.dirname(model_path), "classes.txt"),
        os.path.join(os.path.dirname(model_path), "coco.names"),
        os.path.join(os.path.dirname(model_path), os.path.basename(model_path).split(".")[0] + ".txt")
    ]
    
    # Prova a caricare da ciascun file
    for file_path in possible_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    classes = {i: name.strip() for i, name in enumerate(f.readlines())}
                logger.info(f"Caricate {len(classes)} classi da {file_path}")
                return classes
            except Exception as e:
                logger.warning(f"Errore nel caricamento delle classi da {file_path}: {e}")
    
    # Classi COCO predefinite
    logger.warning("Impossibile trovare file delle classi, uso classi COCO predefinite")
    return {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
        45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 
        55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 
        65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
        78: 'hair drier', 79: 'toothbrush'
    }

def convert_yolov8_to_onnx(model_path: str, output_path: Optional[str] = None) -> str:
    """
    Converte un modello YOLOv8 in formato ONNX
    
    Args:
        model_path: Percorso del modello YOLOv8 (.pt)
        output_path: Percorso di output per il modello ONNX (.onnx)
        
    Returns:
        Percorso del modello ONNX generato
    """
    try:
        from ultralytics import YOLO
        
        # Se output_path non è specificato, usa lo stesso nome con estensione diversa
        if output_path is None:
            output_path = os.path.splitext(model_path)[0] + ".onnx"
        
        logger.info(f"Conversione modello YOLOv8 da {model_path} a {output_path}")
        
        # Carica modello
        model = YOLO(model_path)
        
        # Esporta in ONNX
        success = model.export(format="onnx", simplify=True)
        
        if success:
            logger.info(f"Conversione completata: {output_path}")
            return output_path
        else:
            logger.error("Errore nella conversione del modello")
            raise RuntimeError("Conversione fallita")
        
    except Exception as e:
        logger.error(f"Errore nella conversione del modello YOLOv8: {e}")
        raise