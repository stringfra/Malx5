import numpy as np
import cv2
import time
import pandas as pd
import mss
import torch
import onnxruntime as ort
import os
import ctypes
import win32api
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass

@dataclass
class Detection:
    """Classe per rappresentare una singola rilevazione"""
    class_id: int
    class_name: str
    confidence: float
    center_x: int
    center_y: int
    width: int
    height: int
    x1: int
    y1: int
    x2: int
    y2: int
    distance_to_center: float


class ONNXObjectDetector:
    """Detector di oggetti basato su ONNX runtime ottimizzato per prestazioni"""
    
    def __init__(
        self, 
        model_path: str = "model.onnx", 
        conf_threshold: float = 0.45, 
        iou_threshold: float = 0.01,
        box_size: int = 400,
        aim_height_ratio: int = 10,
        device: Optional[str] = None
    ):
        """
        Inizializza il rilevatore di oggetti con modello ONNX
        
        Args:
            model_path: Percorso del modello ONNX
            conf_threshold: Soglia di confidenza per le rilevazioni
            iou_threshold: Soglia IOU per NMS (Non-Maximum Suppression)
            box_size: Dimensione del box di rilevamento (centrato sullo schermo)
            aim_height_ratio: Rapporto per l'altezza del mirino
            device: Dispositivo di esecuzione (auto, cuda, directml, cpu)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.box_size = box_size
        self.aim_height_ratio = aim_height_ratio
        
        # Risoluzione schermo
        self.screen_res_x = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_res_y = ctypes.windll.user32.GetSystemMetrics(1)
        self.screen_center_x = int(self.screen_res_x / 2)
        self.screen_center_y = int(self.screen_res_y / 2)
        
        # Determina il dispositivo da utilizzare
        self.device = self._determine_device(device)
        
        # Carica il modello ONNX
        self._load_model(model_path)
        
        # Genera colori casuali per le classi
        self.colors = np.random.uniform(0, 255, size=(100, 3))
        
        # Inizializza mss per acquisizione schermo (più veloce di pyautogui)
        self.sct = mss.mss()
        
        # Impostazioni capture box (centrato)
        self.update_detection_box()
        
        # Carica nomi delle classi dal file di configurazione
        self.class_names = self._load_class_names(model_path)
    
    def _determine_device(self, device: Optional[str]) -> str:
        """Determina il dispositivo migliore da utilizzare"""
        available_providers = ort.get_available_providers()
        
        print(f"Provider ONNX disponibili: {available_providers}")
        
        if device is None or device == "auto":
            # Priorità: NVIDIA GPU > AMD GPU > CPU
            if 'CUDAExecutionProvider' in available_providers and torch.cuda.is_available():
                device = "cuda"
                print("Utilizzo GPU NVIDIA con CUDA")
            elif 'DmlExecutionProvider' in available_providers:
                device = "directml"
                print("Utilizzo GPU AMD con DirectML")
            else:
                device = "cpu"
                print("Utilizzo CPU")
        
        # Verifica compatibilità con il dispositivo richiesto
        if device == "cuda" and 'CUDAExecutionProvider' not in available_providers:
            print("AVVISO: CUDA richiesto ma non disponibile, utilizzo CPU")
            device = "cpu"
        elif device == "directml" and 'DmlExecutionProvider' not in available_providers:
            print("AVVISO: DirectML richiesto ma non disponibile, utilizzo CPU")
            print("Per supportare GPU AMD, installa onnxruntime-directml con: pip install onnxruntime-directml")
            device = "cpu"
            
        return device
    
    def _load_model(self, model_path: str):
        """Carica il modello ONNX con il provider appropriato"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello non trovato: {model_path}")
        
        # Configura le opzioni in base al dispositivo
        session_options = ort.SessionOptions()
        session_options.enable_profiling = False
        session_options.enable_mem_pattern = True
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Imposta il provider in base al dispositivo
        if self.device == "cuda":
            provider_options = {
                'device_id': 0,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'arena_extend_strategy': 'kNextPowerOfTwo',
            }
            providers = [('CUDAExecutionProvider', provider_options), 'CPUExecutionProvider']
        elif self.device == "directml":
            provider_options = {
                'device_id': 0,
            }
            providers = [('DmlExecutionProvider', provider_options), 'CPUExecutionProvider']
        else:
            # Opzioni CPU ottimizzate
            session_options.intra_op_num_threads = os.cpu_count()
            providers = ['CPUExecutionProvider']
        
        print(f"Caricamento modello {model_path} su {self.device}")
        try:
            self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
            
            # Ottieni informazioni sul modello
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Determina se il modello è YOLOv8 o un altro formato
            self.is_yolov8 = len(self.output_names) >= 3
            print(f"Formato modello: {'YOLOv8' if self.is_yolov8 else 'Generico'}")
            print(f"Input shape: {self.input_shape}")
        except Exception as e:
            print(f"Errore nel caricamento del modello ONNX: {e}")
            raise
    
    def _load_class_names(self, model_path: str) -> Dict[int, str]:
        """Carica i nomi delle classi dal file di configurazione"""
        # Cerca file di classi nella stessa directory del modello
        model_dir = os.path.dirname(model_path)
        class_files = [
            os.path.join(model_dir, "classes.txt"),
            os.path.join(model_dir, "coco.names"),
            os.path.join(model_dir, os.path.basename(model_path).replace(".onnx", ".txt"))
        ]
        
        # Classi COCO predefinite come fallback
        default_classes = {
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
        
        # Prova a caricare da file
        for class_file in class_files:
            if os.path.exists(class_file):
                try:
                    with open(class_file, 'r') as f:
                        classes = {i: name.strip() for i, name in enumerate(f.readlines())}
                    print(f"Caricate {len(classes)} classi da {class_file}")
                    return classes
                except Exception as e:
                    print(f"Errore nel caricamento delle classi da {class_file}: {e}")
        
        print("Utilizzando classi COCO predefinite")
        return default_classes
    
    def update_detection_box(self):
        """Aggiorna le dimensioni e la posizione del box di rilevamento"""
        self.detection_box = {
            'left': int(self.screen_center_x - self.box_size // 2),
            'top': int(self.screen_center_y - self.box_size // 2),
            'width': int(self.box_size),
            'height': int(self.box_size)
        }
    
    def capture_screen(self):
        """Cattura uno screenshot dell'intero schermo"""
        try:
            with mss.mss() as sct:
                monitor = {"top": 0, "left": 0, "width": self.screen_res_x, "height": self.screen_res_y}
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img, dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                return frame
        except Exception as e:
            print(f"Errore nella cattura dello schermo: {e}")
            return np.zeros((self.screen_res_y, self.screen_res_x, 3), dtype=np.uint8)
    
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """Preelabora l'immagine per l'inferenza ONNX"""
        try:
            # Ottieni dimensioni target dal modello (dinamiche o statiche)
            if len(self.input_shape) == 4:
                height, width = self.input_shape[2], self.input_shape[3]
            else:
                height, width = 640, 640  # Default fallback
            
            # Se il modello accetta dimensioni dinamiche, usa le dimensioni originali o un valore standard
            if height <= 0 or width <= 0:
                height, width = 640, 640  # Usa dimensioni standard
            
            # Verifica che le dimensioni siano valide
            if height <= 0 or width <= 0:
                print(f"Dimensioni di input non valide: {height}x{width}, uso 640x640")
                height, width = 640, 640
            
            print(f"Ridimensionamento immagine a {width}x{height}")
            
            # Ridimensiona l'immagine in modo sicuro
            try:
                input_img = cv2.resize(frame, (width, height))
            except Exception as resize_error:
                print(f"Errore nel ridimensionamento: {resize_error}")
                # Crea una tela vuota delle dimensioni corrette
                input_img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Normalizza e trasforma in formato NCHW in modo sicuro
            try:
                input_img = input_img.astype(np.float32) / 255.0
                input_img = np.transpose(input_img, (2, 0, 1))  # HWC -> CHW
                input_img = np.expand_dims(input_img, axis=0)  # CHW -> NCHW
            except Exception as transform_error:
                print(f"Errore nella trasformazione dell'immagine: {transform_error}")
                # Crea un tensor vuoto delle dimensioni corrette
                input_img = np.zeros((1, 3, height, width), dtype=np.float32)
            
            return input_img
        
        except Exception as e:
            print(f"Errore generale nella preelaborazione dell'immagine: {e}")
            # Ritorna un tensor vuoto
            return np.zeros((1, 3, 640, 640), dtype=np.float32)
    
    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Esegue la rilevazione degli oggetti usando ONNX Runtime"""
        try:
            # Preelabora l'immagine
            input_tensor = self.preprocess_image(frame)
            
            # Esegui inferenza
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Debug info
            for i, out in enumerate(outputs):
                print(f"Output {i} shape: {out.shape}")
                
            # Determina automaticamente il tipo di output
            if len(outputs) == 1:
                out_shape = outputs[0].shape
                if len(out_shape) == 3 and out_shape[1] > 5:  # Formato YOLOv8 moderno
                    print("Rilevato formato output YOLOv8 moderno")
                    return self._process_yolov8_output(outputs, frame)
                else:  # Formato generico
                    print("Rilevato formato output generico")
                    return self._process_generic_output(outputs[0], frame)
            else:
                print("Formato output non supportato")
                return []
                
        except Exception as e:
            print(f"Errore nel rilevamento: {e}")
            return []
        
    def _process_yolov8_output(self, outputs: List[np.ndarray], frame: np.ndarray) -> List[Detection]:
        """Processa l'output di un modello YOLOv8 nel formato (1, 8, 8400)"""
        try:
            # Prendiamo il primo output
            output = outputs[0]
            orig_height, orig_width = frame.shape[:2]
            
            # Debug info
            print(f"Processing YOLOv8 output with shape: {output.shape}")
            
            detections = []
            
            # Se l'output ha formato (1, 8, 8400) o simile
            if len(output.shape) == 3 and output.shape[1] > 5:
                # Trasponi l'output per ottenere (num_detections, dimensions)
                # Da (1, 8, 8400) a (8400, 8)
                transposed_output = output[0].T
                
                # Numero di classi
                num_classes = output.shape[1] - 5  # Sottraiamo 5 per x,y,w,h,conf
                
                # Processa ogni rilevazione
                for i in range(transposed_output.shape[0]):
                    try:
                        # Confidenza oggetto generale
                        confidence = float(transposed_output[i, 4])
                        
                        # Filtra per confidenza
                        if confidence < self.conf_threshold:
                            continue
                        
                        # Trova la classe con il punteggio più alto
                        class_scores = transposed_output[i, 5:5+num_classes]
                        class_id = int(np.argmax(class_scores))
                        class_score = float(class_scores[class_id])
                        
                        # Confidenza finale
                        final_confidence = float(confidence * class_score)
                        if final_confidence < self.conf_threshold:
                            continue
                        
                        # Coordinate normalizzate
                        x, y, w, h = [float(val) for val in transposed_output[i, 0:4]]
                        
                        # Converti in coordinate pixel
                        x1 = max(0, int((x - w/2) * orig_width))
                        y1 = max(0, int((y - h/2) * orig_height))
                        x2 = min(orig_width, int((x + w/2) * orig_width))
                        y2 = min(orig_height, int((y + h/2) * orig_height))
                        
                        # Ignora box non validi
                        if x1 >= x2 or y1 >= y2:
                            continue
                            
                        # Ignora rilevazioni proprie
                        if x1 < 15 or (x1 < orig_width/5 and y2 > orig_height/1.2):
                            continue
                        
                        # Calcola dati per targeting
                        width = x2 - x1
                        height = y2 - y1
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2 - height / self.aim_height_ratio)
                        
                        # Distanza dal centro
                        box_center_x = orig_width // 2
                        box_center_y = orig_height // 2
                        distance = float(((center_x - box_center_x)**2 + (center_y - box_center_y)**2)**0.5)
                        
                        # Crea oggetto Detection
                        detection = Detection(
                            class_id=class_id,
                            class_name=self.class_names.get(class_id, f"class_{class_id}"),
                            confidence=final_confidence,
                            center_x=center_x,
                            center_y=center_y,
                            width=width,
                            height=height,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            distance_to_center=distance
                        )
                        
                        detections.append(detection)
                    except Exception as e:
                        print(f"Errore nell'elaborazione della rilevazione {i}: {e}")
                        continue
            else:
                print(f"Formato output non supportato: {output.shape}")
            
            print(f"Rilevati {len(detections)} oggetti")
            return detections
            
        except Exception as e:
            print(f"Errore generale nell'elaborazione dell'output YOLOv8: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    def _process_generic_output(self, output: np.ndarray, frame: np.ndarray) -> List[Detection]:
        """Processa l'output di un modello generico di object detection"""
        try:
            orig_height, orig_width = frame.shape[:2]
            detections = []
            
            # Debug info
            print(f"Shape of generic output: {output.shape}")
            
            # Formato generico: [n, 7] dove n è il numero di rilevazioni
            # e 7 rappresenta [batch_id, class_id, confidence, x1, y1, x2, y2]
            for i in range(output.shape[0]):
                try:
                    # Estrai valori in modo sicuro
                    confidence = float(output[i, 2])
                    if confidence < self.conf_threshold:
                        continue
                    
                    # Estrai class_id con conversione sicura
                    class_id_val = output[i, 1]
                    if isinstance(class_id_val, np.ndarray):
                        if class_id_val.size > 0:
                            class_id = int(class_id_val.item())
                        else:
                            continue
                    else:
                        class_id = int(class_id_val)
                    
                    # Estrai coordinate normalizzate con conversione esplicita
                    x1 = int(float(output[i, 3]) * orig_width)
                    y1 = int(float(output[i, 4]) * orig_height)
                    x2 = int(float(output[i, 5]) * orig_width)
                    y2 = int(float(output[i, 6]) * orig_height)
                    
                    # Verifica validità del bounding box
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    # Ignora rilevazioni proprie
                    if x1 < 15 or (x1 < orig_width/5 and y2 > orig_height/1.2):
                        continue
                    
                    # Calcola dati aggiuntivi per targeting
                    width = x2 - x1
                    height = y2 - y1
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2 - height / self.aim_height_ratio)
                    
                    # Calcola distanza dal centro
                    box_center_x = orig_width // 2
                    box_center_y = orig_height // 2
                    distance = float(((center_x - box_center_x)**2 + (center_y - box_center_y)**2)**0.5)
                    
                    # Crea oggetto Detection
                    detection_obj = Detection(
                        class_id=class_id,
                        class_name=self.class_names.get(class_id, f"class_{class_id}"),
                        confidence=confidence,
                        center_x=center_x,
                        center_y=center_y,
                        width=width,
                        height=height,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        distance_to_center=distance
                    )
                    
                    detections.append(detection_obj)
                except Exception as e:
                    print(f"Errore nell'elaborazione della rilevazione {i}: {e}")
                    continue
            
            return detections
        
        except Exception as e:
            print(f"Errore generale nell'elaborazione dell'output generico: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_detections(self, detections: List[Detection], frame: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Elabora le rilevazioni e prepara le informazioni di visualizzazione"""
        annotated_frame = frame.copy()
        
        if not detections:
            return pd.DataFrame(), annotated_frame
        
        # Disegna le rilevazioni e crea dataframe
        detection_data = []
        for detection in detections:
            # Aggiungi dati al dataframe
            detection_data.append({
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "center_x": detection.center_x,
                "center_y": detection.center_y,
                "width": detection.width,
                "height": detection.height,
                "x1": detection.x1,
                "y1": detection.y1,
                "x2": detection.x2,
                "y2": detection.y2,
                "distance_to_center": detection.distance_to_center
            })
            
            # Disegna box e informazioni
            color = self.colors[detection.class_id % len(self.colors)]
            cv2.rectangle(annotated_frame, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
            
            # Disegna mirino al centro della rilevazione
            cv2.circle(annotated_frame, (detection.center_x, detection.center_y), 3, (0, 255, 0), -1)
            
            # Linea dal centro al target
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            cv2.line(annotated_frame, (detection.center_x, detection.center_y), (center_x, center_y), (255, 255, 0), 1)
            
            # Etichetta con nome classe e confidenza
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(annotated_frame, label, (detection.x1, detection.y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Disegna indicatore al centro dello schermo
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        cv2.line(annotated_frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
        cv2.line(annotated_frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)
        
        # Crea dataframe e ordina per distanza dal centro
        df = pd.DataFrame(detection_data)
        if not df.empty:
            df = df.sort_values('distance_to_center')
            
        return df, annotated_frame
    
    def detect_screen(self) -> Tuple[pd.DataFrame, np.ndarray, float]:
        """Rileva oggetti dallo schermo e calcola FPS"""
        start_time = time.time()
        
        # Cattura lo screenshot
        frame = self.capture_screen()
        
        # Rileva oggetti
        detections = self.detect_objects(frame)
        
        # Elabora le rilevazioni e prepara il frame annotato
        detections_df, annotated_frame = self.process_detections(detections, frame)
        
        # Calcola FPS
        fps = 1.0 / (time.time() - start_time)
        
        return detections_df, annotated_frame, fps
    
    def is_target_locked(self, abs_x: int, abs_y: int, threshold: int = 5) -> bool:
        """Verifica se il target è al centro dello schermo"""
        return (self.screen_center_x - threshold <= abs_x <= self.screen_center_x + threshold and 
                self.screen_center_y - threshold <= abs_y <= self.screen_center_y + threshold)