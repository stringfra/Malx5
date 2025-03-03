import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
from class_manager import load_classes

def test_onnx_model(model_path, image_path=None, conf_threshold=0.25):
    """Testa un modello ONNX con un'immagine o uno schermo vuoto"""
    print("\n--- TEST MODELLO ONNX ---")
    print(f"Modello: {model_path}")
    
    # Verifica provider disponibili
    providers = ort.get_available_providers()
    print(f"Provider disponibili: {providers}")
    
    # Scegli il provider appropriato
    if 'DmlExecutionProvider' in providers:
        provider = 'DmlExecutionProvider'
        print("Utilizzo GPU AMD con DirectML")
    elif 'CUDAExecutionProvider' in providers:
        provider = 'CUDAExecutionProvider'
        print("Utilizzo GPU NVIDIA con CUDA")
    else:
        provider = 'CPUExecutionProvider'
        print("Utilizzo CPU")
    
    # Carica modello
    try:
        session = ort.InferenceSession(model_path, providers=[provider, 'CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"Input name: {input_name}")
        print(f"Input shape: {input_shape}")
        print(f"Output names: {output_names}")
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        return
    
    # Carica classi
    classes = load_classes(model_path)
    
    # Prepara input (immagine o schermo vuoto)
    if image_path and os.path.exists(image_path):
        print(f"Caricamento immagine: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Impossibile caricare l'immagine: {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("Usando un'immagine di test vuota")
        img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Ridimensiona per il modello
    height, width = input_shape[2:] if len(input_shape) == 4 else (640, 640)
    if height <= 0 or width <= 0:  # Dimensioni dinamiche
        height, width = 640, 640
        
    # Prepara input per inferenza    
    input_img = cv2.resize(img, (width, height))
    orig_shape = img.shape[:2]
    
    # Normalizza e trasforma
    input_tensor = input_img.astype(np.float32) / 255.0
    input_tensor = input_tensor.transpose(2, 0, 1)  # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)  # CHW -> NCHW
    
    # Esegui inferenza
    try:
        print("Esecuzione inferenza...")
        outputs = session.run(None, {input_name: input_tensor})
        
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")
            print(f"Output {i} min: {np.min(output)}, max: {np.max(output)}")
            
        # Analisi dei risultati
        print("Analisi risultati...")
        detections = process_output(outputs, classes, orig_shape, conf_threshold)
        print(f"Rilevate {len(detections)} entità")
        
        # Disegna risultati
        if len(detections) > 0:
            result_img = draw_detections(img.copy(), detections)
            output_path = "risultato.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            print(f"Immagine con rilevazioni salvata in: {output_path}")
            
    except Exception as e:
        print(f"Errore nell'inferenza: {e}")
        import traceback
        traceback.print_exc()

def process_output(outputs, classes, orig_shape, conf_threshold=0.25):
    """Elabora l'output del modello ONNX"""
    detections = []
    
    # Determina il tipo di output
    if len(outputs) > 1:
        # Probabilmente formato YOLOv8 con output multipli
        print("Formato output multiplo rilevato")
        # Implementa logica per gestire output multipli
        pass
    else:
        # Output singolo
        output = outputs[0]
        
        # Verifica se è formato YOLOv8
        if output.shape[1] > 7:  # YOLOv8 ha più di 7 elementi per rilevazione
            print("Formato YOLOv8 rilevato")
            # Formato [batch, num_detections, (x, y, w, h, conf, cls_1, cls_2, ...)]
            for i in range(output.shape[1]):
                detection = output[0, i, :]
                confidence = float(detection[4])
                
                if confidence < conf_threshold:
                    continue
                
                # Identifica la classe più probabile
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                combined_score = confidence * class_score
                
                if combined_score < conf_threshold:
                    continue
                
                # Converti coordinate normalizzate
                x, y, w, h = detection[0:4]
                
                # Scala alle dimensioni originali
                orig_h, orig_w = orig_shape
                x1 = int((x - w/2) * orig_w)
                y1 = int((y - h/2) * orig_h)
                x2 = int((x + w/2) * orig_w)
                y2 = int((y + h/2) * orig_h)
                
                detections.append({
                    'class_id': int(class_id),
                    'class_name': classes.get(int(class_id), f"class_{class_id}"),
                    'confidence': combined_score,
                    'box': [x1, y1, x2, y2]
                })
        else:
            print("Formato standard rilevato")
            # Formato standard [batch_id, class_id, confidence, x1, y1, x2, y2]
            for detection in output:
                confidence = float(detection[2])
                if confidence < conf_threshold:
                    continue
                
                # Estrai coordinate normalizzate
                class_id = int(detection[1])
                x1 = int(detection[3] * orig_shape[1])
                y1 = int(detection[4] * orig_shape[0])
                x2 = int(detection[5] * orig_shape[1])
                y2 = int(detection[6] * orig_shape[0])
                
                detections.append({
                    'class_id': class_id,
                    'class_name': classes.get(class_id, f"class_{class_id}"),
                    'confidence': confidence,
                    'box': [x1, y1, x2, y2]
                })
    
    return detections

def draw_detections(image, detections):
    """Disegna le rilevazioni sull'immagine"""
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        class_id = det['class_id']
        class_name = det['class_name']
        confidence = det['confidence']
        
        # Assicurati che le coordinate siano all'interno dell'immagine
        x1 = max(0, min(x1, image.shape[1]-1))
        y1 = max(0, min(y1, image.shape[0]-1))
        x2 = max(0, min(x2, image.shape[1]-1))
        y2 = max(0, min(y2, image.shape[0]-1))
        
        # Disegna il box
        color = tuple(map(int, colors[class_id % len(colors)]))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Prepara il testo
        text = f"{class_name}: {confidence:.2f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Disegna sfondo per il testo
        cv2.rectangle(image, (x1, y1-25), (x1+text_size[0], y1), color, -1)
        
        # Disegna il testo
        cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa un modello ONNX")
    parser.add_argument("--model", type=str, required=True, help="Percorso del modello ONNX")
    parser.add_argument("--image", type=str, help="Percorso di un'immagine di test (opzionale)")
    parser.add_argument("--conf", type=float, default=0.25, help="Soglia di confidenza")
    
    args = parser.parse_args()
    test_onnx_model(args.model, args.image, args.conf)