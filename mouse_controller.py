import time
import ctypes
import win32api
import threading
from typing import Tuple, Optional, List, Generator, Dict

class MouseController:
    """Controller del mouse per assistenza al puntamento"""
    
    def __init__(self):
        """Inizializza il controller del mouse"""
        # Impostazioni per il movimento del mouse
        self.mouse_delay = 0.0009
        self.settings = {
            "targeting_scale": 0.5,    # Fattore di scala per mirino
            "lock_threshold": 5,       # Soglia per considerare target "bloccato"
            "crosshair_trigger": False # Auto-fire quando il target è centrato
        }
        
        # Stato del controller
        self.auto_control = False
        self.target_class = None
        self.control_thread = None
        self.screen_center_x = 0
        self.screen_center_y = 0
        
        # Inizializza posizione del box di rilevamento
        self.detection_box_left = 0
        self.detection_box_top = 0
    
    def update_settings(self, **kwargs):
        """Aggiorna le impostazioni del controller"""
        for key, value in kwargs.items():
            if key in self.settings:
                self.settings[key] = value
    
    def is_right_button_pressed(self) -> bool:
        """Verifica se il pulsante destro è premuto (targeting)"""
        return win32api.GetKeyState(0x02) in (-127, -128)
    
    def left_click(self):
        """Simula un click sinistro"""
        ctypes.windll.user32.mouse_event(0x0002)  # left mouse down
        time.sleep(0.0001)
        ctypes.windll.user32.mouse_event(0x0004)  # left mouse up
    
    def interpolate_coordinates(self, target_x: int, target_y: int, scale: float = 0.5) -> Generator[Tuple[int, int], None, None]:
        """Genera le coordinate per un movimento fluido del mouse"""
        # Calcola la differenza tra la posizione attuale e il target
        diff_x = (target_x - self.screen_center_x) * scale
        diff_y = (target_y - self.screen_center_y) * scale
        
        # Calcola la lunghezza del percorso
        length = int(((diff_x**2 + diff_y**2)**0.5))
        if length == 0:
            return
            
        # Calcola l'incremento unitario
        unit_x = diff_x / length
        unit_y = diff_y / length
        
        # Genera le coordinate intermedie con correzione per il rounding
        x = y = sum_x = sum_y = 0
        for k in range(length):
            sum_x += x
            sum_y += y
            x, y = round(unit_x * k - sum_x), round(unit_y * k - sum_y)
            yield x, y
    
    def move_mouse(self, target_x: int, target_y: int) -> bool:
        """
        Muove il mouse verso il target in modo fluido
        
        Ritorna:
            True se il movimento è stato eseguito, False altrimenti
        """
        # Verifica che il tasto destro sia premuto per attivare il targeting
        if not self.is_right_button_pressed():
            return False
            
        # Ottieni il fattore di scala
        scale = self.settings["targeting_scale"]
        
        # Definizione strutture per SendInput (più efficiente di mouse_event)
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            
        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long),
                        ("dy", ctypes.c_long),
                        ("mouseData", ctypes.c_ulong),
                        ("dwFlags", ctypes.c_ulong),
                        ("time", ctypes.c_ulong),
                        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]
                        
        class Input_I(ctypes.Union):
            _fields_ = [("mi", MouseInput)]
            
        class Input(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong),
                        ("ii", Input_I)]
        
        # Preparazione per SendInput
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        
        # Interpola e muovi il mouse con movimento fluido
        for rel_x, rel_y in self.interpolate_coordinates(target_x, target_y, scale):
            ii_.mi = MouseInput(rel_x, rel_y, 0, 0x0001, 0, ctypes.pointer(extra))
            input_obj = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.byref(input_obj), ctypes.sizeof(input_obj))
            time.sleep(self.mouse_delay)
        
        return True
    
    def is_target_locked(self, abs_x: int, abs_y: int) -> bool:
        """Verifica se il target è al centro dello schermo"""
        threshold = self.settings["lock_threshold"]
        return (self.screen_center_x - threshold <= abs_x <= self.screen_center_x + threshold and 
                self.screen_center_y - threshold <= abs_y <= self.screen_center_y + threshold)
    
    def start_auto_control(self, 
                           detection_callback,
                           screen_center_x: int,
                           screen_center_y: int,
                           target_class: Optional[str] = None):
        """
        Avvia il controllo automatico
        
        Args:
            detection_callback: Funzione che fornisce le rilevazioni correnti
            screen_center_x: Coordinata X del centro dello schermo
            screen_center_y: Coordinata Y del centro dello schermo
            target_class: Classe target da tracciare (se None, traccia qualsiasi oggetto)
        """
        self.screen_center_x = screen_center_x
        self.screen_center_y = screen_center_y
        self.target_class = target_class
        self.auto_control = True
        self.detection_callback = detection_callback
        
        if self.control_thread is None or not self.control_thread.is_alive():
            self.control_thread = threading.Thread(target=self._auto_control_loop)
            self.control_thread.daemon = True
            self.control_thread.start()
            print(f"Controllo automatico avviato{' per classe ' + target_class if target_class else ''}")
        else:
            print("Thread di controllo già attivo")
    
    def stop_auto_control(self):
        """Ferma il controllo automatico"""
        if not self.auto_control:
            return
            
        self.auto_control = False
        
        if self.control_thread and self.control_thread.is_alive():
            # Aspetta che il thread termini, con timeout per sicurezza
            self.control_thread.join(timeout=1.0)
            print("Controllo automatico fermato")
    
    def _auto_control_loop(self):
        """Loop principale per il controllo automatico"""
        previous_target = None
        target_lost_counter = 0
        
        while self.auto_control:
            try:
                # Ottieni le rilevazioni attuali
                detections_df = self.detection_callback()
                
                # Salta se non ci sono rilevazioni
                if detections_df.empty:
                    time.sleep(0.01)
                    target_lost_counter += 1
                    # Se perdiamo il target per troppo tempo, resettiamo
                    if target_lost_counter > 50 and previous_target is not None:
                        previous_target = None
                    continue
                
                # Filtra per classe target se specificata
                if self.target_class is not None:
                    targets = detections_df[detections_df['class_name'] == self.target_class]
                    if targets.empty:
                        time.sleep(0.01)
                        continue
                else:
                    targets = detections_df
                
                # Reset contatore
                target_lost_counter = 0
                
                # Selezione del target
                # Se abbiamo un target precedente, tenta di mantenerlo per continuità
                if previous_target is not None:
                    # Cerca il target precedente nelle rilevazioni attuali
                    # Usa una combinazione di distanza e similarità di dimensioni per identificarlo
                    for _, target in targets.iterrows():
                        # Se è lo stesso target (basato su overlap o distanza), continua a seguirlo
                        if self._is_same_target(previous_target, target):
                            closest_target = target
                            break
                    else:
                        # Se non troviamo lo stesso target, prendi il più vicino
                        closest_target = targets.iloc[0]
                else:
                    # Prendi il target più vicino al centro
                    closest_target = targets.iloc[0]  # È già ordinato per distanza
                
                # Aggiorna il target precedente
                previous_target = closest_target.copy()
                
                # Converti coordinate relative in assolute
                abs_x = closest_target['center_x'] + self.detection_box_left
                abs_y = closest_target['center_y'] + self.detection_box_top
                
                # Verifica se il target è bloccato
                if self.is_target_locked(abs_x, abs_y):
                    if self.settings["crosshair_trigger"] and self.is_right_button_pressed():
                        self.left_click()
                else:
                    # Muovi il mouse verso il target
                    self.move_mouse(abs_x, abs_y)
                
            except Exception as e:
                print(f"Errore nel controllo automatico: {e}")
            
            # Breve pausa per ridurre l'uso della CPU
            time.sleep(0.005)
    
    def _is_same_target(self, prev_target, curr_target, iou_threshold=0.3, dist_threshold=100):
        """
        Verifica se due rilevazioni si riferiscono allo stesso target
        usando overlap (IoU) e distanza
        """
        # Se sono classi diverse, non è lo stesso target
        if prev_target['class_id'] != curr_target['class_id']:
            return False
        
        # Calcola distanza tra i centri
        prev_cx = prev_target['center_x']
        prev_cy = prev_target['center_y']
        curr_cx = curr_target['center_x']
        curr_cy = curr_target['center_y']
        
        dist = ((prev_cx - curr_cx)**2 + (prev_cy - curr_cy)**2)**0.5
        
        # Se la distanza è piccola, considera lo stesso target
        if dist < dist_threshold:
            return True
        
        # Calcola IoU (Intersection over Union)
        # Coordinate dei box
        prev_x1, prev_y1 = prev_target['x1'], prev_target['y1']
        prev_x2, prev_y2 = prev_target['x2'], prev_target['y2']
        curr_x1, curr_y1 = curr_target['x1'], curr_target['y1']
        curr_x2, curr_y2 = curr_target['x2'], curr_target['y2']
        
        # Calcola intersezione
        x_left = max(prev_x1, curr_x1)
        y_top = max(prev_y1, curr_y1)
        x_right = min(prev_x2, curr_x2)
        y_bottom = min(prev_y2, curr_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return False  # No overlap
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calcola aree
        prev_area = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
        curr_area = (curr_x2 - curr_x1) * (curr_y2 - curr_y1)
        
        # IoU
        iou = intersection_area / float(prev_area + curr_area - intersection_area)
        
        return iou > iou_threshold
    
    def set_detection_box_position(self, left: int, top: int):
        """Imposta la posizione del box di rilevamento"""
        self.detection_box_left = left
        self.detection_box_top = top