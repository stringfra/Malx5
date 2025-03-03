import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import time
import threading
import keyboard
import numpy as np
import pyautogui
import os
import sys
from collections import deque
from typing import Optional, Dict, List, Tuple

import pandas as pd
# Importa i moduli personalizzati
from detector import ONNXObjectDetector
from mouse_controller import MouseController

class FPSCounter:
    """Classe per il calcolo e la visualizzazione FPS con media mobile"""
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.last_time = time.time()
        self.frame_count = 0
    
    def update(self):
        """Aggiorna il contatore FPS"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        # Aggiorna FPS ogni 0.5 secondi
        if elapsed >= 0.5:
            fps = self.frame_count / elapsed
            self.fps_history.append(fps)
            self.frame_count = 0
            self.last_time = current_time
            
            return True
        return False
    
    def get_fps(self) -> float:
        """Restituisce l'FPS medio corrente"""
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)

class DetectorApp:
    """Interfaccia utente per il rilevatore di oggetti su schermo"""
    
    def __init__(self, root, detector: ONNXObjectDetector, controller: MouseController):
        """
        Inizializza l'applicazione
        
        Args:
            root: Root window Tkinter
            detector: Istanza del detector di oggetti
            controller: Istanza del controller del mouse
        """
        self.root = root
        self.detector = detector
        self.controller = controller
        
        # Variabili di stato
        self.running = True
        self.fps_counter = FPSCounter(window_size=30)
        self.detection_active = True
        self.last_key_press_time = 0
        
        # Passa le coordinate di riferimento al controller
        self.controller.screen_center_x = self.detector.screen_center_x
        self.controller.screen_center_y = self.detector.screen_center_y
        
        # Configura la finestra principale
        self._configure_root()
        
        # Crea l'interfaccia
        self._create_ui()
        
        # Imposta hotkey globali
        self._setup_hotkeys()
        
        # Avvia il thread di rilevamento
        self._start_detection_thread()
    
    def _configure_root(self):
        """Configura la finestra principale"""
        self.root.title("Advanced Object Detector")
        self.root.attributes('-topmost', True)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        
        # Imposta dimensione e posizione
        self.root.geometry("1200x700")
        self.root.minsize(900, 600)
        
        # Applica un tema scuro (opzionale)
        self._apply_dark_theme()
    
    def _apply_dark_theme(self):
        """Applica un tema scuro all'interfaccia"""
        # Colori del tema
        bg_color = "#2E2E2E"
        fg_color = "#E0E0E0"
        accent_color = "#007ACC"  # Blu
        
        # Applica stile
        style = ttk.Style()
        style.theme_use('clam')  # Usa il tema 'clam' come base
        
        # Configura elementi principali
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TButton", background=bg_color, foreground=fg_color)
        style.configure("TCheckbutton", background=bg_color, foreground=fg_color)
        style.configure("TScale", background=bg_color, troughcolor="#555555")
        
        # Configura menu e combobox
        style.configure("TMenubutton", background=bg_color, foreground=fg_color)
        style.configure("TCombobox", fieldbackground=bg_color, background=bg_color, foreground=fg_color)
        
        # Configura notebook (tabs)
        style.configure("TNotebook", background=bg_color, foreground=fg_color)
        style.configure("TNotebook.Tab", background=bg_color, foreground=fg_color, padding=[10, 2])
        style.map("TNotebook.Tab",
                 background=[("selected", accent_color)],
                 foreground=[("selected", "#FFFFFF")])
        
        # Configura la finestra principale
        self.root.configure(bg=bg_color)
    
    def _create_ui(self):
        """Crea l'interfaccia utente"""
        # Configura il layout principale
        main_frame = ttk.Frame(self.root, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dividi in due sezioni principali
        left_panel = ttk.Frame(main_frame, padding=5, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)  # Mantiene la larghezza fissa
        
        right_panel = ttk.Frame(main_frame, padding=5)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Crea notebook per categorie impostazioni
        settings_notebook = ttk.Notebook(left_panel)
        settings_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 1. Tab Rilevamento
        detection_tab = ttk.Frame(settings_notebook, padding=10)
        self._create_detection_settings(detection_tab)
        settings_notebook.add(detection_tab, text="Rilevamento")
        
        # 2. Tab Assistenza al puntamento
        aiming_tab = ttk.Frame(settings_notebook, padding=10)
        self._create_aiming_settings(aiming_tab)
        settings_notebook.add(aiming_tab, text="Assistenza")
        
        # 3. Tab Modello
        model_tab = ttk.Frame(settings_notebook, padding=10)
        self._create_model_settings(model_tab)
        settings_notebook.add(model_tab, text="Modello")
        
        # Pannello inferiore con informazioni status
        status_frame = ttk.Frame(left_panel, padding=5)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Barra stato
        self.status_var = tk.StringVar(value="Stato: Inattivo")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 10))
        status_label.pack(fill=tk.X, pady=2)
        
        # FPS
        self.fps_var = tk.StringVar(value="FPS: 0")
        fps_label = ttk.Label(status_frame, textvariable=self.fps_var, font=("Arial", 10))
        fps_label.pack(fill=tk.X, pady=2)
        
        # Pulsanti di controllo
        control_frame = ttk.Frame(left_panel, padding=5)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Pulsante avvia/ferma
        self.start_stop_var = tk.StringVar(value="Ferma rilevamento")
        self.start_stop_btn = ttk.Button(
            control_frame, 
            textvariable=self.start_stop_var,
            command=self.toggle_detection
        )
        self.start_stop_btn.pack(fill=tk.X, pady=2)
        
        # Pulsante chiusura
        quit_btn = ttk.Button(control_frame, text="Chiudi applicazione", command=self.quit)
        quit_btn.pack(fill=tk.X, pady=2)
        
        # Pannello aiuto
        help_frame = ttk.LabelFrame(left_panel, text="Tasti rapidi", padding=5)
        help_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(help_frame, text="F1: Attiva/disattiva assistenza").pack(anchor=tk.W)
        ttk.Label(help_frame, text="F2: Attiva/disattiva rilevamento").pack(anchor=tk.W)
        ttk.Label(help_frame, text="F3: Chiudi applicazione").pack(anchor=tk.W)
        ttk.Label(help_frame, text="Tasto destro: Attiva targeting").pack(anchor=tk.W)
        
        # Canvas per la visualizzazione
        self.canvas_frame = ttk.Frame(right_panel)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Ridimensionamento del canvas al cambio dimensioni finestra
        self.canvas.bind("<Configure>", self._on_canvas_resize)
    
    def _create_detection_settings(self, parent):
        """Crea i controlli per le impostazioni di rilevamento"""
        # Confidenza
        ttk.Label(parent, text="Soglia confidenza:").pack(anchor=tk.W, pady=(0, 2))
        conf_frame = ttk.Frame(parent)
        conf_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.conf_var = tk.DoubleVar(value=self.detector.conf_threshold)
        conf_scale = ttk.Scale(
            conf_frame, 
            from_=0.05, 
            to=0.95, 
            variable=self.conf_var,
            command=lambda _: self._update_confidence()
        )
        conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        conf_label = ttk.Label(conf_frame, textvariable=self.conf_var, width=5)
        conf_label.pack(side=tk.RIGHT)
        
        # Dimensione box
        ttk.Label(parent, text="Dimensione area:").pack(anchor=tk.W, pady=(0, 2))
        box_frame = ttk.Frame(parent)
        box_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.box_var = tk.IntVar(value=self.detector.box_size)
        box_scale = ttk.Scale(
            box_frame, 
            from_=200, 
            to=1000, 
            variable=self.box_var,
            command=lambda _: self._update_box_size()
        )
        box_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        box_label = ttk.Label(box_frame, textvariable=self.box_var, width=5)
        box_label.pack(side=tk.RIGHT)
        
        # NMS (Non-Maximum Suppression)
        ttk.Label(parent, text="Soglia IOU:").pack(anchor=tk.W, pady=(0, 2))
        iou_frame = ttk.Frame(parent)
        iou_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.iou_var = tk.DoubleVar(value=self.detector.iou_threshold)
        iou_scale = ttk.Scale(
            iou_frame, 
            from_=0.01, 
            to=0.5, 
            variable=self.iou_var,
            command=lambda _: self._update_iou()
        )
        iou_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        iou_label = ttk.Label(iou_frame, textvariable=self.iou_var, width=5)
        iou_label.pack(side=tk.RIGHT)
        
        # Priorità performance
        ttk.Label(parent, text="Priorità:").pack(anchor=tk.W, pady=(0, 2))
        performance_frame = ttk.Frame(parent)
        performance_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.performance_var = tk.StringVar(value="bilanciato")
        ttk.Radiobutton(
            performance_frame, 
            text="Precisione", 
            value="precision", 
            variable=self.performance_var,
            command=self._update_performance
        ).pack(side=tk.LEFT)
        
        ttk.Radiobutton(
            performance_frame, 
            text="Bilanciato", 
            value="balanced", 
            variable=self.performance_var,
            command=self._update_performance
        ).pack(side=tk.LEFT)
        
        ttk.Radiobutton(
            performance_frame, 
            text="Velocità", 
            value="speed", 
            variable=self.performance_var,
            command=self._update_performance
        ).pack(side=tk.LEFT)
    
    def _create_aiming_settings(self, parent):
        """Crea i controlli per le impostazioni di assistenza al puntamento"""
        # Attiva/disattiva assistenza
        assistance_frame = ttk.Frame(parent)
        assistance_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.assist_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            assistance_frame, 
            text="Abilita assistenza (F1)", 
            variable=self.assist_var,
            command=self._toggle_aim_assist
        ).pack(side=tk.LEFT, fill=tk.X)
        
        # Trigger automatico
        trigger_frame = ttk.Frame(parent)
        trigger_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.trigger_var = tk.BooleanVar(value=self.controller.settings["crosshair_trigger"])
        ttk.Checkbutton(
            trigger_frame, 
            text="Auto-fire quando centrato", 
            variable=self.trigger_var,
            command=self._toggle_trigger
        ).pack(side=tk.LEFT, fill=tk.X)
        
        # Classe target
        ttk.Label(parent, text="Classe target:").pack(anchor=tk.W, pady=(0, 2))
        
        self.target_class_var = tk.StringVar()
        self.target_combo = ttk.Combobox(parent, textvariable=self.target_class_var)
        self.target_combo.pack(fill=tk.X, pady=(0, 10))
        self._update_class_list()
        
        # Altezza mira
        ttk.Label(parent, text="Altezza mira:").pack(anchor=tk.W, pady=(0, 2))
        height_frame = ttk.Frame(parent)
        height_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.aim_height_var = tk.IntVar(value=self.detector.aim_height_ratio)
        height_scale = ttk.Scale(
            height_frame, 
            from_=1, 
            to=20, 
            variable=self.aim_height_var,
            command=lambda _: self._update_aim_height()
        )
        height_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        height_label = ttk.Label(height_frame, textvariable=self.aim_height_var, width=5)
        height_label.pack(side=tk.RIGHT)
        
        # Sensibilità
        ttk.Label(parent, text="Sensibilità:").pack(anchor=tk.W, pady=(0, 2))
        sens_frame = ttk.Frame(parent)
        sens_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.sens_var = tk.DoubleVar(value=self.controller.settings["targeting_scale"])
        sens_scale = ttk.Scale(
            sens_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.sens_var,
            command=lambda _: self._update_sensitivity()
        )
        sens_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        sens_label = ttk.Label(sens_frame, textvariable=self.sens_var, width=5)
        sens_label.pack(side=tk.RIGHT)
        
        # Soglia lock-on
        ttk.Label(parent, text="Soglia lock-on:").pack(anchor=tk.W, pady=(0, 2))
        lock_frame = ttk.Frame(parent)
        lock_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lock_var = tk.IntVar(value=self.controller.settings["lock_threshold"])
        lock_scale = ttk.Scale(
            lock_frame, 
            from_=1, 
            to=20, 
            variable=self.lock_var,
            command=lambda _: self._update_lock_threshold()
        )
        lock_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        lock_label = ttk.Label(lock_frame, textvariable=self.lock_var, width=5)
        lock_label.pack(side=tk.RIGHT)
    
    def _create_model_settings(self, parent):
        """Crea i controlli per le impostazioni del modello"""
        # Informazioni sul modello corrente
        info_frame = ttk.LabelFrame(parent, text="Informazioni modello")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_path_var = tk.StringVar(value="Nessun modello caricato")
        ttk.Label(info_frame, textvariable=self.model_path_var).pack(anchor=tk.W, pady=2)
        
        self.device_var = tk.StringVar(value=f"Dispositivo: {self.detector.device}")
        ttk.Label(info_frame, textvariable=self.device_var).pack(anchor=tk.W, pady=2)
        
        self.classes_var = tk.StringVar(value=f"Classi: {len(self.detector.class_names)}")
        ttk.Label(info_frame, textvariable=self.classes_var).pack(anchor=tk.W, pady=2)
        
        # Controlli caricamento modello
        model_frame = ttk.Frame(parent)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            model_frame, 
            text="Carica modello ONNX", 
            command=self._load_model_dialog
        ).pack(fill=tk.X, pady=2)
        
        # Dispositivo di inferenza
        device_frame = ttk.Frame(parent)
        device_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(device_frame, text="Dispositivo di inferenza:").pack(anchor=tk.W, pady=(0, 2))
        
        self.device_option_var = tk.StringVar(value=self.detector.device)
        device_combo = ttk.Combobox(
            device_frame, 
            textvariable=self.device_option_var,
            values=["auto", "cuda", "directml", "cpu"]
        )
        device_combo.pack(fill=tk.X, pady=(0, 5))
        device_combo.bind("<<ComboboxSelected>>", lambda _: self._device_selected())
        
        ttk.Button(
            device_frame, 
            text="Applica cambio dispositivo", 
            command=self._reload_model
        ).pack(fill=tk.X, pady=2)
    
    def _setup_hotkeys(self):
        """Configura i tasti rapidi globali"""
        keyboard.on_press_key("f1", lambda _: self._hotkey_callback("f1"))
        keyboard.on_press_key("f2", lambda _: self._hotkey_callback("f2"))
        keyboard.on_press_key("f3", lambda _: self._hotkey_callback("f3"))
    
    def _hotkey_callback(self, key):
        """Gestisce gli eventi dei tasti rapidi con debounce"""
        current_time = time.time()
        if current_time - self.last_key_press_time < 0.3:
            return  # Ignora pressioni troppo ravvicinate
        
        self.last_key_press_time = current_time
        
        if key == "f1":
            self.assist_var.set(not self.assist_var.get())
            self._toggle_aim_assist()
        elif key == "f2":
            self.toggle_detection()
        elif key == "f3":
            self.quit()
    
    def _start_detection_thread(self):
        """Avvia il thread dedicato al rilevamento"""
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Avvia anche un timer per aggiornare l'interfaccia
        self.root.after(100, self._update_ui)
    
    def _detection_loop(self):
        """Loop principale per il rilevamento"""
        last_detection_time = time.time()
        frame_buffer = None
        
        while self.running:
            try:
                if not self.detection_active:
                    time.sleep(0.1)
                    continue
                
                # Misura il tempo di esecuzione
                start_time = time.time()
                
                # Rileva oggetti dallo schermo
                detections_df, annotated_frame, _ = self.detector.detect_screen()
                
                # Memorizza il frame per l'interfaccia utente
                frame_buffer = annotated_frame.copy()
                
                # Aggiorna FPS counter
                self.fps_counter.update()
                
                # Aggiorna tempo ultima rilevazione
                last_detection_time = time.time()
                
                # Calcola tempo di esecuzione rilevamento
                detection_time = last_detection_time - start_time
                
                # Pausa dinamica per bilanciare CPU usage/FPS
                if detection_time < 0.01:  # Se il rilevamento è molto veloce
                    time.sleep(0.005)  # Breve pausa per ridurre uso CPU
                
            except Exception as e:
                print(f"Errore nel loop di rilevamento: {e}")
                time.sleep(0.1)  # Pausa per evitare loop di errori
    
    def _get_current_detections(self):
        """Callback per ottenere le rilevazioni attuali per il controller"""
        if not self.detection_active:
            return pd.DataFrame()  # DataFrame vuoto se il rilevamento è disattivato
            
        try:
            detections_df, _, _ = self.detector.detect_screen()
            return detections_df
        except Exception as e:
            print(f"Errore nell'ottenere le rilevazioni: {e}")
            return pd.DataFrame()
    
    def _update_ui(self):
        """Aggiorna l'interfaccia utente con le informazioni di rilevamento"""
        if not self.running:
            return
            
        try:
            # Cattura schermo e rileva oggetti
            detections_df, annotated_frame, fps = self.detector.detect_screen()
            
            # Aggiorna FPS visualizzato (arrotondato a intero)
            self.fps_var.set(f"FPS: {int(self.fps_counter.get_fps())}")
            
            # Aggiorna status
            aim_status = "Attiva" if self.assist_var.get() else "Inattiva"
            detect_status = "Attivo" if self.detection_active else "Inattivo"
            self.status_var.set(f"Assistenza: {aim_status} | Rilevamento: {detect_status}")
            
            # Converti frame per visualizzazione su canvas
            self._update_canvas(annotated_frame)
            
            # Riprogramma aggiornamento
            self.root.after(30, self._update_ui)  # ~30fps massimo per l'interfaccia
            
        except Exception as e:
            print(f"Errore nell'aggiornamento UI: {e}")
            # Riprogramma con maggiore intervallo in caso di errore
            self.root.after(100, self._update_ui)
    
    def _update_canvas(self, frame):
        """Aggiorna il canvas con l'immagine corrente"""
        # Ottieni dimensioni canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Verifica dimensioni minime
        if canvas_width < 10 or canvas_height < 10:
            return
            
        # Ridimensiona l'immagine per adattarla al canvas mantenendo le proporzioni
        img_height, img_width = frame.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        if scale < 1:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Converti per Tkinter (BGR -> RGB)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Centra l'immagine nel canvas
        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=imgtk, anchor=tk.CENTER)
        self.canvas.image = imgtk  # Mantieni un riferimento
    
    def _on_canvas_resize(self, event):
        """Gestisce il ridimensionamento del canvas"""
        # Ottiene le nuove dimensioni
        width, height = event.width, event.height
        if width > 10 and height > 10:
            # Ridefinisci le dimensioni del canvas
            self.canvas.config(width=width, height=height)
    
    def toggle_detection(self):
        """Attiva/disattiva il rilevamento"""
        self.detection_active = not self.detection_active
        
        if self.detection_active:
            self.start_stop_var.set("Ferma rilevamento")
        else:
            self.start_stop_var.set("Avvia rilevamento")
            # Ferma anche assistenza quando si ferma il rilevamento
            if self.assist_var.get():
                self.assist_var.set(False)
                self._toggle_aim_assist()
    
    def _toggle_aim_assist(self):
        """Attiva/disattiva l'assistenza al puntamento"""
        assist_active = self.assist_var.get()
        
        if assist_active:
            # Verifica che il rilevamento sia attivo
            if not self.detection_active:
                messagebox.showwarning("Attenzione", "Attivare prima il rilevamento")
                self.assist_var.set(False)
                return
                
            # Avvia il controllo automatico
            self.controller.set_detection_box_position(
                self.detector.detection_box['left'],
                self.detector.detection_box['top']
            )
            
            target_class = self.target_class_var.get() if self.target_class_var.get() != "Tutte le classi" else None
            
            self.controller.start_auto_control(
                detection_callback=self._get_current_detections,
                screen_center_x=self.detector.screen_center_x,
                screen_center_y=self.detector.screen_center_y,
                target_class=target_class
            )
        else:
            # Ferma il controllo automatico
            self.controller.stop_auto_control()
    
    def _toggle_trigger(self):
        """Attiva/disattiva il trigger automatico"""
        self.controller.update_settings(crosshair_trigger=self.trigger_var.get())
    
    def _update_confidence(self):
        """Aggiorna la soglia di confidenza"""
        self.detector.conf_threshold = self.conf_var.get()
    
    def _update_box_size(self):
        """Aggiorna la dimensione del box di rilevamento"""
        new_size = self.box_var.get()
        self.detector.box_size = new_size
        self.detector.update_detection_box()
    
    def _update_iou(self):
        """Aggiorna la soglia IOU per NMS"""
        self.detector.iou_threshold = self.iou_var.get()
    
    def _update_aim_height(self):
        """Aggiorna l'altezza di mira"""
        self.detector.aim_height_ratio = self.aim_height_var.get()
    
    def _update_sensitivity(self):
        """Aggiorna la sensibilità del mouse"""
        self.controller.update_settings(targeting_scale=self.sens_var.get())
    
    def _update_lock_threshold(self):
        """Aggiorna la soglia di lock-on"""
        self.controller.update_settings(lock_threshold=self.lock_var.get())
    
    def _update_performance(self):
        """Aggiorna le impostazioni di performance"""
        mode = self.performance_var.get()
        
        # Regola parametri in base alla modalità
        if mode == "precision":
            self.conf_var.set(0.65)
            self.iou_var.set(0.05)
        elif mode == "balanced":
            self.conf_var.set(0.45)
            self.iou_var.set(0.01)
        elif mode == "speed":
            self.conf_var.set(0.25)
            self.iou_var.set(0.01)
            
        # Aggiorna i parametri effettivi
        self._update_confidence()
        self._update_iou()
    
    def _update_class_list(self):
        """Aggiorna la lista delle classi rilevabili"""
        class_names = ["Tutte le classi"] + list(self.detector.class_names.values())
        self.target_combo['values'] = class_names
        if class_names:
            self.target_combo.current(0)
    
    def _load_model_dialog(self):
        """Apre un dialogo per caricare un nuovo modello ONNX"""
        filetypes = [("ONNX model", "*.onnx"), ("All files", "*.*")]
        model_path = filedialog.askopenfilename(
            title="Seleziona un modello ONNX",
            filetypes=filetypes
        )
        
        if not model_path:
            return  # L'utente ha annullato
            
        try:
            # Ferma rilevamento e assistenza
            was_active = self.detection_active
            if self.detection_active:
                self.toggle_detection()
            
            if self.assist_var.get():
                self.assist_var.set(False)
                self._toggle_aim_assist()
                
            # Imposta il nuovo modello
            device = self.device_option_var.get()
            self._reload_model_from_path(model_path, device)
            
            # Ripristina stato precedente
            if was_active:
                self.toggle_detection()
                
            # Aggiorna info modello
            self._update_model_info(model_path)
            
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile caricare il modello: {e}")
    
    def _device_selected(self):
        """Gestisce la selezione di un nuovo dispositivo"""
        # Aggiorna solo la variabile, il caricamento viene fatto con il pulsante
        pass
    
    def _reload_model(self):
        """Ricarica il modello sul dispositivo selezionato"""
        try:
            # Ferma rilevamento e assistenza
            was_active = self.detection_active
            if self.detection_active:
                self.toggle_detection()
            
            if self.assist_var.get():
                self.assist_var.set(False)
                self._toggle_aim_assist()
                
            # Ottieni percorso e dispositivo
            model_path = getattr(self.detector, "model_path", None)
            if not model_path:
                messagebox.showwarning("Attenzione", "Nessun modello caricato")
                return
                
            device = self.device_option_var.get()
            
            # Ricarica modello
            self._reload_model_from_path(model_path, device)
            
            # Ripristina stato precedente
            if was_active:
                self.toggle_detection()
                
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile ricaricare il modello: {e}")
    
    def _reload_model_from_path(self, model_path, device=None):
        """Ricarica un modello dal percorso specificato"""
        # Crea una nuova istanza del detector con lo stesso modello ma dispositivo diverso
        new_detector = ONNXObjectDetector(
            model_path=model_path,
            conf_threshold=self.detector.conf_threshold,
            iou_threshold=self.detector.iou_threshold,
            box_size=self.detector.box_size,
            aim_height_ratio=self.detector.aim_height_ratio,
            device=device
        )
        
        # Sostituisci il detector
        self.detector = new_detector
        
        # Aggiorna altre impostazioni
        self._update_class_list()
        self._update_model_info(model_path)
    
    def _update_model_info(self, model_path):
        """Aggiorna le informazioni sul modello caricato"""
        self.detector.model_path = model_path  # Mantieni riferimento
        self.model_path_var.set(f"Modello: {os.path.basename(model_path)}")
        self.device_var.set(f"Dispositivo: {self.detector.device}")
        self.classes_var.set(f"Classi: {len(self.detector.class_names)}")
    
    def quit(self):
        """Chiude l'applicazione"""
        self.running = False
        
        # Ferma il controllo automatico
        if self.controller:
            self.controller.stop_auto_control()
            
        # Aspetta che il thread di rilevamento termini
        if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
            
        self.root.destroy()
        print("Applicazione chiusa correttamente")