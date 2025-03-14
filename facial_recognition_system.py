import cv2
import numpy as np
import time
import os
from datetime import datetime
import json
from sklearn.metrics.pairwise import cosine_similarity


class FacialRecognitionSystem:
    STATE_INITIAL = 0
    STATE_CLOSE = 2
    STATE_FAR = 1
    STATE_TURN_LEFT = 3
    STATE_TURN_RIGHT = 4
    STATE_BLINK = 5  # mantido mas será implementado no futuro, por enqautno não haverá o passo de piscar.
    STATE_DONE = 6

    def __init__(self, config=None):
        self.log_file = f"logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.config = {
            "face_db_path": "face_database",
            "model_path": "models",
            "similarity_threshold": 0.7,
            "desired_width": 800,
            "focus_time": 2.5,
            "min_face_size": 150,
            "max_face_size": 200,
            "enable_liveness": True,
            "enable_logging": True,
            "feedback_time": 1.5,
            "validation_strictness": 0.7,
            "skip_blink": True  # Nova opção para pular o estágio de piscar
        }

        if config:
            self.config.update(config)

        os.makedirs(self.config["face_db_path"], exist_ok=True)
        os.makedirs(self.config["model_path"], exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        self._load_models()

        self.current_stage = self.STATE_INITIAL
        self.stage_start_time = 0
        self.embeddings = []
        self.captured_images = []
        self.captured_stages = []
        self.user_id = None
        self.log_file = f"logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        self.capture_success = False
        self.capture_failed = False
        self.validation_message = ""
        self.feedback_start_time = 0

    def _log_with_state(self, message):
        """Log mais detalhado que inclui o estado atual do sistema"""
        state_names = {
            self.STATE_INITIAL: "INITIAL",
            self.STATE_CLOSE: "CLOSE",
            self.STATE_FAR: "FAR",
            self.STATE_TURN_LEFT: "TURN_LEFT",
            self.STATE_TURN_RIGHT: "TURN_RIGHT",
            self.STATE_BLINK: "BLINK",
            self.STATE_DONE: "DONE"
        }

        state_name = state_names.get(self.current_stage, "UNKNOWN")
        self._log(f"[Estado: {state_name}] {message}")

    def _load_models(self):
        """Carrega todos os modelos necessários para reconhecimento facial"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        model_path = os.path.join(self.config["model_path"], "nn4.small2.v1.t7")
        if not os.path.exists(model_path):
            self._log("Modelo de embeddings não encontrado. Usando caminho padrão.")
            model_path = "models/nn4.small2.v1.t7"  # Tenta usar diretamente
        self.embedder = cv2.dnn.readNetFromTorch(model_path)

        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.landmark_detector = None
        try:
            import dlib
            self.landmark_detector = dlib.shape_predictor(os.path.join(
                self.config["model_path"], "shape_predictor_68_face_landmarks.dat"))
            self.face_detector = dlib.get_frontal_face_detector()
            self._log("Detector de marcos faciais do dlib carregado com sucesso.")
        except (ImportError, RuntimeError, FileNotFoundError):
            self._log("Detector de marcos faciais do dlib não disponível. Alguns recursos podem ser limitados.")

    def _log(self, message):
        """Registra mensagens em arquivo de log se habilitado"""
        if self.config["enable_logging"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] {message}"
            print(log_message)
            with open(self.log_file, "a") as f:
                f.write(log_message + "\n")

    def register_user(self, user_id):
        """Inicia o processo de registro de um novo usuário"""
        self.user_id = user_id
        self.mode = "register"
        self.current_stage = self.STATE_INITIAL
        self.embeddings = []
        self.captured_images = []
        self.captured_stages = []
        self._log(f"Iniciando registro do usuário: {user_id}")
        return self._start_capture()

    def verify_user(self, user_id=None):
        """Inicia o processo de verificação de um usuário"""
        self.user_id = user_id
        self.mode = "verify"
        self.current_stage = self.STATE_INITIAL
        self.embeddings = []
        self.captured_images = []
        self.captured_stages = []
        self._log(f"Iniciando verificação do usuário: {user_id if user_id else 'desconhecido'}")
        return self._start_capture()

    def _start_capture(self):
        """Inicia o processo de captura através da webcam"""
        cap = cv2.VideoCapture(0)
        self.stage_start_time = time.time()

        if not cap.isOpened():
            self._log("Erro: Não foi possível acessar a câmera.")
            return False

        while True:
            ret, frame = cap.read()
            if not ret:
                self._log("Erro: Falha ao ler frame da câmera.")
                break

            frame_processed = self._process_frame(frame)

            if self.current_stage == self.STATE_DONE:
                self._log("Captura concluída com sucesso.")
                break

            cv2.imshow('Sistema de Reconhecimento Facial', frame_processed)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._log("Captura interrompida pelo usuário.")
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.mode == "register" and len(self.embeddings) > 0:
            self._save_user_data()
            return True
        elif self.mode == "verify" and len(self.embeddings) > 0:
            return self._verify_user_identity()
        else:
            self._log("Processo finalizado sem dados suficientes.")
            return False

    def _validate_face_quality(self, face_image):
        """Valida a qualidade da imagem facial capturada

        Retorna:
            tuple: (is_valid, message) - Um booleano indicando se passou na validação e uma mensagem descritiva
        """
        if len(face_image.shape) == 3:
            gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = face_image.copy()

        height, width = gray_image.shape[:2]
        if width < 96 or height < 96:
            return False, "Rosto muito pequeno"

        brightness = np.mean(gray_image)
        if brightness < 40:
            return False, "Imagem muito escura"
        if brightness > 220:
            return False, "Imagem muito clara"

        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        if laplacian_var < 100:  # Valor arbitrário, ajuste conforme necessário
            return False, "Imagem fora de foco"

        if self.current_stage != self.STATE_BLINK:
            eyes = self.eye_cascade.detectMultiScale(gray_image, 1.1, 4)
            if len(eyes) < 2:
                return False, "Olhos não detectados corretamente"

        # Tentar detectar o rosto na imagem já recortada
        faces = self.face_cascade.detectMultiScale(gray_image, 1.1, 4)
        if len(faces) == 0:
            return False, "Rosto não detectado claramente"

        if self.current_stage == self.STATE_CLOSE:
            face_area_ratio = (width * height) / (face_image.shape[1] * face_image.shape[0])
            if face_area_ratio < 0.5:  # O rosto deve ocupar pelo menos 50% da imagem
                return False, "Rosto não está próximo o suficiente"

        elif self.current_stage == self.STATE_FAR:
            face_detail = cv2.cornerHarris(gray_image, 2, 3, 0.04)
            detail_points = np.sum(face_detail > 0.01 * face_detail.max())
            if detail_points < 100:  # Ajuste conforme necessário
                return False, "Detalhes faciais insuficientes"

        elif self.current_stage == self.STATE_TURN_LEFT or self.current_stage == self.STATE_TURN_RIGHT:
            # Aqui precisaríamos verificar a rotação do rosto
            # Simplificando, vamos verificar se o rosto não está totalmente frontal
            if self.landmark_detector:
                try:
                    import dlib
                    # Converter imagem para formato reconhecido pelo dlib
                    dlib_rect = dlib.rectangle(0, 0, width - 1, height - 1)
                    landmarks = self.landmark_detector(gray_image, dlib_rect)

                    # Analisar a posição horizontal dos olhos e do nariz para determinar rotação
                    left_eye = landmarks.part(36)  # Canto esquerdo do olho esquerdo
                    right_eye = landmarks.part(45)  # Canto direito do olho direito
                    nose_tip = landmarks.part(30)  # Ponta do nariz

                    # Calcular posição relativa do nariz em relação à linha dos olhos
                    eye_center_x = (left_eye.x + right_eye.x) / 2

                    # Direção da cabeça baseada na posição do nariz em relação ao centro dos olhos
                    nose_deviation = (nose_tip.x - eye_center_x) / (right_eye.x - left_eye.x)

                    if self.current_stage == self.STATE_TURN_LEFT:
                        # Para virar à esquerda, o nariz deve estar à esquerda do centro dos olhos
                        if nose_deviation > -0.15:  # Ajuste conforme necessário
                            return False, "Rosto não está virado suficientemente para a esquerda"

                    elif self.current_stage == self.STATE_TURN_RIGHT:
                        # Para virar à direita, o nariz deve estar à direita do centro dos olhos
                        if nose_deviation < 0.15:  # Ajuste conforme necessário
                            return False, "Rosto não está virado suficientemente para a direita"

                except (ImportError, RuntimeError, Exception) as e:
                    self._log(f"Erro ao validar rotação: {str(e)}")
                    # Falha silenciosa se o dlib não estiver disponível
                    pass

        # Passou em todas as verificações
        return True, "Imagem de boa qualidade"

    def _process_frame(self, frame):
        """Processa cada frame da câmera com feedback de validação"""
        h, w = frame.shape[:2]
        new_h = int((self.config["desired_width"] / w) * h)
        frame_resized = cv2.resize(frame, (self.config["desired_width"], new_h))
        frame_center = (self.config["desired_width"] // 2, new_h // 2)

        if hasattr(self, 'capture_success') and self.capture_success:
            feedback_elapsed = time.time() - self.feedback_start_time

            cv2.putText(frame_resized, "Captura bem-sucedida!",
                        (frame_center[0] - 150, frame_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if feedback_elapsed > self.config.get("feedback_time", 1.5):
                self._advance_to_next_stage()
                self.capture_success = False

            return frame_resized

        if hasattr(self, 'capture_failed') and self.capture_failed:
            feedback_elapsed = time.time() - self.feedback_start_time

            cv2.putText(frame_resized, f"Falha: {self.validation_message}",
                        (frame_center[0] - 200, frame_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame_resized, "Tente novamente",
                        (frame_center[0] - 100, frame_center[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if feedback_elapsed > self.config.get("feedback_time", 1.5):
                self.capture_failed = False
                self.stage_start_time = time.time()

            return frame_resized

        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        min_size, max_size = self._get_face_size_limits(self.current_stage)
        main_face = None
        progress = 0

        debug_text = f"Stage: {self.current_stage} - "
        debug_text += f"Min size: {min_size}, Max size: {max_size}"
        cv2.putText(frame_resized, debug_text, (20, new_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        closest_face = None
        min_distance = float('inf')

        for (x, y, fw, fh) in faces:
            face_center = (x + fw // 2, y + fh // 2)
            distance = np.linalg.norm(np.array(face_center) - np.array(frame_center))

            cv2.rectangle(frame_resized, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"w:{fw}, h:{fh}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if distance < min_distance:
                min_distance = distance
                closest_face = (x, y, fw, fh)

        if closest_face:
            x, y, fw, fh = closest_face

            if self.current_stage == self.STATE_CLOSE:
                # Para o estágio CLOSE, queremos que o rosto seja grande o suficiente
                if fw >= min_size and min_distance < 150:
                    main_face = closest_face
                    cv2.rectangle(frame_resized, (x, y), (x + fw, y + fh), (0, 0, 255), 3)  # Destacar face principal
            elif self.current_stage == self.STATE_FAR:
                # Para o estágio FAR, queremos que o rosto seja grande o suficiente também
                if fw >= min_size and min_distance < 150:
                    main_face = closest_face
                    cv2.rectangle(frame_resized, (x, y), (x + fw, y + fh), (0, 0, 255), 3)
            else:
                # Para outros estágios, apenas verificar a distância do centro
                if min_distance < 150:
                    main_face = closest_face
                    cv2.rectangle(frame_resized, (x, y), (x + fw, y + fh), (0, 0, 255), 3)

        if main_face:
            if self.current_stage == self.STATE_INITIAL:
                self.current_stage = self.STATE_CLOSE
                self.stage_start_time = time.time()
                self._log(f"Avançando para STATE_CLOSE")
            else:
                elapsed = time.time() - self.stage_start_time
                progress = min(elapsed / self.config["focus_time"], 1)

                cv2.putText(frame_resized, f"Progress: {progress:.2f}",
                            (20, new_h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)

                if elapsed > self.config["focus_time"]:
                    self._capture_face_data(frame_resized, main_face)
        else:
            progress = 0

        axes = self._get_dynamic_axes(self.current_stage, progress if main_face else 0)
        self._draw_dynamic_ellipse(frame_resized, frame_center, axes, progress if main_face else 0)

        self._add_help_text(frame_resized)

        return frame_resized

    def _get_face_size_limits(self, stage):
        """Define os tamanhos esperados do rosto para cada estágio"""
        if stage == self.STATE_FAR:
            return self.config["min_face_size"], 9999
        elif stage == self.STATE_CLOSE:
            return self.config["min_face_size"], 9999  # Sem limite superior para o rosto próximo
        else:
            return 0, 9999  # Estado inicial ou outros estados sem limites estritos

    def _capture_face_data(self, frame, face_coords):
        """Captura dados faciais para o estágio atual com validação de qualidade"""
        x, y, fw, fh = face_coords
        face_roi = frame[y:y + fh, x:x + fw]

        # Validar qualidade da imagem facial antes de aceitar
        is_valid, validation_message = self._validate_face_quality(face_roi)

        if not is_valid:
            # Falha na validação, definir estado de falha para mostrar feedback
            self.capture_failed = True
            self.validation_message = validation_message
            self.feedback_start_time = time.time()
            self._log(f"Validação falhou no estágio {self.current_stage}: {validation_message}")
            return False

        # Se chegou aqui, a validação passou
        self.captured_images.append(face_roi.copy())
        self.captured_stages.append(self.current_stage)

        # Calcular o embedding facial para o rosto capturado
        face_aligned = cv2.resize(face_roi, (96, 96))
        blob = cv2.dnn.blobFromImage(face_aligned, 1 / 255, (96, 96), (0, 0, 0), True)
        self.embedder.setInput(blob)
        vec = self.embedder.forward().flatten()
        self.embeddings.append(vec)

        self._log(f"Face capturada com sucesso no estágio {self.current_stage}")

        self.capture_success = True
        self.feedback_start_time = time.time()
        return True

    def _advance_to_next_stage(self):
        """Avança para o próximo estágio do processo de captura"""
        state_names = {
            self.STATE_INITIAL: "INITIAL",
            self.STATE_CLOSE: "CLOSE",
            self.STATE_FAR: "FAR",
            self.STATE_TURN_LEFT: "TURN_LEFT",
            self.STATE_TURN_RIGHT: "TURN_RIGHT",
            self.STATE_BLINK: "BLINK",
            self.STATE_DONE: "DONE"
        }

        old_stage = self.current_stage

        if self.config["enable_liveness"]:
            if self.config.get("skip_blink", True):
                stages_sequence = {
                    self.STATE_INITIAL: self.STATE_CLOSE,  # Aproxime-se
                    self.STATE_CLOSE: self.STATE_FAR,  # Aproxime-se mais
                    self.STATE_FAR: self.STATE_TURN_LEFT,  # Vire à esquerda
                    self.STATE_TURN_LEFT: self.STATE_TURN_RIGHT,  # Vire à direita
                    self.STATE_TURN_RIGHT: self.STATE_DONE  # Finalizar (pula o piscar)
                }
            else:
                stages_sequence = {
                    self.STATE_INITIAL: self.STATE_CLOSE,  # Aproxime-se
                    self.STATE_CLOSE: self.STATE_FAR,  # Aproxime-se mais
                    self.STATE_FAR: self.STATE_TURN_LEFT,  # Vire à esquerda
                    self.STATE_TURN_LEFT: self.STATE_TURN_RIGHT,  # Vire à direita
                    self.STATE_TURN_RIGHT: self.STATE_BLINK,  # Pisque os olhos
                    self.STATE_BLINK: self.STATE_DONE  # Finalizar
                }
        else:
            stages_sequence = {
                self.STATE_INITIAL: self.STATE_CLOSE,  # Aproxime-se
                self.STATE_CLOSE: self.STATE_FAR,  # Aproxime-se mais
                self.STATE_FAR: self.STATE_DONE  # Finalizar
            }

        if self.mode == "verify":
            if self.current_stage == self.STATE_CLOSE:
                self.current_stage = self.STATE_DONE
            else:
                self.current_stage = stages_sequence.get(self.current_stage, self.STATE_DONE)
        else:
            self.current_stage = stages_sequence.get(self.current_stage, self.STATE_DONE)

        self.stage_start_time = time.time()

        old_stage_name = state_names.get(old_stage, "UNKNOWN")
        new_stage_name = state_names.get(self.current_stage, "UNKNOWN")
        self._log(f"Mudando estágio: {old_stage_name} -> {new_stage_name}")

    def _get_dynamic_axes(self, stage, progress):
        """Retorna os eixos da elipse com base no estágio e progresso"""
        if stage == self.STATE_INITIAL:
            base_axes = (80, 100)
            target_axes = (100, 120)
        elif stage == self.STATE_CLOSE:
            base_axes = (100, 120)
            target_axes = (150, 180)
        elif stage == self.STATE_FAR:
            base_axes = (150, 180)
            target_axes = (100, 120)
        elif stage in [self.STATE_TURN_LEFT, self.STATE_TURN_RIGHT, self.STATE_BLINK]:
            base_axes = (100, 120)
            target_axes = (110, 130)
        else:
            base_axes = (80, 100)
            target_axes = (80, 100)

        # Interpolação linear
        axes_x = int(base_axes[0] + (target_axes[0] - base_axes[0]) * progress)
        axes_y = int(base_axes[1] + (target_axes[1] - base_axes[1]) * progress)
        return (axes_x, axes_y)

    def _draw_dynamic_ellipse(self, frame, center, axes, progress):
        """Desenha a elipse de enquadramento facial com feedback visual"""
        colors = {
            self.STATE_INITIAL: (0, 0, 255),  # Vermelho: Aguardando
            self.STATE_CLOSE: (0, 255, 0),  # Verde: Aproxime-se
            self.STATE_FAR: (255, 165, 0),  # Laranja: Aproxime-se mais
            self.STATE_TURN_LEFT: (255, 0, 255),  # Magenta: Vire à esquerda
            self.STATE_TURN_RIGHT: (255, 255, 0),  # Ciano: Vire à direita
            self.STATE_BLINK: (128, 0, 128),  # Roxo: Pisque os olhos
            self.STATE_DONE: (0, 255, 255)  # Amarelo: Concluído
        }

        color = colors.get(self.current_stage, (255, 255, 255))
        cv2.ellipse(frame, center, axes, 0, 0, 360, color, 2)

        inner_axes = (int(axes[0] * 0.9), int(axes[1] * 0.9))
        cv2.ellipse(frame, center, inner_axes, 0, 0, int(360 * progress), color, 3)

    def _add_help_text(self, frame):
        """Adiciona instruções na interface com melhorias visuais"""
        help_texts = {
            self.STATE_INITIAL: "Posicione seu rosto no oval.",
            self.STATE_CLOSE: "Aproxime-se do oval verde.",
            self.STATE_FAR: "Aproxime-se ainda mais para capturar detalhes.",
            self.STATE_TURN_LEFT: "Vire levemente para a esquerda.",
            self.STATE_TURN_RIGHT: "Vire levemente para a direita.",
            self.STATE_BLINK: "Pisque os olhos naturalmente (feche e abra os olhos).",
            self.STATE_DONE: "Captura concluída!"
        }

        cv2.putText(frame, help_texts.get(self.current_stage, ""),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        mode_text = "MODO: CADASTRO" if self.mode == "register" else "MODO: VERIFICAÇÃO"
        cv2.putText(frame, mode_text,
                    (self.config["desired_width"] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.mode == "register":
            total_stages = 5 if self.config["enable_liveness"] else 3  # Reduzido para 5 (sem o blink)
            current_progress = min((self.current_stage / total_stages) * 100, 100)
            cv2.putText(frame, f"Progresso: {int(current_progress)}%",
                        (20, self.config["desired_width"] // 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.current_stage == self.STATE_TURN_LEFT:
            # esquerda
            arrow_x = self.config["desired_width"] - 100
            arrow_y = 100
            cv2.arrowedLine(frame, (arrow_x + 30, arrow_y), (arrow_x - 30, arrow_y), (255,255, 255), 2, tipLength=0.3)

        elif self.current_stage == self.STATE_TURN_RIGHT:
            # Seta direita
            arrow_x = self.config["desired_width"] - 100
            arrow_y = 100
            cv2.arrowedLine(frame, (arrow_x - 30, arrow_y), (arrow_x + 30, arrow_y), (255, 255, 255), 2, tipLength=0.3)

    def _save_user_data(self):
        """Salva os dados do usuário após o registro"""
        if not self.user_id:
            self.user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        user_dir = os.path.join(self.config["face_db_path"], self.user_id)
        os.makedirs(user_dir, exist_ok=True)

        mean_embedding = np.mean(self.embeddings, axis=0)
        np.save(os.path.join(user_dir, "embedding.npy"), mean_embedding)

        for i, (img, stage) in enumerate(zip(self.captured_images, self.captured_stages)):
            cv2.imwrite(os.path.join(user_dir, f"face_{i}_stage_{stage}.jpg"), img)

        metadata = {
            "user_id": self.user_id,
            "registration_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_images": len(self.captured_images),
            "stages_captured": self.captured_stages,
            "liveness_verified": self.config["enable_liveness"]
        }

        with open(os.path.join(user_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        self._log(f"Dados do usuário {self.user_id} salvos com sucesso.")
        return True

    def _verify_user_identity(self):
        """Verifica a identidade do usuário"""
        if not self.user_id:
            self._log("Tentando identificar usuário...")
            return self._identify_user()

        user_dir = os.path.join(self.config["face_db_path"], self.user_id)
        if not os.path.exists(user_dir):
            self._log(f"Usuário {self.user_id} não encontrado no banco de dados.")
            return False

        try:
            stored_embedding = np.load(os.path.join(user_dir, "embedding.npy"))
        except FileNotFoundError:
            self._log(f"Dados biométricos do usuário {self.user_id} não encontrados.")
            return False

        current_embedding = np.mean(self.embeddings, axis=0)

        similarity = cosine_similarity([current_embedding], [stored_embedding])[0][0]

        is_match = similarity >= self.config["similarity_threshold"]

        # Registrar resultado
        self._log(f"Verificação do usuário {self.user_id}: " +
                  f"{'SUCESSO' if is_match else 'FALHA'} (similaridade: {similarity:.4f})")

        return {
            "success": is_match,
            "user_id": self.user_id,
            "similarity": similarity,
            "threshold": self.config["similarity_threshold"]
        }

    def _identify_user(self):
        """Identifica o usuário entre todos os cadastrados"""
        users = [d for d in os.listdir(self.config["face_db_path"])
                 if os.path.isdir(os.path.join(self.config["face_db_path"], d))]

        if not users:
            self._log("Nenhum usuário cadastrado no sistema.")
            return False

        current_embedding = np.mean(self.embeddings, axis=0)

        best_match = None
        best_similarity = -1

        for user in users:
            user_dir = os.path.join(self.config["face_db_path"], user)
            try:
                stored_embedding = np.load(os.path.join(user_dir, "embedding.npy"))
                similarity = cosine_similarity([current_embedding], [stored_embedding])[0][0]

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = user
            except Exception as e:
                self._log(f"Erro ao processar usuário {user}: {str(e)}")
                continue

        is_match = best_similarity >= self.config["similarity_threshold"]

        if is_match:
            self._log(f"Usuário identificado: {best_match} (similaridade: {best_similarity:.4f})")
        else:
            self._log(f"Nenhum usuário identificado. Melhor similaridade: {best_similarity:.4f} com {best_match}")

        return {
            "success": is_match,
            "user_id": best_match if is_match else None,
            "similarity": best_similarity,
            "threshold": self.config["similarity_threshold"]
        }


if __name__ == "__main__":
    config = {
        "face_db_path": "face_database",
        "similarity_threshold": 0.65,
        "enable_liveness": True,
        "focus_time": 2.5,
        "min_face_size": 150,
        "max_face_size": 200,
        "desired_width": 800,
        "enable_logging": True,
        "feedback_time": 1.5,
        "validation_strictness": 0.7,
        "skip_blink": True  # Configuração para pular estágio de piscar, pra implementar no futuro
    }

    face_system = FacialRecognitionSystem(config)

    while True:
        print("\n===== SISTEMA DE RECONHECIMENTO FACIAL =====")
        print("1. Cadastrar novo usuário")
        print("2. Verificar usuário específico")
        print("3. Identificar usuário")
        print("4. Sair")

        option = input("\nEscolha uma opção: ")

        if option == "1":
            user_id = input("Digite o ID do novo usuário: ")
            result = face_system.register_user(user_id)
            if result:
                print(f"Usuário {user_id} cadastrado com sucesso!")
            else:
                print("Falha no cadastro do usuário.")

        elif option == "2":
            user_id = input("Digite o ID do usuário a verificar: ")
            result = face_system.verify_user(user_id)
            if result and result["success"]:
                print(f"Verificação bem-sucedida! Similaridade: {result['similarity']:.4f}")
            else:
                print("Verificação falhou. Usuário não reconhecido.")

        elif option == "3":
            result = face_system.verify_user()
            if result and result["success"]:
                print(f"Usuário identificado: {result['user_id']}")
                print(f"Similaridade: {result['similarity']:.4f}")
            else:
                print("Nenhum usuário reconhecido.")

        elif option == "4":
            print("Encerrando o sistema...")
            break

        else:
            print("Opção inválida. Tente novamente.")
