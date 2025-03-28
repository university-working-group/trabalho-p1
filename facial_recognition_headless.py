import cv2
import numpy as np
import time
import os
from datetime import datetime
import json
from sklearn.metrics.pairwise import cosine_similarity
import argparse


class FacialRecognitionHeadless:
    """Versão headless do sistema de reconhecimento facial para uso em servidores ou ambientes sem interface gráfica"""

    STATE_INITIAL = 0
    STATE_CLOSE = 2
    STATE_FAR = 1
    STATE_TURN_LEFT = 3
    STATE_TURN_RIGHT = 4
    STATE_BLINK = 5
    STATE_DONE = 6

    def __init__(self, config=None, output_dir="output"):
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
            "skip_blink": True,
            "save_frames": True,
            "output_dir": output_dir
        }

        if config:
            self.config.update(config)

        os.makedirs(self.config["face_db_path"], exist_ok=True)
        os.makedirs(self.config["model_path"], exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs(self.config["output_dir"], exist_ok=True)

        self._load_models()

        self.current_stage = self.STATE_INITIAL
        self.stage_start_time = 0
        self.embeddings = []
        self.captured_images = []
        self.captured_stages = []
        self.user_id = None

        self.capture_success = False
        self.capture_failed = False
        self.validation_message = ""
        self.feedback_start_time = 0

        self.frame_count = 0

    def _log(self, message):
        """Registra mensagens em arquivo de log se habilitado"""
        if self.config["enable_logging"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] {message}"
            print(log_message)
            with open(self.log_file, "a") as f:
                f.write(log_message + "\n")

    def _load_models(self):
        """Carrega todos os modelos necessários para reconhecimento facial"""
        self._log("Carregando modelos...")
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self._log("Modelo haarcascade carregado com sucesso")
        except Exception as e:
            self._log(f"Erro ao carregar haarcascade: {str(e)}")
            raise e

        try:
            model_path = os.path.join(self.config["model_path"], "nn4.small2.v1.t7")
            if not os.path.exists(model_path):
                self._log("Modelo de embeddings não encontrado. Usando caminho padrão.")
                model_path = "models/nn4.small2.v1.t7"  # Tenta usar diretamente
            self.embedder = cv2.dnn.readNetFromTorch(model_path)
            self._log(f"Modelo de embeddings carregado com sucesso de {model_path}")
        except Exception as e:
            self._log(f"Erro ao carregar modelo de embeddings: {str(e)}")
            raise e

        try:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self._log("Modelo haarcascade_eye carregado com sucesso")
        except Exception as e:
            self._log(f"Erro ao carregar haarcascade_eye: {str(e)}")

        self.landmark_detector = None
        try:
            import dlib
            self._log("Biblioteca dlib importada com sucesso")

            landmark_path = os.path.join(self.config["model_path"], "shape_predictor_68_face_landmarks.dat")
            if os.path.exists(landmark_path):
                self.landmark_detector = dlib.shape_predictor(landmark_path)
                self.face_detector = dlib.get_frontal_face_detector()
                self._log("Detector de marcos faciais do dlib carregado com sucesso.")
            else:
                self._log("Arquivo de marcos faciais não encontrado em: " + landmark_path)
        except Exception as e:
            self._log(f"Detector de marcos faciais do dlib não disponível: {str(e)}")

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
        self._log("Iniciando captura da webcam...")
        cap = cv2.VideoCapture(0)
        self.stage_start_time = time.time()
        session_dir = os.path.join(self.config["output_dir"],
                                   f"{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        if self.config["save_frames"]:
            os.makedirs(session_dir, exist_ok=True)

        if not cap.isOpened():
            self._log("Erro: Não foi possível acessar a câmera.")
            return False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self._log("Erro: Falha ao ler frame da câmera.")
                    break

                frame_processed, result = self._process_frame(frame)

                if self.config["save_frames"]:
                    output_path = os.path.join(session_dir, f"frame_{self.frame_count:04d}.jpg")
                    cv2.imwrite(output_path, frame_processed)
                    self.frame_count += 1

                if result and result.get("stage_complete"):
                    self._log(f"Estágio {self.current_stage} concluído.")

                if self.current_stage == self.STATE_DONE:
                    self._log("Captura concluída com sucesso.")
                    break

                time.sleep(0.1)

                if time.time() - self.stage_start_time > 60:  # timeout de 60 segundos
                    self._log("Tempo limite de captura excedido.")
                    break
        except KeyboardInterrupt:
            self._log("Captura interrompida pelo usuário.")
        except Exception as e:
            self._log(f"Erro durante a captura: {str(e)}")
        finally:
            cap.release()
            self._log("Captura finalizada e recursos liberados.")

        if self.mode == "register" and len(self.embeddings) > 0:
            return self._save_user_data()
        elif self.mode == "verify" and len(self.embeddings) > 0:
            return self._verify_user_identity()
        else:
            self._log("Processo finalizado sem dados suficientes.")
            return False

    def _process_frame(self, frame):
        """Processa cada frame da câmera com feedback de validação

        Retorna:
            tuple: (processed_frame, result_dict)
        """
        result = {"stage": self.current_stage}

        h, w = frame.shape[:2]
        new_h = int((self.config["desired_width"] / w) * h)
        frame_resized = cv2.resize(frame, (self.config["desired_width"], new_h))

        self._add_info_text(frame_resized)

        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        closest_face = self._find_closest_face(faces, frame_resized)

        if closest_face:
            x, y, fw, fh = closest_face
            cv2.rectangle(frame_resized, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            cv2.putText(frame_resized, f"Face: {fw}x{fh}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if self.current_stage == self.STATE_INITIAL:
                self.current_stage = self.STATE_CLOSE
                self.stage_start_time = time.time()
                self._log("Avançando para STATE_CLOSE")
                result["stage_changed"] = True
            else:
                elapsed = time.time() - self.stage_start_time
                progress = min(elapsed / self.config["focus_time"], 1)

                cv2.putText(frame_resized, f"Progress: {progress:.2f}",
                            (20, new_h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)

                if elapsed > self.config["focus_time"]:
                    face_roi = frame_resized[y:y + fh, x:x + fw]
                    validation_result = self._validate_face(face_roi)

                    if validation_result["is_valid"]:
                        self.captured_images.append(face_roi.copy())
                        self.captured_stages.append(self.current_stage)

                        face_aligned = cv2.resize(face_roi, (96, 96))
                        blob = cv2.dnn.blobFromImage(face_aligned, 1 / 255, (96, 96), (0, 0, 0), True)
                        self.embedder.setInput(blob)
                        vec = self.embedder.forward().flatten()
                        self.embeddings.append(vec)

                        result["stage_complete"] = True
                        self._advance_to_next_stage()
                    else:
                        cv2.putText(frame_resized, f"Falha: {validation_result['message']}",
                                    (20, new_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 2)
                        self.stage_start_time = time.time()

        return frame_resized, result

    def _find_closest_face(self, faces, frame):
        """Encontra a face mais próxima do centro do frame"""
        if len(faces) == 0:
            return None

        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        closest_face = None
        min_distance = float('inf')

        for (x, y, fw, fh) in faces:
            face_center = (x + fw // 2, y + fh // 2)
            distance = np.linalg.norm(np.array(face_center) - np.array(frame_center))

            if distance < min_distance:
                min_distance = distance
                closest_face = (x, y, fw, fh)

        return closest_face

    def _validate_face(self, face_image):
        """Valida a qualidade da imagem facial capturada"""
        if len(face_image.shape) == 3:
            gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = face_image.copy()

        height, width = gray_image.shape[:2]

        if width < 96 or height < 96:
            return {"is_valid": False, "message": "Rosto muito pequeno"}

        brightness = np.mean(gray_image)
        if brightness < 40:
            return {"is_valid": False, "message": "Imagem muito escura"}
        if brightness > 220:
            return {"is_valid": False, "message": "Imagem muito clara"}

        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        if laplacian_var < 100:
            return {"is_valid": False, "message": "Imagem fora de foco"}

        if self.current_stage == self.STATE_CLOSE:
            faces = self.face_cascade.detectMultiScale(gray_image, 1.1, 4)
            if len(faces) == 0:
                return {"is_valid": False, "message": "Rosto não detectado claramente"}

            x, y, w, h = faces[0]
            face_area_ratio = (w * h) / (width * height)
            if face_area_ratio < 0.5:
                return {"is_valid": False, "message": "Rosto não está próximo o suficiente"}

        elif self.current_stage in [self.STATE_TURN_LEFT, self.STATE_TURN_RIGHT]:
            if self.landmark_detector:
                try:
                    import dlib
                    dlib_rect = dlib.rectangle(0, 0, width - 1, height - 1)
                    landmarks = self.landmark_detector(gray_image, dlib_rect)

                    # Analisar posição dos marcos faciais
                    left_eye = landmarks.part(36)  # Canto esquerdo do olho esquerdo
                    right_eye = landmarks.part(45)  # Canto direito do olho direito
                    nose_tip = landmarks.part(30)  # Ponta do nariz

                    eye_center_x = (left_eye.x + right_eye.x) / 2
                    nose_deviation = (nose_tip.x - eye_center_x) / (right_eye.x - left_eye.x)

                    if self.current_stage == self.STATE_TURN_LEFT and nose_deviation > -0.15:
                        return {"is_valid": False, "message": "Rosto não está virado para a esquerda"}
                    elif self.current_stage == self.STATE_TURN_RIGHT and nose_deviation < 0.15:
                        return {"is_valid": False, "message": "Rosto não está virado para a direita"}

                except Exception as e:
                    self._log(f"Erro ao analisar rotação: {str(e)}")

        return {"is_valid": True, "message": "Imagem válida"}

    def _advance_to_next_stage(self):
        """Avança para o próximo estágio do processo de captura"""
        old_stage = self.current_stage

        if self.config["enable_liveness"]:
            if self.config.get("skip_blink", True):
                stages_sequence = {
                    self.STATE_INITIAL: self.STATE_CLOSE,
                    self.STATE_CLOSE: self.STATE_FAR,
                    self.STATE_FAR: self.STATE_TURN_LEFT,
                    self.STATE_TURN_LEFT: self.STATE_TURN_RIGHT,
                    self.STATE_TURN_RIGHT: self.STATE_DONE
                }
            else:
                stages_sequence = {
                    self.STATE_INITIAL: self.STATE_CLOSE,
                    self.STATE_CLOSE: self.STATE_FAR,
                    self.STATE_FAR: self.STATE_TURN_LEFT,
                    self.STATE_TURN_LEFT: self.STATE_TURN_RIGHT,
                    self.STATE_TURN_RIGHT: self.STATE_BLINK,
                    self.STATE_BLINK: self.STATE_DONE
                }
        else:
            stages_sequence = {
                self.STATE_INITIAL: self.STATE_CLOSE,
                self.STATE_CLOSE: self.STATE_FAR,
                self.STATE_FAR: self.STATE_DONE
            }

        if self.mode == "verify":
            if self.current_stage == self.STATE_CLOSE:
                self.current_stage = self.STATE_DONE
            else:
                self.current_stage = stages_sequence.get(self.current_stage, self.STATE_DONE)
        else:
            self.current_stage = stages_sequence.get(self.current_stage, self.STATE_DONE)

        self.stage_start_time = time.time()
        self._log(f"Mudando estágio: {old_stage} -> {self.current_stage}")

    def _add_info_text(self, frame):
        """Adiciona informações de status no frame"""
        h, w = frame.shape[:2]

        stage_texts = {
            self.STATE_INITIAL: "Posicione seu rosto",
            self.STATE_CLOSE: "Aproxime-se da câmera",
            self.STATE_FAR: "Mantenha o rosto visível",
            self.STATE_TURN_LEFT: "Vire levemente para a esquerda",
            self.STATE_TURN_RIGHT: "Vire levemente para a direita",
            self.STATE_BLINK: "Pisque naturalmente",
            self.STATE_DONE: "Processo concluído"
        }

        stage_text = stage_texts.get(self.current_stage, "")
        cv2.putText(frame, stage_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        mode_text = "REGISTRO" if self.mode == "register" else "VERIFICAÇÃO"
        cv2.putText(frame, f"MODO: {mode_text}", (w - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.mode == "register":
            total_stages = 5 if self.config["enable_liveness"] else 3
            if not self.config.get("skip_blink", True):
                total_stages += 1

            current_progress = min((self.current_stage / total_stages) * 100, 100)
            cv2.putText(frame, f"Progresso: {int(current_progress)}%",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

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
    parser = argparse.ArgumentParser(description='Sistema de Reconhecimento Facial (Versão Headless)')

    parser.add_argument('--mode', type=str, choices=['register', 'verify'], required=True,
                        help='Modo de operação: registro ou verificação')

    parser.add_argument('--user_id', type=str, default=None,
                        help='ID do usuário para registro/verificação')

    parser.add_argument('--output', type=str, default='output',
                        help='Diretório para salvar frames processados')

    parser.add_argument('--liveness', action='store_true',
                        help='Ativar verificação de vivacidade')

    parser.add_argument('--threshold', type=float, default=0.65,
                        help='Limite de similaridade para verificação')

    parser.add_argument('--save_frames', action='store_true',
                        help='Salvar frames processados')

    args = parser.parse_args()

    config = {
        "face_db_path": "face_database",
        "similarity_threshold": args.threshold,
        "enable_liveness": args.liveness,
        "save_frames": args.save_frames,
        "output_dir": args.output
    }

    face_system = FacialRecognitionHeadless(config)

    if args.mode == 'register':
        user_id = args.user_id or f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = face_system.register_user(user_id)
        if result:
            print(f"Usuário {user_id} registrado com sucesso!")
        else:
            print("Falha no registro do usuário.")

    elif args.mode == 'verify':
        result = face_system.verify_user(args.user_id)
        if result and result["success"]:
            print(f"Verificação bem-sucedida!")
            print(f"Usuário: {result['user_id']}")
            print(f"Similaridade: {result['similarity']:.4f}")
        else:
            print("Verificação falhou. Usuário não reconhecido.")
