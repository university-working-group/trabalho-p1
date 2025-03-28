#!/bin/bash
set -e

# Verificar se o modelo facial existe
if [ ! -f "/app/models/nn4.small2.v1.t7" ]; then
    echo "AVISO: Modelo facial nn4.small2.v1.t7 não encontrado!"
    echo "Por favor, monte o modelo como volume usando:"
    echo "docker run -v /caminho/para/nn4.small2.v1.t7:/app/models/nn4.small2.v1.t7 ..."
    exit 1
fi

# Verificar modo de execução
if [ "$1" = "gui" ]; then
    # Verificar permissões de acesso à webcam
    if [ ! -r "/dev/video0" ]; then
        echo "AVISO: Sem acesso à webcam (/dev/video0)!"
        echo "Por favor, execute o contêiner com:"
        echo "docker run --device=/dev/video0:/dev/video0 ..."
    fi

    # Verificar configuração do display
    if [ -z "$DISPLAY" ]; then
        echo "AVISO: Variável DISPLAY não definida!"
        echo "Por favor, execute o contêiner com:"
        echo "docker run -e DISPLAY=\$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ..."
    fi

    # Executar a versão GUI
    echo "Iniciando sistema de reconhecimento facial (GUI)..."
    exec python facial_recognition_system.py

elif [ "$1" = "headless" ]; then
    # Obter parâmetros adicionais
    shift
    echo "Iniciando sistema de reconhecimento facial (Headless)..."
    exec python facial_recognition_headless.py "$@"

else
    echo "Uso: entrypoint.sh [gui|headless] [argumentos adicionais para modo headless]"
    echo ""
    echo "Exemplos:"
    echo "  entrypoint.sh gui                        # Iniciar interface gráfica"
    echo "  entrypoint.sh headless --mode register --user_id usuario1      # Registrar usuário"
    echo "  entrypoint.sh headless --mode verify --save_frames --output /app/output  # Verificar identidade"
    exit 1
fi