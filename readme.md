# Sistema de Reconhecimento Facial

Um sistema completo de reconhecimento facial com verificação de vivacidade (liveness detection) para cadastro, verificação e identificação de usuários.

## Índice

1. [Visão Geral](#visão-geral)
2. [Requisitos do Sistema](#requisitos-do-sistema)
3. [Instalação](#instalação)
4. [Como Usar](#como-usar)
5. [Arquitetura do Sistema](#arquitetura-do-sistema)
6. [Fluxo de Captura](#fluxo-de-captura)
7. [Algoritmos e Técnicas Utilizadas](#algoritmos-e-técnicas-utilizadas)
8. [Estrutura do Código](#estrutura-do-código)
9. [Contribuições](#contribuições)
10. [Solução de Problemas](#solução-de-problemas)

## Visão Geral

Este sistema utiliza visão computacional para realizar o reconhecimento facial com verificação de vivacidade (liveness detection), garantindo que apenas pessoas reais sejam cadastradas e reconhecidas. O sistema implementa uma abordagem em etapas para capturar diferentes ângulos do rosto, validando a qualidade de cada imagem e criando um perfil biométrico robusto.

### Principais Funcionalidades:

- **Cadastro de Usuários**: Captura múltiplas imagens faciais em diferentes ângulos
- **Verificação de Identidade**: Compara o rosto atual com um perfil específico cadastrado
- **Identificação de Usuários**: Busca o perfil mais semelhante em toda a base de dados
- **Detecção de Vivacidade**: Garante que é uma pessoa real através de movimentos faciais
- **Validação de Qualidade**: Verifica iluminação, foco, centralização e outros parâmetros

## Requisitos do Sistema

### Hardware:
- Webcam: Qualquer webcam integrada ou USB (resolução mínima recomendada: 720p)
- Processador: Intel Core i3/AMD Ryzen 3 ou superior
- Memória: Mínimo 4GB de RAM
- Armazenamento: 100MB para a aplicação + espaço para armazenamento de perfis

### Software:
- Python 3.7 ou superior
- Bibliotecas Python (detalhadas na seção de instalação)
- Sistema Operacional: Windows 10/11, macOS 10.14+, Ubuntu 18.04+ ou distribuição Linux equivalente

### Recomendações:
- Iluminação adequada no ambiente: evitar luz direta no rosto ou ambientes muito escuros
- Fundo neutro: preferencialmente uma parede lisa de cor clara
- Distância apropriada da câmera: 50-80cm

## Instalação

### 1. Instalar Python e Pip
Caso ainda não tenha o Python instalado, baixe em [python.org](https://www.python.org/downloads/).

### 2. Clone o repositório ou baixe os arquivos
```bash
git clone https://github.com/seu-usuario/sistema-reconhecimento-facial.git
cd sistema-reconhecimento-facial
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

Ou instalar manualmente:
```bash
pip install opencv-python numpy scikit-learn dlib
```

### 4. Baixar modelos pré-treinados
Crie uma pasta `models` e baixe os seguintes arquivos:
- [Modelo de Embeddings Faciais (nn4.small2.v1.t7)](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel)

Coloque os modelos baixados na pasta `models`.

### 5. Preparar diretórios
```bash
mkdir -p face_database logs
```

## Como Usar

### Executar o programa principal
```bash
python facial_recognition_system.py
```

### Menu de Opções
Ao iniciar o sistema, você verá o seguinte menu:
```
===== SISTEMA DE RECONHECIMENTO FACIAL =====
1. Cadastrar novo usuário
2. Verificar usuário específico
3. Identificar usuário
4. Sair
```

### 1. Cadastrar Novo Usuário
- Selecione a opção 1
- Digite um ID único para o usuário (ex: "joao_silva")
- Siga as instruções na tela, posicionando seu rosto de acordo com as instruções
- O processo capturará múltiplas imagens em diferentes ângulos

### 2. Verificar Usuário Específico
- Selecione a opção 2
- Digite o ID do usuário a verificar
- O sistema capturará seu rosto e comparará com o registro armazenado desse usuário
- O resultado da verificação será exibido junto com a pontuação de similaridade

### 3. Identificar Usuário
- Selecione a opção 3
- O sistema capturará seu rosto e comparará com todos os usuários cadastrados
- Se encontrar uma correspondência, exibirá o ID do usuário e a similaridade

### 4. Sair
- Encerra o programa

## Arquitetura do Sistema

### Componentes Principais

#### 1. Módulo de Detecção Facial
Utiliza Haar Cascades do OpenCV para detectar e localizar rostos na imagem da webcam.

#### 2. Módulo de Extração de Características (Embeddings)
Utiliza um modelo de rede neural pré-treinado para extrair um vetor de características de 128 dimensões que representa unicamente cada rosto.

#### 3. Módulo de Detecção de Landmarks
Utiliza dlib para detectar 68 pontos-chave no rosto, permitindo análises de rotação facial e outras verificações.

#### 4. Módulo de Validação de Qualidade
Analisa diversos parâmetros como iluminação, nitidez, enquadramento e outros para garantir imagens faciais de boa qualidade.

#### 5. Módulo de Verificação de Vivacidade
Implementa uma sequência de etapas para garantir que está interagindo com uma pessoa real (aproximar, rotação da cabeça).

#### 6. Módulo de Reconhecimento
Compara embeddings faciais para determinar a identidade da pessoa, utilizando similaridade de cosseno.

#### 7. Sistema de Armazenamento
Gerencia o armazenamento de perfis, embeddings e metadados no sistema de arquivos.

## Fluxo de Captura

O sistema implementa um fluxo de captura em etapas para garantir imagens faciais de qualidade e verificação de vivacidade:

1. **INITIAL**: Posicionamento inicial do rosto
2. **CLOSE**: Aproximação do rosto para captura detalhada
3. **FAR**: Segunda aproximação para capturar mais detalhes em diferente distância
4. **TURN_LEFT**: Rotação do rosto para a esquerda
5. **TURN_RIGHT**: Rotação do rosto para a direita
6. **DONE**: Finalização do processo

Para cada etapa:
- O sistema valida a qualidade da imagem (foco, iluminação, enquadramento)
- Fornece feedback visual ao usuário
- Salva as imagens e embeddings correspondentes

## Algoritmos e Técnicas Utilizadas

### Detecção Facial
- **Haar Cascades**: Algoritmo rápido e eficiente para detecção facial em tempo real
- **Redimensionamento de Imagem**: Padronização para melhorar a performance e consistência

### Extração de Características Faciais
- **Modelo de Rede Neural (nn4.small2.v1)**: Modelo pré-treinado que extrai um vetor de características (embedding) de 128 dimensões
- **Alinhamento Facial**: Padronização de pose para melhorar a precisão do reconhecimento

### Detecção de Landmarks Faciais
- **dlib Shape Predictor**: Localiza 68 pontos específicos no rosto para análise detalhada
- **Análise de Posição**: Calcula desvios e proporções para verificar rotação da cabeça

### Verificação de Qualidade da Imagem
- **Laplaciano da Gaussiana**: Mede o nível de foco/nitidez da imagem
- **Análise de Histograma**: Avalia a iluminação e contraste
- **Proporção de Área**: Verifica se o rosto ocupa uma proporção adequada do quadro

### Reconhecimento Facial
- **Similaridade de Cosseno**: Mede a semelhança entre os vetores de embeddings
- **Limiar Adaptativo**: Define o nível de confiança para aceitação de uma correspondência

### Interface Visual
- **Feedback Dinâmico**: Elipses e indicadores visuais que guiam o usuário durante o processo
- **Visualização de Estados**: Codificação por cores e mensagens para indicar o estado atual e progresso

## Estrutura do Código

### Classe `FacialRecognitionSystem`
Classe principal que implementa todo o sistema.

### Atributos Principais
- `config`: Dicionário com todas as configurações e parâmetros do sistema
- `embeddings`: Lista de vetores de características extraídos durante a captura
- `captured_images`: Lista de imagens faciais capturadas
- `captured_stages`: Lista de estágios correspondentes a cada imagem capturada

### Métodos Principais

#### `__init__(self, config=None)`
Inicializa o sistema com configurações padrão ou personalizadas, carrega modelos e prepara diretórios.

#### `_load_models(self)`
Carrega os modelos necessários para detecção facial, extração de embeddings e landmarks faciais.

#### `register_user(self, user_id)`
Inicia o processo de registro de um novo usuário, capturando múltiplas imagens em diferentes posições.

#### `verify_user(self, user_id=None)`
Verifica a identidade de um usuário específico ou identifica entre todos os cadastrados.

#### `_start_capture(self)`
Inicia a captura de vídeo e o processamento de frames da webcam.

#### `_process_frame(self, frame)`
Processa cada frame da webcam, detectando rostos, validando qualidade e gerenciando o fluxo de captura.

#### `_validate_face_quality(self, face_image)`
Valida diversos parâmetros de qualidade da imagem facial (foco, iluminação, etc.).

#### `_capture_face_data(self, frame, face_coords)`
Captura dados faciais (imagem e embedding) após validação de qualidade.

#### `_advance_to_next_stage(self)`
Gerencia a transição entre os estágios do fluxo de captura.

#### `_save_user_data(self)`
Salva os dados do usuário (embeddings, imagens e metadados) no sistema de arquivos.

#### `_verify_user_identity(self)`
Verifica a identidade de um usuário comparando embeddings.

#### `_identify_user(self)`
Busca entre todos os usuários cadastrados o perfil mais semelhante.

### Constantes de Estado
- `STATE_INITIAL = 0`: Estado inicial de posicionamento do rosto
- `STATE_CLOSE = 2`: Estágio de aproximação próxima
- `STATE_FAR = 1`: Estágio de aproximação adicional (mais detalhes)
- `STATE_TURN_LEFT = 3`: Estágio de rotação do rosto para esquerda
- `STATE_TURN_RIGHT = 4`: Estágio de rotação do rosto para direita
- `STATE_BLINK = 5`: Estágio de piscar (opcional, configurável)
- `STATE_DONE = 6`: Estágio final, captura concluída

## Contribuições

Contribuições são bem-vindas! Algumas áreas potenciais para melhoria:

- Implementação de uma interface gráfica mais completa
- Melhorias no algoritmo de detecção de vivacidade
- Otimização de performance para execução em dispositivos de baixo poder computacional
- Adição de autenticação em dois fatores
- Integração com sistemas de controle de acesso

## Solução de Problemas

### Erro ao carregar modelos
- Verifique se os modelos foram baixados corretamente
- Certifique-se de que os caminhos estão configurados corretamente

### Detecção facial não funciona corretamente
- Verifique a iluminação do ambiente
- Ajuste o parâmetro `min_face_size` na configuração

### Falhas na verificação de vivacidade
- Ajuste o parâmetro `validation_strictness` para tornar a validação mais flexível
- Certifique-se de seguir as instruções na tela corretamente

### Baixa precisão no reconhecimento
- Aumente o número de imagens durante o cadastro
- Ajuste o limiar de similaridade (`similarity_threshold`)
- Recadastre o usuário em condições de iluminação variadas

---

Desenvolvido com ❤️ por Passos, Jujubs, Hugo, Celinho e Jajau
