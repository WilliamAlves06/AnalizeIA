# RELATÓRIO – Atividade Integradora
## Sistema Inteligente de Análise de Imagens com IA

---

## 🔹 a) Explique o pipeline

O pipeline do sistema segue uma sequência linear de transformações sobre a imagem,
da aquisição até a interpretação final:

1. **Aquisição (Etapa 1):** A imagem é carregada do disco via `cv2.imread()`, que
   internamente a decodifica como uma matriz NumPy tridimensional (altura × largura × 3
   canais BGR). Alternativamente, a webcam é acessada por `cv2.VideoCapture(0)`.

2. **Processamento (Etapa 2):** A imagem colorida é convertida para escala de cinza
   (`cvtColor → GRAY`), reduzindo de 3 canais para 1. O desfoque gaussiano
   (`GaussianBlur`) atenua ruídos de alta frequência, preparando a imagem para o
   detector de bordas de Canny, que aplica gradientes de intensidade para localizar
   transições abruptas de cor.

3. **Análise de cor / HSV (Etapa 3):** A imagem é re-mapeada para o espaço HSV
   (Matiz, Saturação, Valor). Os três canais são separados, o canal H (matiz) é
   rotacionado e a saturação pode ser amplificada. Isso demonstra como propriedades
   perceptuais de cor são manipuladas independentemente da luminosidade.

4. **Histograma (Etapa 4):** O histograma da imagem em escala de cinza conta a
   frequência de cada nível de intensidade (0–255). A distribuição resultante
   revela a faixa tonal predominante da cena.

5. **Binarização / Segmentação (Etapa 5):** Um limiar (threshold) separa pixels
   "escuros" de "claros", produzindo uma imagem binária preto-e-branco. Com Otsu,
   o limiar ótimo é calculado automaticamente a partir do histograma.

6. **IA – YOLOv8 (Etapa 6):** O modelo de detecção de objetos recebe o frame e,
   em uma única passagem pela rede neural, retorna caixas delimitadoras, classes e
   scores de confiança para cada objeto identificado.

7. **Resultado final (Etapa 7):** As informações são consolidadas: lista de objetos,
   classificação da cena, nível de iluminação e threshold utilizado.

---

## 🔹 b) Qual a função do histograma?

O histograma de intensidade mostra **quantos pixels existem em cada nível de
brilho** (0 = preto, 255 = branco). Sua utilidade no pipeline é:

- **Diagnóstico de iluminação:** Uma distribuição concentrada à esquerda indica
  imagem subexposta (escura); concentrada à direita, superexposta (clara). Uma
  distribuição ampla e uniforme indica boa dinâmica tonal.

- **Base para binarização inteligente:** O método de Otsu analisa o histograma
  para encontrar o ponto de corte que melhor separa fundo e objeto, minimizando
  a variância intra-classe.

- **Equalização:** Histogramas achatados permitem aumentar o contraste
  automaticamente via `cv2.equalizeHist()`.

Em resumo, o histograma transforma a percepção visual em dados quantitativos,
fundamentando decisões de pré-processamento.

---

## 🔹 c) Por que usar HSV?

O espaço RGB representa cor como uma combinação de vermelho, verde e azul —
canais altamente correlacionados que misturam cor e luminosidade. Isso dificulta
operações como "mudar apenas a tonalidade sem alterar o brilho".

O espaço **HSV** separa estas propriedades em três canais independentes:

| Canal | Significado             | Faixa (OpenCV) |
|-------|-------------------------|----------------|
| H     | Matiz (ângulo de cor)   | 0 – 179        |
| S     | Saturação (pureza)      | 0 – 255        |
| V     | Valor (luminosidade)    | 0 – 255        |

**Vantagens práticas:**
- Segmentar um objeto por cor (ex.: detectar frutas laranjas) é trivial: basta
  filtrar uma faixa de H, sem depender da iluminação (V).
- Alterar só a tonalidade (H) ou intensificar cores (S) sem distorcer o brilho.
- Robustez a variações de iluminação: algoritmos baseados em H+S ignoram mudanças
  de V causadas por sombras ou luz direta.

Por isso, HSV é o espaço padrão em aplicações de segmentação por cor e rastreamento
de objetos coloridos.

---

## 🔹 d) Diferença entre visão humana e computacional

| Aspecto              | Visão Humana                           | Visão Computacional                     |
|----------------------|----------------------------------------|-----------------------------------------|
| Representação        | Percepção subjetiva contínua           | Matriz de pixels (inteiros 0–255)       |
| Cor                  | Tricromática (cones L, M, S)           | BGR / RGB com 3 canais numéricos        |
| Contexto             | Compreensão semântica intuitiva        | Padrões estatísticos aprendidos         |
| Invariância          | Robusta a rotação, escala, iluminação  | Sensível sem augmentation/normalização  |
| Velocidade           | ~10–12 fixações/segundo                | Dezenas de frames por segundo           |
| Paralelismo          | Processamento paralelo maciço          | GPU com milhares de núcleos             |
| Aprendizado          | Contínuo, poucos exemplos (few-shot)   | Requer milhares/milhões de exemplos     |
| Falhas               | Ilusões ópticas, fadiga                | Adversarial examples, overfitting       |

O ser humano extrai **significado** de uma cena instantaneamente (reconhece uma
pessoa pelo jeito de andar). A visão computacional extrai **padrões matemáticos**
(bordas, texturas, frequências) e os associa a categorias por otimização estatística.
CNNs e modelos YOLO conseguem superar humanos em velocidade e consistência para
tarefas bem definidas, mas ainda são frágeis fora da distribuição de treinamento.

---

## 🔹 e) Qual técnica de IA foi usada?

Foi utilizada a **YOLOv8 (You Only Look Once – versão 8)**, uma rede neural
convolucional (CNN) de detecção de objetos em tempo real.

**Como funciona:**
- A imagem é redimensionada para 640×640 e processada por um backbone CNN
  (baseado em CSPDarknet / EfficientNet) que extrai features em múltiplas escalas.
- Um neck FPN (Feature Pyramid Network) combina features de diferentes resoluções.
- A cabeça de detecção prediz, para cada célula da grade, caixas delimitadoras
  (x, y, w, h), score de objetividade e probabilidades de classe.
- A inferência inteira ocorre em **uma única passagem** pela rede (daí "You Only
  Look Once"), tornando o modelo extremamente rápido.

**Por que YOLO e não CNN de classificação?**
- Classificação CNN → diz **o que** tem na imagem (uma classe global).
- YOLO → diz **o que** e **onde** (múltiplos objetos com localização), ideal
  para cenas complexas com vários objetos simultâneos.

**Modelo utilizado:** `yolov8n.pt` (nano) — menor e mais rápido, adequado para
demonstração em CPU. Modelos maiores (s, m, l, x) oferecem maior precisão.

---

*Relatório gerado como parte da Atividade Integradora – Visão Computacional com IA.*
