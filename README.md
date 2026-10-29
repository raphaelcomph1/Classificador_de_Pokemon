# Pokédex AI

Classificador de Pokémon (Gen 1) por imagem usando transfer learning com PyTorch.
Projeto de estudo que evolui para API (FastAPI), dashboard (Power BI) e site web.

## Setup (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

## Dados

Dataset: [lantian773030/pokemonclassification](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) (Kaggle).
Os dados **não** são versionados. Para guardá-los fora do projeto (ex.: fora do OneDrive):

```powershell
[Environment]::SetEnvironmentVariable("POKEDEX_DATA_ROOT", "C:\caminho\para\dados", "User")
```

```powershell
python -m pokedex.data.download      # baixa para data/raw/
python -m pokedex.data.check         # contagens, corrompidos, duplicatas -> docs/dados.md
python -m pokedex.data.split         # split estratificado -> processed/{train,val,test}/
```

## Treino (Fase 2)

```powershell
# dependências de ML (local: torch só CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,ml]"

python -m pokedex.train --arch small_cnn --smoke-test        # teste rápido na CPU
python -m pokedex.train --arch mobilenet_v3_large            # treino de verdade (Colab/GPU)
python -m pokedex.evaluate models/mobilenet_v3_large/best.pt # métricas no teste
python -m pokedex.predict imagem.jpg --checkpoint models/mobilenet_v3_large/best.pt
```

### Treino na GPU (Google Colab)

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelcomph1/Classificador_de_Pok-mon/blob/main/notebooks/train_colab.ipynb)

O notebook `notebooks/train_colab.ipynb` baixa os dados, refaz o split (idêntico, por causa da seed),
treina as arquiteturas, avalia no teste e salva os modelos no Google Drive.

## Pokédex v1 (interface)

Envie a foto de um Pokémon e veja os 3 palpites do modelo com probabilidade, além de número,
tipos, stats, fraquezas e resistências vindos da [PokéAPI](https://pokeapi.co) (com cache local).

```powershell
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Precisa do modelo treinado em `models/resnet18/best.pt` (gerado pelo notebook do Colab).

Formas de enviar a imagem: arquivo, colar da área de transferência, foto pela câmera ou
**scanner ao vivo**, em que um quadrado acompanha o Pokémon no vídeo e mostra quem é.
A mira dinâmica do scanner usa o YOLO-World para localizar e o classificador para identificar.
Para prepará-la, uma vez só:

```powershell
pip install -e ".[detector]"
python -m pokedex.detector   # gera models/yolo/pokedex-world-s.pt
```

## Resultados

150 classes (Gen 1 + Alolan Sandslash), 6.760 imagens. Split estratificado 70/15/15 (seed 42).
Métricas no conjunto de teste (1.016 imagens):

| Modelo | Acurácia | Top-3 | F1 macro |
|---|---|---|---|
| CNN simples (treinada do zero) | 35,6% | 59,8% | 0,32 |
| MobileNetV3-Large (transfer learning) | 88,5% | 96,8% | 0,88 |
| **ResNet18 (transfer learning)** | **92,9%** | **98,0%** | **0,93** |

Os erros restantes se concentram em pares de evolução visualmente parecidos (Krabby/Kingler,
Poliwhirl/Poliwrath, Grimer/Muk). Análise completa em `docs/decisoes.md`.

## Documentação

- `docs/dados.md`: relatório do dataset (contagens, desbalanceamento, duplicatas, split).
- `docs/decisoes.md`: decisões técnicas, com alternativas consideradas e justificativas.

## Aviso

As imagens do dataset vêm de busca na web e podem ter direitos autorais; elas não são redistribuídas neste repositório.
