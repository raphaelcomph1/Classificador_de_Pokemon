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

## Documentação

- `docs/dados.md`: relatório do dataset (contagens, desbalanceamento, duplicatas, split).
- `docs/decisoes.md`: decisões técnicas, com alternativas consideradas e justificativas.

## Aviso

As imagens do dataset vêm de busca na web e podem ter direitos autorais; elas não são redistribuídas neste repositório.
