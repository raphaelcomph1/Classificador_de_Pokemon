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
```n
## Estrutura

Ver `CLAUDE.md`. Decisões técnicas ficam registradas em `docs/decisoes.md`.

## Aviso

As imagens do dataset vêm de busca na web e podem ter direitos autorais; elas não são redistribuídas aqui.
