# Pokédex AI

Projeto pessoal de estudo: um classificador de Pokémon por imagem (transfer learning com PyTorch) que evolui para API, dashboard no Power BI e site web.

## Objetivo e perfil do usuário

- O usuário é estudante de Ciência da Computação (FIAP) e quer **aprender com o projeto**, não só ter o código pronto.
- Ele usa **Windows (PowerShell)**. Use `pathlib`, evite comandos exclusivos de Linux e dê comandos para PowerShell quando precisar.
- Responda em **português**. Código, nomes de variáveis, funções e pastas em **inglês**.

## Como trabalhar comigo

1. **Uma fase por vez.** Não adiantar fases. Ao terminar uma, resuma o que foi feito e espere confirmação.
2. **Explique as decisões.** Para cada escolha relevante (modelo, hiperparâmetro, biblioteca), registre em `docs/decisoes.md` com data, alternativas consideradas e o porquê. Esse arquivo é meu material de estudo.
3. **Parte central é minha.** Na Fase 2, NÃO escreva sozinho o loop de treino nem a troca da camada final do modelo. Entregue um esqueleto com `# TODO` e comentários guiando, deixe eu escrever e depois revise. O resto (API, interface, infra, testes) pode escrever direto.
4. **Pergunte antes de** baixar dados grandes, instalar dependências pesadas ou mudar a estrutura de pastas.
5. **Commits pequenos e frequentes**, um por tarefa coerente, com mensagem clara.
6. Se algo der errado, explique a causa antes de corrigir.

## Stack

- Python 3.11+, ambiente virtual em `.venv` (ou `uv`)
- PyTorch + torchvision (transfer learning)
- Treino em **Google Colab com GPU**; localmente apenas teste rápido em CPU (`--smoke-test`, poucos batches)
- scikit-learn (métricas), matplotlib/seaborn (gráficos)
- FastAPI (Fase 4), Streamlit ou Gradio (Fase 3)
- pytest, ruff

## Estrutura de pastas

```
pokedex-ai/
├── CLAUDE.md
├── README.md
├── pyproject.toml          # ou requirements.txt
├── configs/                # YAML com hiperparâmetros
├── data/                   # NÃO versionar (.gitignore)
│   ├── raw/
│   └── processed/          # train/ val/ test/
├── docs/
│   └── decisoes.md         # log de decisões (material de estudo)
├── notebooks/              # exploração e o notebook do Colab
├── src/pokedex/
│   ├── data/               # download, checagem, split, datasets, transforms
│   ├── models/             # construção do modelo
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app/                    # interface (Fase 3)
├── api/                    # FastAPI (Fase 4)
├── models/                 # pesos treinados (.pt) — não versionar
└── tests/
```

## Fases (cada uma com critério de pronto)

**Fase 0 — Setup.** Repositório, ambiente, `.gitignore` (data/, models/, .venv), README inicial.
Pronto quando: `pytest` roda (mesmo com 1 teste trivial) e o ambiente se recria do zero.

**Fase 1 — Dados.**
- Dataset principal: **lantian773030/pokemonclassification** (Kaggle, ~6,5 mil imagens, Gen 1). Alternativa: dataset de Ben Hawks (Gen 1). Complemento futuro: vishalsubbiah/pokemon-images-and-types (imagem limpa + tipos).
- Contar as classes de verdade (fontes divergem entre 149 e 150) e listar imagens por classe.
- Detectar arquivos corrompidos e **duplicatas/quase duplicatas** (hash perceptual, ex.: `imagehash`).
- Split **estratificado** treino/validação/teste (ex.: 70/15/15) **antes** de qualquer augmentation, com seed fixa. Duplicatas devem ficar no mesmo split para evitar vazamento de dados.
Pronto quando: existe um relatório (`docs/dados.md`) com contagens, desbalanceamento e duplicatas tratadas.

**Fase 2 — Transfer learning (núcleo do projeto).**
- Baseline simples primeiro, depois MobileNetV3 e/ou ResNet18/50 pré-treinados no ImageNet.
- Estratégia: congelar a base, treinar só a cabeça; depois fine-tuning das últimas camadas com learning rate menor.
- Augmentation só no treino; validação/teste só com resize + normalização do ImageNet.
- Métricas: acurácia, top-3, matriz de confusão, F1 por classe. Registrar as classes mais confundidas.
Pronto quando: modelo salvo em `models/`, métricas no teste documentadas e comparação entre ao menos 2 arquiteturas em `docs/decisoes.md`.

**Fase 3 — Pokédex v1.** Interface com upload de imagem → top-3 previsões com probabilidade → dados da PokéAPI (tipos, stats, fraquezas). Cachear respostas da PokéAPI.

**Fase 4 — API.** FastAPI com `POST /predict`, `GET /pokemon/{name}`, `GET /health`. Registrar cada predição (timestamp, classe, confiança) em SQLite ou CSV.

**Fase 5 — Dashboard Power BI.** Ler os registros da API + stats dos Pokémon. Medidas em DAX.

**Fase 6 — Site web.** Front-end consumindo a API, com deploy.

**Fase 7 — Extra.** Detecção com YOLO em screenshots de jogos (Quetzal/Cobblemon).

## Convenções técnicas

- Fixar seeds (`random`, `numpy`, `torch`) e registrar a seed usada.
- Hiperparâmetros em `configs/*.yaml`, não espalhados no código.
- Detectar device automaticamente (`cuda` se disponível, senão `cpu`).
- No Windows, `DataLoader` com `num_workers>0` exige código protegido por `if __name__ == "__main__":`. Em teste local use `num_workers=0`.
- Funções pequenas, com type hints e docstring curta. Testes para split, transforms e predict.
- Nunca commitar datasets, pesos treinados, chaves ou tokens.
- As imagens do dataset vêm de busca na web e podem ter direitos autorais. Para o site público (Fase 6), usar sprites/dados da PokéAPI e não redistribuir o dataset.

## Primeira tarefa

Executar a **Fase 0** e começar a **Fase 1**: criar a estrutura, o ambiente e o `.gitignore`, e escrever o script de download e contagem de classes. Pare ao final da checagem de dados e me mostre o relatório antes de fazer o split.
