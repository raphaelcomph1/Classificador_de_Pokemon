# Log de decisões

Formato: data · decisão · alternativas · por quê.

---

## 2026-10-28 · Python 3.11 no ambiente local

- **Alternativas:** Python 3.14 (também instalado na máquina).
- **Por quê:** o PyTorch costuma demorar a publicar wheels para versões novas do Python, e o Colab roda 3.11/3.12. Usar 3.11 localmente deixa o ambiente parecido com o de treino e evita surpresas de compatibilidade.

## 2026-10-28 · `venv` + `pip` com `pyproject.toml`

- **Alternativas:** `uv` (mais rápido), `requirements.txt`, Poetry.
- **Por quê:** `uv` não está instalado e `venv`/`pip` vêm junto com o Python — menos uma ferramenta para aprender agora. O `pyproject.toml` é o padrão atual e já concentra dependências, config do pytest e do ruff. O pacote é instalado em modo editável (`pip install -e .`), com layout `src/`, para que `import pokedex` funcione nos testes e scripts sem gambiarra de `sys.path`.

## 2026-10-28 · Dependências pesadas adiadas

- **Decisão:** `torch`, `torchvision`, `scikit-learn` etc. só serão instalados na Fase 2.
- **Por quê:** a Fase 1 só precisa de Pillow (ler imagens) e `imagehash` (duplicatas). O PyTorch tem centenas de MB e o treino de verdade será no Colab.

## 2026-10-28 · `kagglehub` para baixar o dataset

- **Alternativas:** CLI `kaggle` (exige `kaggle.json` com token), download manual pelo site.
- **Por quê:** o `kagglehub` baixa datasets públicos sem token e permite escolher a pasta de destino (`output_dir`), então o download fica reproduzível com um único comando.

## 2026-10-28 · Checagem de imagens: `verify()` + `load()` do Pillow

- **Por quê:** `verify()` só confere a estrutura do arquivo; um JPEG truncado pode passar nele. O `load()` decodifica os pixels de verdade e pega esses casos. Usar os dois custa pouco e pega mais problemas.

## 2026-10-28 · Duplicatas por hash perceptual (pHash), limiar 4

- **Alternativas:** só MD5 (pega apenas cópias byte a byte), aHash/dHash (mais rápidos, porém mais sensíveis a mudanças de brilho), embeddings de CNN (melhores, mas exigem o PyTorch agora).
- **Por quê:** a mesma imagem costuma aparecer redimensionada ou recomprimida em sites diferentes, e aí o MD5 muda. O pHash (DCT da imagem reduzida) se mantém parecido nesses casos. A distância de Hamming ≤ 4 (de 64 bits) é um limiar conservador e fica em `configs/data.yaml` para ajuste depois de olhar os grupos encontrados.
- **Agrupamento:** union-find. Se A≈B e B≈C, as três imagens ficam no mesmo grupo e vão juntas para o mesmo split, o que evita vazamento.
- **Manifesto:** `data/processed/manifest.csv` guarda, por arquivo, a classe, se está ok, os hashes e o `dup_group`. O split vai partir desse arquivo.
