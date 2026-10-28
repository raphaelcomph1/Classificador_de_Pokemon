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
