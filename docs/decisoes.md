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

## 2026-10-28 · Dados fora do OneDrive (`POKEDEX_DATA_ROOT`)

- **Alternativas:** manter em `data/` dentro do projeto; colocar caminho absoluto no YAML.
- **Por quê:** o projeto está no OneDrive, que sincronizaria milhares de imagens. Caminho absoluto no YAML quebraria no Colab. Uma variável de ambiente resolve os dois casos: localmente aponta para `C:\Users\r4fae\datasets\pokedex`, e no Colab pode apontar para o Drive. Sem a variável, o padrão continua `data/`.

## 2026-10-28 · Exclusões marcadas no manifesto, não apagadas

- **Alternativas:** apagar os arquivos de `raw/`; copiar só os bons para outra pasta.
- **Por quê:** `raw/` fica idêntico ao download, então qualquer exclusão pode ser desfeita mudando `configs/data.yaml`. O `manifest.csv` registra *o quê* foi excluído e *por quê* (`exclude_reason`), o que facilita auditar depois.

## 2026-10-28 · Formas regionais (Alolan Sandslash) fora do dataset — *substituída pela decisão abaixo*

- **Contexto:** o dataset tem 149 espécies da Gen 1 (faltam Nidoran♀ e Nidoran♂) + 1 forma regional, Alolan Sandslash. Por isso as fontes divergem entre 149 e 150 classes.
- **Alternativas:** (a) classe separada; (b) juntar com Sandslash; (c) excluir.
- **Por quê (c):** o classificador aprende **aparência**. A forma de Alola tem aparência bem diferente (gelo, azul), então juntar com o Sandslash normal (b) faria uma classe com dois visuais distintos. Mantê-la como classe própria (a) foge do escopo Gen 1 e seria a única forma regional, sem Alolan Vulpix, Raichu etc. Se um dia as formas regionais entrarem, entram todas, cada uma como classe própria (ex.: `sandslash-alola`, o mesmo nome que a PokéAPI usa).
- **Consequência:** o modelo final tem **149 classes**. Nidoran♀/♂ ficam sem cobertura (anotar na Pokédex v1).

## 2026-10-28 · Formato detectado pelo conteúdo, não pela extensão

- **Por quê:** 13 imagens tinham extensão errada (ex.: JPEG salvo como `.asp` ou `.png`). O Pillow identifica o formato pelos bytes iniciais do arquivo (*magic number*). Aceito só JPEG/PNG/GIF/BMP/WEBP (`formats=`), e os SVGs (vetor) são descartados.
- **Atenção para o split:** o `ImageFolder` do torchvision filtra por extensão. Na hora de gerar `processed/`, os arquivos devem ser salvos com a extensão correta.

## 2026-10-28 · Duplicatas entre classes removidas

- **Por quê:** as 10 ocorrências eram pares de evolução com a mesma imagem (provavelmente artes com os dois Pokémon). Não existe rótulo certo para elas, e mantê-las ensinaria o modelo a confundir justamente as classes mais parecidas. Foram removidas as 20 imagens.

## 2026-10-28 · Revisão: formas regionais como classes próprias (150 classes)

- **Decisão:** Sandslash e Alolan Sandslash ficam como **duas classes separadas**. A exclusão anterior foi revertida (`excluded_classes: []`).
- **Por quê:** mantém todos os dados disponíveis, e classes separadas respeitam a regra de que *uma classe = uma aparência*. Vale como padrão para o futuro: toda forma regional que entrar vira classe própria, com o nome da PokéAPI na Fase 3 (`Alolan Sandslash` → `sandslash-alola`).
- **Ponto de atenção:** é o par mais parecido "por definição". Conferir na matriz de confusão da Fase 2 se o modelo os separa bem.

## 2026-10-28 · Duplicatas na mesma classe: manter só uma (opção B)

- **Alternativas:** (A) manter todas e sortear o split por grupo, com as cópias indo juntas; (B) manter só uma por grupo.
- **Por quê (B):** remove só 45 imagens (0,7%) e elimina o risco de vazamento por construção. Na opção A, a cópia extra continuaria no treino e "pesaria em dobro". Fica a imagem de **maior resolução**; em caso de empate, a primeira pelo caminho (regra determinística).

## 2026-10-28 · Split estratificado 70/15/15, seed 42

- **Como:** cada classe é embaralhada e dividida separadamente (estratificação), então todas as classes aparecem em todos os splits na mesma proporção. Os arquivos são ordenados antes do embaralhamento, para que o resultado dependa só da seed e não da ordem do sistema de arquivos.
- **Alternativas:** `train_test_split(stratify=...)` do scikit-learn (adiado: é dependência da Fase 2 e só faz duas partes por chamada); split aleatório simples (classes pequenas poderiam ficar sem exemplo no teste).
- **Arredondamento:** val e test recebem `round(n × 0,15)` com mínimo 1; o resto vai para treino. Resultado: 4.728 / 1.016 / 1.016. A classe menor (25 imagens) fica 17/4/4.
- **Saída:** `processed/{train,val,test}/<classe>/<md5>.<ext>`, no formato do `ImageFolder`. O nome pelo MD5 evita colisões e já corrige a extensão. `processed/splits.csv` liga cada cópia ao arquivo original.
- **Augmentation:** nenhuma até aqui. Ela entra só no treino, na Fase 2.

## 2026-10-28 · Fase 2: PyTorch CPU localmente

- **Por quê:** a versão CUDA do torch tem alguns GB e a máquina local só roda o `--smoke-test`. A versão CPU (`--index-url https://download.pytorch.org/whl/cpu`) é bem menor. No Colab o torch com CUDA já vem instalado.

## 2026-10-28 · Arquiteturas: SmallCNN (baseline), MobileNetV3-Large, ResNet18, ResNet50

- **Baseline do zero (`SmallCNN`, ~430 mil parâmetros):** serve de régua. Com só ~32 imagens de treino por classe, um modelo do zero deve ir mal. A diferença mostra *quanto* o pré-treino no ImageNet ajuda.
- **MobileNetV3-Large (~5,4 M):** leve e rápido, bom para a API e o site (Fases 4 e 6). É a arquitetura padrão da config.
- **ResNet18 (~11 M) / ResNet50 (~25 M):** arquitetura clássica e fácil de entender. A ResNet50 é mais pesada, mas costuma generalizar melhor.
- **Comparação justa:** mesma seed, mesmo split, mesmas transforms e mesmo número de épocas.

## 2026-10-28 · Treino em duas etapas

- **Etapa 1 (head):** base congelada, só a cabeça nova treina, com lr 1e-3 por 5 épocas. A cabeça nova começa aleatória. Se a base estivesse descongelada, os gradientes grandes do início destruiriam as features boas do ImageNet.
- **Etapa 2 (finetune):** descongela os 3 últimos blocos, com lr 1e-4 (10x menor) por 10 épocas. Os últimos blocos têm features mais específicas (formas, partes de objetos) e se beneficiam do ajuste. Os primeiros (bordas, texturas) servem para qualquer imagem.
- **Seleção do modelo:** guarda o checkpoint com a melhor acurácia de **validação**. O teste só é usado no final, uma vez, para não "escolher pelo teste".

## 2026-10-28 · Augmentation e pré-processamento

- **Treino:** `RandomResizedCrop(224, scale 0.7–1.0)`, flip horizontal, rotação de ±10°, brilho/contraste ±20% e saturação ±10%.
- **Sem mudança de matiz (hue):** a cor é a pista principal entre classes parecidas (Sandslash vs. Alolan Sandslash, Pokémon shiny à parte).
- **Validação/teste:** `Resize(224×224)` + normalização do ImageNet, sem crop, para não cortar partes do Pokémon.
- **Transparência:** PNGs transparentes são compostos sobre fundo **branco**. O `convert("RGB")` padrão deixaria o fundo preto (ou com lixo), algo que nunca aparece numa foto real.

## 2026-10-28 · Loss e otimizador

- **CrossEntropy com `label_smoothing=0.1`:** em vez de exigir 100% de certeza na classe correta, o alvo vira 90% + 10% espalhado. Isso reduz a confiança excessiva, o que importa porque a Pokédex vai mostrar probabilidades no top-3.
- **AdamW, weight decay 1e-4, scheduler cosseno:** o AdamW é um bom padrão para fine-tuning e converge sem muito ajuste. O cosseno reduz o lr suavemente até o fim de cada etapa.

## 2026-10-28 · TODOs da Fase 2 implementados pelo Claude (a pedido)

- **Contexto:** a regra do CLAUDE.md era o aluno escrever a troca da cabeça e o loop de treino. Depois de estudar a teoria e fazer os exercícios (Linear, softmax/argmax, trocar `rede2[-1]`), o aluno pediu para o Claude finalizar a etapa e deixar a prática para **exercícios no fim do projeto**.
- **Como ficou:** os comentários de TODO viraram explicações linha a linha em `models/build.py` e `train.py`. Os testes de `tests/test_model.py` e `tests/test_train.py` servem de gabarito para os exercícios futuros.

## 2026-10-28 · "Bloco" no fine-tuning = bloco residual (ResNet) / bloco da `features` (MobileNet)

- **Problema encontrado no smoke test:** contar `layer1..layer4` como blocos fazia `unfreeze_last: 3` descongelar 99% da ResNet18 (cada layer é enorme), contra 72% da MobileNet.
- **Correção:** na ResNet, contar os blocos residuais de dentro dos layers (8 na ResNet18, 16 na ResNet50).
- **Por que ainda é muito (86% / 64% / 72%):** os parâmetros se concentram no fim da rede, onde há mais canais. É normal, e com lr 10x menor o ajuste continua delicado.
- **Ponto para estudar depois:** com a base congelada, o BatchNorm em `model.train()` ainda atualiza as estatísticas (média/variância) dos batches. Algumas receitas deixam o BN congelado em `eval()`. Vale testar se a acurácia mudar muito.
