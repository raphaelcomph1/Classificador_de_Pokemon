# Relatório de dados

Gerado por `python -m pokedex.data.check` em 2026-10-28.
Dataset: `lantian773030/pokemonclassification` — pasta `raw/PokemonData`.

## Resumo

| Item | Valor |
|---|---|
| Classes (pastas) | 150 |
| Arquivos | 6837 |
| Imagens válidas | 6825 |
| ↳ com extensão errada (recuperadas) | 13 |
| Arquivos com problema | 12 |
| Cópias exatas (MD5) além da 1ª | 12 |
| Grupos de (quase) duplicatas, pHash ≤ 4 | 55 (55 imagens a mais) |
| Grupos com classes diferentes | 10 |
| Imagens com lado < 32 px | 0 |
| **Dataset final: classes / imagens** | **149 / 6764** |

## Exclusões

Nada é apagado de `raw/`: as exclusões ficam marcadas no `manifest.csv`
(colunas `excluded` e `exclude_reason`) e as regras vêm de `configs/data.yaml`.

- classe excluída: 41
- duplicata entre classes: 20
- arquivo inválido: 12
- Classes excluídas: Alolan Sandslash

## Desbalanceamento (dataset final)

- Mínimo: **25** (Alakazam) · Máximo: **66** (Venusaur)
- Média: 45.4 · Mediana: 46 · Desvio padrão: 6.8
- Razão máx/mín: **2.6x**
- Classes com menos da metade da mediana: 0

## Formatos

- Extensões: {'.jpg': 6659, '.png': 121, '.jpeg': 40, '.svg': 12, ".jpg')": 1, '.sb-334870d9-fk2tvi': 1, '.sb-334870d9-nld60h': 1, '.asp': 1, '.jpg~c200': 1}
- Modos de cor: {'RGB': 6713, 'RGBA': 107, 'L': 3, 'P': 2}
- Largura: mín 43, mediana 354, máx 5000
- Altura: mín 42, mediana 353, máx 4356

## Arquivos com problema

- `Cloyster/ff270ebfab0f46b3b05c3fecd6a15ef9.svg` — não é imagem raster suportada (ex.: SVG, HTML)
- `Dewgong/e3ecbb47e192470aa8686734b7813b43.svg` — não é imagem raster suportada (ex.: SVG, HTML)
- `Exeggutor/2b8bc491c5ca4e788999f8ea53617638.svg` — não é imagem raster suportada (ex.: SVG, HTML)
- `Exeggutor/f29b4f9fccff4055b167ece6556143cd.svg` — não é imagem raster suportada (ex.: SVG, HTML)
- `Golduck/bc08a691e7cb41b0bec2adf931dc9634.svg` — não é imagem raster suportada (ex.: SVG, HTML)
- `Graveler/3fe3c6a0bbd14de89092d2c675f743c1.svg` — não é imagem raster suportada (ex.: SVG, HTML)
- `Magneton/45e434d002bd44eeb1acbc2decc8ee0f.svg` — não é imagem raster suportada (ex.: SVG, HTML)
- `Omastar/1be6e7da1bba4379b298b20013c57b4b.svg` — não é imagem raster suportada (ex.: SVG, HTML)
- `Scyther/a8734f10941a4deb95fcf7388e4fe8e2.svg` — não é imagem raster suportada (ex.: SVG, HTML)
- `Tentacruel/d6fb0a3046364ea2b49525c6f6b966e5.svg` — não é imagem raster suportada (ex.: SVG, HTML)
- `Venomoth/bb5d47bced6e4e33a77faa17e1eec906.svg` — não é imagem raster suportada (ex.: SVG, HTML)
- `Zapdos/c99b2d19544c4cbbacd7711ade660279.svg` — não é imagem raster suportada (ex.: SVG, HTML)

## Recuperadas (conteúdo válido, extensão errada)

- `Bulbasaur/00000003.png` — na verdade é JPEG
- `Bulbasaur/00000006.jpg` — na verdade é PNG
- `Bulbasaur/00000022.png` — na verdade é JPEG
- `Charmander/00000044.png` — na verdade é JPEG
- `Ekans/b24c3fa1a8b742af9af97c58c002c85b.jpg')` — na verdade é JPEG
- `Mewtwo/00000008.png.sb-334870d9-FK2TVI` — na verdade é JPEG
- `Mewtwo/00000042.png.sb-334870d9-nlD60H` — na verdade é JPEG
- `Muk/d4ddd14c1d4340a680018d0b805bd1b2.asp` — na verdade é JPEG
- `Pikachu/00000009.png` — na verdade é JPEG
- `Pikachu/00000040.png` — na verdade é JPEG
- `Pikachu/00000055.png` — na verdade é JPEG
- `Squirtle/00000015.png` — na verdade é JPEG
- `Wigglytuff/4f3c72df431a49d4b3fbc84c94f2261e.jpg~c200` — na verdade é JPEG

## Duplicatas entre classes diferentes

Pares de evolução com a mesma imagem (rótulo ambíguo). Arquivos de cada grupo:

- `Alakazam/64527ef991ff42c6883623f263f3c2f4.jpg`, `Kadabra/f02640d4ea714d91ba16fafd46b3a49b.jpg`
- `Cubone/9d5269c2392b49b6b8f37e15ff284351.jpg`, `Marowak/f2f2769669734a28bb848108a8da09ad.jpg`
- `Dewgong/b605edb4e7544d5e80d13153fde73f8a.jpg`, `Seel/2ee695fb27c94e9faedc3c2b3eda757b.jpg`
- `Dodrio/4c310176bd664c0aab3f997323dee002.jpg`, `Doduo/616c2218a8f84f6aa879fc141b395b7a.jpg`
- `Grimer/6e8e877680bb4664966d0128f76223a9.jpg`, `Muk/d99a419aea0f414a86f5b28d29cf4c6f.jpg`
- `Kingler/c93362919a154cf4ac09703cf3819f33.jpg`, `Krabby/b8bb78e3da4c49a2afadc98f7f7d80f3.jpg`
- `Mankey/17d9534325c148a083b4e1526ca3646c.jpg`, `Primeape/6e6834872cc54d50b40f71018d243bf8.jpg`
- `Pidgeot/95d8d837370a4433b3d8428868ab026e.jpg`, `Pidgeotto/c9d0e5645c8a41bb98e1924728130028.jpg`
- `Poliwhirl/82158b93b226423bbe2cbe3c7dd6cdfd.jpg`, `Poliwrath/f3d9b13bfb0846fa996b9ec733aaf45b.jpg`
- `Poliwhirl/ef4d09c9e96c4fc1b2bdb1b427a92820.jpg`, `Poliwrath/bc26a7bff0124a82a24e2bb8cea32679.jpg`

## Exemplos de duplicatas na mesma classe (até 15 grupos)

- `Alolan Sandslash/1fc0f99cc95240138620f2b8a1ba86ec.jpg`, `Alolan Sandslash/ce945e0129244681a44c925016667ae0 copy.jpg`
- `Alolan Sandslash/6525ea05a6aa4932b6aa6307f287d0c3.jpg`, `Alolan Sandslash/f5d8df5c3e03418d8c92b065a6ba7274.jpg`
- `Arbok/ad5202b436824705b1ed0675eaa39454.jpg`, `Arbok/fea033db0e48467090dd3067e7769d3d.jpg`
- `Bulbasaur/00000004.png`, `Bulbasaur/24e2594b0dc14e5895d894b2e636f150.jpg`
- `Bulbasaur/00000006.jpg`, `Bulbasaur/00000022.png`
- `Charmeleon/c2f40ab63f3d4e70a3178156e70d46e2.jpg`, `Charmeleon/e92e45465b6645f6bcf6367d13d21a0f.jpg`
- `Clefairy/bcb44348ad7644b2b9ddd3cfdcdbf313.jpg`, `Clefairy/da5f403219a741eab4e86c34ab5f52d3.jpg`
- `Cloyster/0049805978be4d23a3fa9e454928982c.jpg`, `Cloyster/a93ed5150b0b4bc6b3d9bb8d9bb3485b.jpg`
- `Drowzee/61c53caf1acf4dc3b1417ad19bfd2cdb.jpg`, `Drowzee/f4aa999f94e942a6b7a8094283feeffa.jpg`
- `Dugtrio/ae770e3e15af4837869e61444c330cc3.jpg`, `Dugtrio/baf1c7f8608e4a0582432818e13ce0c9.jpg`
- `Electrode/7fe8852c83644ac18026fc3ffdf017b2.jpg`, `Electrode/de6b731598dc433ca4e600183dd91029.jpg`
- `Fearow/383d67ea6c164d6ea44d7c8be153b016.jpg`, `Fearow/888108f0922443b1822c3ee8bab82bf4.jpg`
- `Fearow/4a2cad2797754dc4968083a1f718d871.jpg`, `Fearow/8034ffa695c74b0aa4400e9768c76cd2.jpg`
- `Flareon/23f57c80ee784b63b23e4cea6cd6f454.jpg`, `Flareon/a403f06e227940019826d96deb7274b5.jpg`
- `Grimer/1feaee12c55749608a69bc1095d2819f.jpg`, `Grimer/e489c8f2a9be405bbdd621e1b0628e4c.jpg`

## Imagens por classe (dataset final)

<details><summary>Tabela completa</summary>

| Classe | Imagens |
|---|---|
| Abra | 35 |
| Aerodactyl | 35 |
| Alakazam | 25 |
| Alolan Sandslash | — (excluída) |
| Arbok | 41 |
| Arcanine | 41 |
| Articuno | 45 |
| Beedrill | 34 |
| Bellsprout | 40 |
| Blastoise | 42 |
| Bulbasaur | 41 |
| Butterfree | 27 |
| Caterpie | 37 |
| Chansey | 43 |
| Charizard | 35 |
| Charmander | 51 |
| Charmeleon | 46 |
| Clefable | 38 |
| Clefairy | 46 |
| Cloyster | 48 |
| Cubone | 39 |
| Dewgong | 42 |
| Diglett | 39 |
| Ditto | 40 |
| Dodrio | 38 |
| Doduo | 34 |
| Dragonair | 42 |
| Dragonite | 48 |
| Dratini | 26 |
| Drowzee | 39 |
| Dugtrio | 51 |
| Eevee | 37 |
| Ekans | 36 |
| Electabuzz | 41 |
| Electrode | 51 |
| Exeggcute | 46 |
| Exeggutor | 50 |
| Farfetchd | 48 |
| Fearow | 48 |
| Flareon | 45 |
| Gastly | 44 |
| Gengar | 49 |
| Geodude | 35 |
| Gloom | 49 |
| Golbat | 48 |
| Goldeen | 47 |
| Golduck | 46 |
| Golem | 46 |
| Graveler | 39 |
| Grimer | 43 |
| Growlithe | 44 |
| Gyarados | 46 |
| Haunter | 48 |
| Hitmonchan | 47 |
| Hitmonlee | 48 |
| Horsea | 54 |
| Hypno | 47 |
| Ivysaur | 40 |
| Jigglypuff | 46 |
| Jolteon | 51 |
| Jynx | 49 |
| Kabuto | 49 |
| Kabutops | 42 |
| Kadabra | 48 |
| Kakuna | 47 |
| Kangaskhan | 51 |
| Kingler | 46 |
| Koffing | 48 |
| Krabby | 47 |
| Lapras | 53 |
| Lickitung | 48 |
| Machamp | 53 |
| Machoke | 42 |
| Machop | 41 |
| Magikarp | 44 |
| Magmar | 44 |
| Magnemite | 46 |
| Magneton | 41 |
| Mankey | 54 |
| Marowak | 29 |
| Meowth | 26 |
| Metapod | 53 |
| Mew | 54 |
| Mewtwo | 54 |
| Moltres | 43 |
| MrMime | 45 |
| Muk | 42 |
| Nidoking | 52 |
| Nidoqueen | 52 |
| Nidorina | 41 |
| Nidorino | 51 |
| Ninetales | 44 |
| Oddish | 51 |
| Omanyte | 45 |
| Omastar | 43 |
| Onix | 42 |
| Paras | 47 |
| Parasect | 43 |
| Persian | 37 |
| Pidgeot | 42 |
| Pidgeotto | 39 |
| Pidgey | 48 |
| Pikachu | 62 |
| Pinsir | 41 |
| Poliwag | 51 |
| Poliwhirl | 48 |
| Poliwrath | 29 |
| Ponyta | 45 |
| Porygon | 46 |
| Primeape | 50 |
| Psyduck | 40 |
| Raichu | 51 |
| Rapidash | 48 |
| Raticate | 41 |
| Rattata | 43 |
| Rhydon | 50 |
| Rhyhorn | 41 |
| Sandshrew | 46 |
| Sandslash | 45 |
| Scyther | 57 |
| Seadra | 48 |
| Seaking | 46 |
| Seel | 47 |
| Shellder | 46 |
| Slowbro | 46 |
| Slowpoke | 41 |
| Snorlax | 59 |
| Spearow | 55 |
| Squirtle | 52 |
| Starmie | 49 |
| Staryu | 47 |
| Tangela | 53 |
| Tauros | 54 |
| Tentacool | 45 |
| Tentacruel | 47 |
| Vaporeon | 59 |
| Venomoth | 52 |
| Venonat | 47 |
| Venusaur | 66 |
| Victreebel | 49 |
| Vileplume | 56 |
| Voltorb | 55 |
| Vulpix | 50 |
| Wartortle | 48 |
| Weedle | 49 |
| Weepinbell | 43 |
| Weezing | 50 |
| Wigglytuff | 57 |
| Zapdos | 51 |
| Zubat | 40 |

</details>
