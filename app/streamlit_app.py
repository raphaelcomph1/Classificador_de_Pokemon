"""Pokédex v1: envie uma foto e veja os palpites do modelo + dados da PokéAPI.

Uso:
    streamlit run app/streamlit_app.py
"""

from typing import BinaryIO

import av
import streamlit as st
from PIL import Image
from streamlit_paste_button import paste_image_button
from streamlit_webrtc import webrtc_streamer

from pokedex.camera import CAMERA_WIDTHS, capture_preset, resize_to_width
from pokedex.config import load_config, project_path
from pokedex.data.datasets import load_rgb, to_rgb
from pokedex.pokeapi import (
    STAT_NAMES,
    TYPE_COLORS,
    TYPE_NAMES,
    PokeAPIClient,
    PokemonInfo,
    display_name,
)
from pokedex.predict import Predictor
from pokedex.scanner import LiveScanner

MAX_STAT = 255  # maior valor possível de um stat base (escala das barras)
MULTIPLIER_LABELS = {4.0: "4x", 2.0: "2x", 0.5: "½x", 0.25: "¼x", 0.0: "0x"}

cfg = load_config("app")


@st.cache_resource(show_spinner="Carregando o modelo...")
def get_predictor() -> Predictor:
    """Carrega o modelo uma vez só (fica em memória entre interações)."""
    return Predictor(project_path(cfg["checkpoint"]))


@st.cache_resource
def get_client() -> PokeAPIClient:
    api = cfg["pokeapi"]
    return PokeAPIClient(api["base_url"], project_path(api["cache_dir"]), api["timeout_s"])


@st.cache_data(show_spinner="Consultando a PokéAPI...")
def get_pokemon(label: str) -> PokemonInfo:
    return get_client().pokemon(label)


def type_badge(type_name: str, suffix: str = "") -> str:
    """Selo colorido de tipo em HTML."""
    color = TYPE_COLORS.get(type_name, "#777")
    return (
        f'<span style="background:{color};color:#fff;padding:3px 12px;border-radius:14px;'
        f"margin:0 6px 6px 0;display:inline-block;font-size:0.85rem;font-weight:600;"
        f'text-shadow:0 1px 1px rgba(0,0,0,.35)">{TYPE_NAMES.get(type_name, type_name)}{suffix}</span>'
    )


def badges(multipliers: dict[str, float]) -> str:
    return "".join(
        type_badge(t, f" · {MULTIPLIER_LABELS.get(m, f'{m:g}x')}") for t, m in multipliers.items()
    )


def show_predictions(predictions: list[tuple[str, float]]) -> None:
    """Top-k com barras de probabilidade."""
    st.subheader("Palpites")
    for rank, (label, prob) in enumerate(predictions, 1):
        st.progress(prob, text=f"**{rank}. {display_name(label)}** — {prob:.1%}")
    top_prob = predictions[0][1]
    if top_prob < cfg["low_confidence"]:
        st.warning(
            "Não tenho muita certeza. A foto pode estar com pouca luz, cortada, ter mais de "
            "um Pokémon ou ser de um Pokémon fora da Gen 1."
        )


def show_card(info: PokemonInfo) -> None:
    """Cartão da Pokédex: sprite, tipos, tamanho, stats e relações de tipo."""
    left, right = st.columns([1, 2], gap="large")
    with left:
        if info.sprite_url:
            st.image(info.sprite_url, width="stretch")
    with right:
        st.markdown(f"### #{info.number:03d} · {info.name}")
        st.markdown("".join(type_badge(t) for t in info.types), unsafe_allow_html=True)
        st.caption(f"Altura {info.height_m:.1f} m · Peso {info.weight_kg:.1f} kg")

        st.markdown("**Stats base**")
        for key, label in STAT_NAMES.items():
            value = info.stats.get(key, 0)
            st.progress(min(value / MAX_STAT, 1.0), text=f"{label}: {value}")
        st.caption(f"Total: {sum(info.stats.values())}")

    st.markdown("**Fraco contra** (recebe mais dano)")
    st.markdown(badges(info.weaknesses) or "—", unsafe_allow_html=True)
    st.markdown("**Resiste a** (recebe menos dano)")
    st.markdown(badges(info.resistances) or "—", unsafe_allow_html=True)
    if info.immunities:
        st.markdown("**Imune a**")
        st.markdown(badges({t: 0.0 for t in info.immunities}), unsafe_allow_html=True)


def show_result(source: BinaryIO | Image.Image) -> None:
    """Fluxo completo para uma imagem: previsão + cartão do Pokémon escolhido."""
    image = to_rgb(source) if isinstance(source, Image.Image) else load_rgb(source)
    predictions = get_predictor().predict(image, k=cfg["top_k"])

    col_image, col_preds = st.columns([1, 1], gap="large")
    with col_image:
        st.image(image, caption="Sua imagem", width="stretch")
    with col_preds:
        show_predictions(predictions)

    st.divider()
    labels = [label for label, _ in predictions]
    chosen = st.radio(
        "Ver na Pokédex:", labels, format_func=display_name, horizontal=True, key="chosen"
    )
    try:
        show_card(get_pokemon(chosen))
    except Exception as exc:  # sem internet, API fora do ar etc.: a previsão continua visível
        st.error(f"Não foi possível consultar a PokéAPI agora ({type(exc).__name__}).")


def main() -> None:
    st.set_page_config(page_title="Pokédex AI", page_icon="🔴", layout="centered")
    st.title("🔴 Pokédex AI")
    st.caption("Envie a foto de um Pokémon da 1ª geração e descubra quem é.")

    if not project_path(cfg["checkpoint"]).exists():
        st.error(
            f"Modelo não encontrado em `{cfg['checkpoint']}`. Treine ou copie o `best.pt` para lá."
        )
        st.stop()

    with st.sidebar:
        st.header("Sobre")
        st.markdown(
            "- Modelo: **ResNet18** com transfer learning\n"
            "- **150 classes**: Gen 1 + Alolan Sandslash\n"
            "- Teste: **92,9%** de acurácia, **98%** no top-3\n"
            "- Dados de tipos e stats: [PokéAPI](https://pokeapi.co)"
        )
        st.caption("Não reconhece Nidoran♀ e Nidoran♂ (ausentes do dataset de treino).")

    # Um modo por vez: só o campo escolhido aparece, então nunca há duas imagens "concorrendo".
    modes = [*INPUT_MODES, SCANNER_MODE]
    mode = st.radio("Como enviar a imagem?", modes, horizontal=True, key="mode")
    if mode == SCANNER_MODE:
        run_scanner()
        return
    source = INPUT_MODES[mode]()
    if source is None:
        st.info("Envie uma imagem para começar.")
        return
    show_result(source)


def input_file() -> BinaryIO | None:
    return st.file_uploader("Imagem", type=["jpg", "jpeg", "png", "webp"])


def input_paste() -> Image.Image | None:
    """Botão que lê a imagem da área de transferência (Ctrl+C numa imagem ou um print)."""
    st.caption(
        "Copie uma imagem (botão direito → **Copiar imagem**, ou um print com **Win+Shift+S**) "
        "e clique no botão abaixo."
    )
    pasted = paste_image_button("📋 Colar imagem", key="paste", errors="raise")
    return pasted.image_data


def input_camera() -> Image.Image | None:
    width = st.select_slider(
        "Resolução da foto (largura)",
        options=CAMERA_WIDTHS,
        value=cfg["camera_width"],
        format_func=lambda w: f"{w} px",
        help="Mais resolução = foto mais nítida, mas arquivo maior. "
        "O modelo reduz tudo para 224×224 na hora de analisar.",
    )
    photo = st.camera_input("Tire uma foto", resolution=capture_preset(width))
    if photo is None:
        return None
    captured = load_rgb(photo)
    image = resize_to_width(captured, width)
    note = "" if captured.width >= width else " (a câmera não alcança a largura escolhida)"
    st.caption(
        f"Capturada em {captured.width}×{captured.height} → usada em "
        f"{image.width}×{image.height}{note}"
    )
    return image


def get_scanner() -> LiveScanner:
    """Um scanner por sessão (cada aba do navegador tem a sua média de quadros)."""
    if "scanner" not in st.session_state:
        predictor = get_predictor()
        sc = cfg["scanner"]
        st.session_state.scanner = LiveScanner(
            predictor.probabilities,
            predictor.top_k,
            display_name,
            square_fraction=sc["square_fraction"],
            smoothing_frames=sc["smoothing_frames"],
            min_interval_s=sc["min_interval_s"],
            confident=sc["confident"],
            uncertain=cfg["low_confidence"],
        )
    return st.session_state.scanner


def run_scanner() -> None:
    """Vídeo ao vivo com mira: enquadre o Pokémon no quadrado e o nome aparece em cima."""
    st.caption(
        "Clique em **Iniciar**, permita a câmera e enquadre o Pokémon na mira. "
        "🟩 verde = confiante · 🟨 amarelo = em dúvida · ⬜ cinza = procurando."
    )
    width = st.select_slider(
        "Resolução do vídeo (largura)",
        options=CAMERA_WIDTHS,
        value=cfg["camera_width"],
        format_func=lambda w: f"{w} px",
        key="scanner_width",
        help="Mais resolução deixa o vídeo mais nítido, mas pesa mais para a CPU.",
    )
    scanner = get_scanner()

    def on_frame(frame: av.VideoFrame) -> av.VideoFrame:
        # Roda numa thread de vídeo separada: não pode usar st.* aqui dentro.
        out = scanner.process(frame.to_ndarray(format="rgb24"))
        return av.VideoFrame.from_ndarray(out, format="rgb24")

    ctx = webrtc_streamer(
        key=f"scanner-{width}",  # trocar a resolução reinicia a câmera com o novo pedido
        video_frame_callback=on_frame,
        media_stream_constraints={"video": {"width": {"ideal": width}}, "audio": False},
        translations=WEBRTC_PT,
    )
    if not ctx.state.playing:
        scanner.smoother.clear()
        return

    if st.button("🔍 Ver na Pokédex o que está na mira", type="primary"):
        result = scanner.result
        if result.label is None:
            st.info("Ainda não reconheci nada na mira. Aproxime o Pokémon e tente de novo.")
            return
        show_predictions(result.top)
        st.divider()
        try:
            show_card(get_pokemon(result.label))
        except Exception as exc:
            st.error(f"Não foi possível consultar a PokéAPI agora ({type(exc).__name__}).")


WEBRTC_PT = {
    "start": "Iniciar",
    "stop": "Parar",
    "select_device": "Escolher câmera",
    "media_api_not_available": "Câmera indisponível neste navegador",
    "device_ask_permission": "Permita o acesso à câmera",
    "device_not_available": "Nenhuma câmera encontrada",
    "device_access_denied": "Acesso à câmera negado",
}
INPUT_MODES = {"📁 Arquivo": input_file, "📋 Colar": input_paste, "📷 Câmera": input_camera}
SCANNER_MODE = "🎯 Scanner ao vivo"


main()
