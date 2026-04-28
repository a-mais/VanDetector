"""
VanDetector - Interface Streamlit adaptada para UCF Crime Dataset.
Usa CNN (ResNet50) + MLP para classificação de frames.
"""

from __future__ import annotations

import csv
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import streamlit as st
import numpy as np
from PIL import Image

from src.ucf_features import (
    cv2_frame_to_features,
    frame_to_features,
    batch_extract_features,
    occlusion_suspicion_map,
    draw_suspicion_overlay,
)


LOG_PATH = Path("outputs/logs/ucf_crime_events.csv")
DEFAULT_MLP_MODEL = Path("models/ucf_crime_mlp.pkl")
DEFAULT_SCALER = Path("models/ucf_scaler.pkl")
DEFAULT_ASSET_VIDEO = Path("assets/vandalismbrasil2.mp4")

LOG_COLUMNS = [
    "timestamp",
    "source",
    "mode",
    "frame",
    "confidence",
    "prediction"  # "normal" ou "crime"
]

# Configuração de confidence
DEFAULT_CRIME_CONFIDENCE = 0.6
DEFAULT_TEMPORAL_WINDOW = 10  # Frames consecutivos
DEFAULT_MIN_CRIME_FRAMES = 7  # Mínimo de frames de crime na janela


def inject_icon_styles() -> None:
    """Adiciona Material Symbols para usar icones sem emojis."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded');
        .vd-icon-line {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0.25rem 0 0.75rem;
        }
        .vd-icon-line .material-symbols-rounded {
            font-family: 'Material Symbols Rounded';
            font-weight: normal;
            font-style: normal;
            font-size: 1.35rem;
            line-height: 1;
            letter-spacing: normal;
            text-transform: none;
            display: inline-block;
            white-space: nowrap;
            direction: ltr;
            -webkit-font-feature-settings: 'liga';
            -webkit-font-smoothing: antialiased;
        }
        .vd-icon-title {
            font-size: 2.1rem;
            font-weight: 700;
        }
        .vd-icon-section {
            font-size: 1.45rem;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def icon_heading(icon: str, text: str, css_class: str = "vd-icon-section") -> None:
    st.markdown(
        f"""
        <div class="vd-icon-line {css_class}">
            <span class="material-symbols-rounded">{icon}</span>
            <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_mlp_model(path: str):
    """Carrega modelo MLP treinado."""
    return joblib.load(path)


@st.cache_resource
def load_scaler(path: str):
    """Carrega scaler para normalização de features."""
    return joblib.load(path)


def classify_ucf_frame(model, frame_features: dict[str, float]) -> tuple[str, float]:
    """
    Classifica um frame usando MLP.
    
    Args:
        model: Modelo MLP treinado
        frame_features: Dict de features extraídas pela CNN
    
    Returns:
        (prediction: "crime" ou "normal", probability: 0-1)
    """
    
    # Converter dict para array na mesma ordem numerica usada no treino.
    feature_names = sorted(frame_features.keys(), key=lambda name: int(name.rsplit("_", 1)[-1]))
    X = np.array([frame_features[name] for name in feature_names]).reshape(1, -1)
    
    # Usar sempre P(crime), para o limiar da interface controlar a classe positiva.
    crime_probability = float(model.predict_proba(X)[0][1])
    label = "crime" if crime_probability >= 0.5 else "normal"
    return label, crime_probability


def write_log(timestamp: str, source: str, mode: str, frame_num: int, 
             confidence: float, prediction: str):
    """Registra detecção em log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        
        # Cabeçalho se arquivo novo
        if LOG_PATH.stat().st_size == 0:
            writer.writeheader()
        
        writer.writerow({
            "timestamp": timestamp,
            "source": source,
            "mode": mode,
            "frame": frame_num,
            "confidence": f"{confidence:.4f}",
            "prediction": prediction
        })


def process_video(video_path: str, model, scaler, 
                 confidence_threshold: float = DEFAULT_CRIME_CONFIDENCE,
                 temporal_window: int = DEFAULT_TEMPORAL_WINDOW,
                 min_crime_frames: int = DEFAULT_MIN_CRIME_FRAMES,
                 live_preview: bool = True,
                 preview_every: int = 1,
                 highlight_suspicious_area: bool = True,
                 collect_crime_frames: bool = True,
                 max_crime_frames: int = 80):
    """
    Processa vídeo e detecta crimes.
    
    Returns:
        list de eventos detectados
    """
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Buffer temporal
    crime_buffer = deque(maxlen=temporal_window)
    events = []
    crime_frames = []
    logged_event_frames = set()
    frame_count = 0
    
    progress_placeholder = st.empty()
    predictions_placeholder = st.empty()
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with predictions_placeholder.container():
        col1, col2, col3 = st.columns(3)
        crime_count_display = col1.empty()
        normal_count_display = col2.empty()
        confidence_display = col3.empty()
    
    crime_count = 0
    normal_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extrair features
        try:
            features = cv2_frame_to_features(frame)
            prediction, confidence = classify_ucf_frame(model, features)
        except Exception as e:
            st.error(f"Erro ao processar frame {frame_count}: {e}")
            prediction = "normal"
            confidence = 0.0
        
        # Atualizar buffer e contadores
        is_crime = prediction == "crime" and confidence >= confidence_threshold
        crime_buffer.append(is_crime)
        
        if is_crime:
            crime_count += 1
        else:
            normal_count += 1
        
        # Aplicar filtro temporal
        if len(crime_buffer) == temporal_window:
            crime_frames_in_window = sum(crime_buffer)
            
            if crime_frames_in_window >= min_crime_frames:
                # Evento detectado
                timestamp = datetime.now().isoformat()
                if frame_count not in logged_event_frames:
                    write_log(timestamp, video_path, "video", frame_count,
                             confidence, prediction)
                    events.append({
                        "idx": len(events),
                        "frame": frame_count,
                        "timestamp": timestamp,
                        "confidence": confidence,
                        "frames_in_window": crime_frames_in_window
                    })
                    logged_event_frames.add(frame_count)

                    if collect_crime_frames and len(crime_frames) < int(max_crime_frames):
                        display_frame = frame.copy()
                        if highlight_suspicious_area:
                            try:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                heatmap, bbox, _ = occlusion_suspicion_map(frame_rgb, model)
                                display_frame = draw_suspicion_overlay(frame, heatmap, bbox)
                            except Exception:
                                display_frame = frame.copy()

                        crime_frames.append({
                            "idx": len(crime_frames),
                            "frame": frame_count,
                            "tempo_s": round(frame_count / fps, 2) if fps else 0.0,
                            "confidence": round(float(confidence), 4),
                            "frames_in_window": int(crime_frames_in_window),
                            "image": cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                        })
        
        # Atualizar UI
        progress = frame_count / total_frames
        progress_placeholder.progress(progress, 
                                     f"Frame {frame_count}/{total_frames}")

        if live_preview and frame_count % preview_every == 0:
            preview_frame = frame.copy()
            color = (0, 0, 255) if is_crime else (0, 200, 0)
            label = f"{prediction.upper()} | conf={confidence:.3f}"

            if highlight_suspicious_area and is_crime:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    heatmap, bbox, baseline_proba = occlusion_suspicion_map(frame_rgb, model)
                    preview_frame = draw_suspicion_overlay(frame, heatmap, bbox)
                    cv2.putText(
                        preview_frame,
                        f"P(crime)={baseline_proba:.3f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                except Exception as exc:
                    st.warning(f"Nao foi possivel gerar destaque da area suspeita: {exc}")

            cv2.putText(
                preview_frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )
            preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                preview_frame,
                channels="RGB",
                use_container_width=True,
                caption=f"Frame {frame_count}"
            )
            status_placeholder.markdown(
                f"**Status atual:** `{prediction}` com confiança `{confidence:.4f}`"
            )
        
        crime_count_display.metric("Crime Frames", crime_count)
        normal_count_display.metric("Normal Frames", normal_count)
        confidence_display.metric("Last Confidence", f"{confidence:.4f}")
        
        frame_count += 1
    
    cap.release()
    return events, crime_frames


def main():
    """Interface principal Streamlit."""
    
    st.set_page_config(page_title="VanDetector UCF Crime", layout="wide")
    inject_icon_styles()
    icon_heading("security", "VanDetector - Detecção de Crimes (UCF Dataset)", "vd-icon-title")
    
    # Carregar modelos
    if not DEFAULT_MLP_MODEL.exists():
        st.error(f"Modelo MLP não encontrado: {DEFAULT_MLP_MODEL}")
        st.info("Execute `python train_ucf_mlp.py` primeiro")
        return
    
    model = load_mlp_model(str(DEFAULT_MLP_MODEL))
    scaler = load_scaler(str(DEFAULT_SCALER)) if DEFAULT_SCALER.exists() else None
    
    # Sidebar com controles
    with st.sidebar:
        icon_heading("tune", "Configuração")
        
        mode = st.radio("Modo:", ["Vídeo Local", "Imagem", "Webcam"])
        
        confidence_threshold = st.slider(
            "Confiança para Crime:",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_CRIME_CONFIDENCE,
            step=0.05
        )
        
        temporal_window = st.slider(
            "Janela Temporal (frames):",
            min_value=5,
            max_value=30,
            value=DEFAULT_TEMPORAL_WINDOW,
            step=1
        )
        
        min_crime_frames = st.slider(
            "Min Crime Frames na Janela:",
            min_value=1,
            max_value=temporal_window,
            value=DEFAULT_MIN_CRIME_FRAMES,
            step=1
        )

        st.divider()
        icon_heading("travel_explore", "Revisão")
        collect_crime_frames = st.checkbox("Coletar frames de crime", value=True)
        max_crime_frames = st.number_input(
            "Máximo de frames na navegação",
            min_value=10,
            max_value=500,
            value=80,
            step=10,
        )
    
    # Main content
    if mode == "Vídeo Local":
        icon_heading("videocam", "Processar Vídeo Local")
        
        uploaded_file = st.file_uploader("Selecione um vídeo", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Salvar arquivo temporário
            temp_video = Path(f"temp_{uploaded_file.name}")
            with open(temp_video, "wb") as f:
                f.write(uploaded_file.getbuffer())

            live_preview = st.checkbox("Mostrar análise em tempo real", value=True)
            preview_every = st.slider("Atualizar preview a cada N frames", 1, 10, 1)
            highlight_suspicious_area = st.checkbox(
                "Destacar área suspeita como no YOLO",
                value=True,
                help="Usa uma heurística por oclusão para aproximar onde está o comportamento suspeito."
            )
            
            if st.button("Analisar Vídeo"):
                events, crime_frames = process_video(
                    str(temp_video), model, scaler,
                    confidence_threshold=confidence_threshold,
                    temporal_window=temporal_window,
                    min_crime_frames=min_crime_frames,
                    live_preview=live_preview,
                    preview_every=preview_every,
                    highlight_suspicious_area=highlight_suspicious_area,
                    collect_crime_frames=collect_crime_frames,
                    max_crime_frames=max_crime_frames,
                )
                st.session_state["ucf_events"] = events
                st.session_state["ucf_crime_frames"] = crime_frames
                
                st.success(f"Análise concluída. {len(events)} eventos detectados.")
                
                if events:
                    icon_heading("notification_important", "Eventos Detectados")
                    for i, event in enumerate(events, 1):
                        st.write(f"**Evento {i}**: Frame {event['frame']}, "
                                f"Confiança: {event['confidence']:.4f}")
            
            # Limpar
            temp_video.unlink(missing_ok=True)
    
    elif mode == "Imagem":
        icon_heading("image_search", "Classificar Imagem")
        
        uploaded_file = st.file_uploader("Selecione uma imagem", 
                                        type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Imagem carregada")
            
            with col2:
                if st.button("Classificar"):
                    frame_array = np.array(image.convert("RGB"))
                    
                    features = frame_to_features(frame_array)
                    prediction, confidence = classify_ucf_frame(model, features)
                    
                    # Exibir resultado
                    color = "CRIME" if prediction == "crime" else "NORMAL"
                    st.markdown(f"## {color}")
                    st.metric("Confiança", f"{confidence:.4f}")

    crime_frames = st.session_state.get("ucf_crime_frames", [])
    if crime_frames:
        st.divider()
        icon_heading("browse_activity", "Navegação dos frames registrados como crime")

        nav_rows = [
            {
                "idx": item["idx"],
                "frame": item["frame"],
                "tempo_s": item["tempo_s"],
                "confidence": item["confidence"],
                "frames_in_window": item["frames_in_window"],
            }
            for item in crime_frames
        ]
        st.dataframe(nav_rows, use_container_width=True, hide_index=True)

        options = [item["idx"] for item in crime_frames]
        selected_idx = st.select_slider(
            "Frame para visualizar",
            options=options,
            value=options[0],
            format_func=lambda value: (
                f"Frame {crime_frames[value]['frame']} | "
                f"{crime_frames[value]['tempo_s']}s | "
                f"confiança {crime_frames[value]['confidence']:.0%}"
            ),
        )
        selected = crime_frames[selected_idx]
        st.image(
            selected["image"],
            caption=f"Frame {selected['frame']} - {selected['tempo_s']}s",
            use_container_width=True,
        )
    
    # Footer
    st.divider()
    st.caption("VanDetector - Detecção de Crimes usando UCF Crime Dataset")


if __name__ == "__main__":
    main()
