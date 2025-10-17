import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from collections import OrderedDict
import time
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ==============================================================================
# 1. DEFINICIÓN DEL MODELO GENERADOR (Sin cambios)
# ==============================================================================


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        channels = 3
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]
        for _ in range(2):
            out_features //= 2
            model += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ==============================================================================
# 2. FUNCIONES AUXILIARES Y CARGA DE MODELOS (Sin cambios en la carga)
# ==============================================================================


@st.cache_resource
def load_all_models():
    """
    Carga el generador y el modelo LPIPS, los mantiene en caché.
    """
    MODEL_PATH = "generator_checkpoint.pth"

    generator_model = GeneratorResNet()
    checkpoint = torch.load(
        MODEL_PATH, map_location=torch.device("cpu"), weights_only=False
    )
    generator_weights = checkpoint["netG"]
    new_state_dict = OrderedDict()
    for k, v in generator_weights.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    generator_model.load_state_dict(new_state_dict)
    generator_model.eval()

    lpips_model = lpips.LPIPS(net="alex")
    lpips_model.eval()

    return generator_model, lpips_model


def preprocess_image(image: Image.Image, size=(512, 512)):
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return transform(image).unsqueeze(0)


def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor * 0.5) + 0.5
    return transforms.ToPILImage()(tensor)


# ==============================================================================
# 3. CONFIGURACIÓN DE LA PÁGINA E INTERFAZ (ACTUALIZADO)
# ==============================================================================
st.set_page_config(
    page_title="Proyecto Huacos - Restaurador", page_icon="🏺", layout="centered"
)

st.markdown(
    """
<style>
.title-box {
    background-color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}
</style>
<div class="title-box">
    <h1 style="color: #333333; margin-bottom: 10px;">🏺 Simulador de restauración digital de huacos peruanos </h1>
    <p style="color: #555555; font-size: 18px; margin: 0;">Sube una imagen de un huaco deteriorado o desgastado para restaurarlo.</p>
</div>
""",
    unsafe_allow_html=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    with st.spinner(
        "Cargando modelos de IA, por favor espere... (Esto puede tardar en el primer inicio)"
    ):
        model, lpips_model = load_all_models()
        model.to(device)
        lpips_model.to(device)
except Exception as e:
    st.error(f"Error fatal al cargar los modelos: {e}")
    st.exception(e)
    st.stop()

uploaded_file = st.file_uploader(
    "Cargue la imagen de un huaco para su restauración:",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    progress_bar = st.progress(0, text="Cargando imagen...")
    input_image = Image.open(uploaded_file).convert("RGB")
    progress_bar.progress(100, text="Imagen cargada.")
    time.sleep(1)
    progress_bar.empty()

    # <<< CORRECCIÓN DE SINTAXIS >>>
    st.image(input_image, caption="Imagen Original", width="stretch")

    # <<< CORRECCIÓN DE SINTAXIS >>>
    if st.button("✨ Restaurar Huaco", width="stretch"):
        st.session_state.restored_image = None
        restore_progress = st.progress(0, text="Iniciando restauración...")
        with torch.no_grad():
            restore_progress.progress(25, text="Pre-procesando imagen...")
            input_tensor = preprocess_image(input_image).to(device)
            restore_progress.progress(50, text="Aplicando modelo CycleGAN...")
            output_tensor = model(input_tensor)
            restore_progress.progress(75, text="Post-procesando resultado...")
            restored_image_pil = tensor_to_pil(output_tensor)
            st.session_state.restored_image = restored_image_pil
            restore_progress.progress(100, text="¡Restauración completa!")
            time.sleep(1)
            restore_progress.empty()

    if (
        "restored_image" in st.session_state
        and st.session_state.restored_image is not None
    ):
        # <<< CORRECCIÓN DE SINTAXIS >>>
        st.image(
            st.session_state.restored_image,
            caption="Imagen Restaurada",
            width="stretch",
        )
        buf = io.BytesIO()
        st.session_state.restored_image.save(buf, format="PNG")
        # <<< CORRECCIÓN DE SINTAXIS >>>
        st.download_button(
            label="📥 Descargar Imagen Restaurada",
            data=buf.getvalue(),
            file_name="huaco_restaurado.png",
            mime="image/png",
            width="stretch",
        )

        st.markdown("---")
        st.subheader("📊 Análisis Cuantitativo de la Transformación")

        # <<< CORRECCIÓN DE SINTAXIS >>>
        if st.button("Calcular Análisis de la Transformación", width="stretch"):
            try:
                with st.spinner(
                    "Analizando... Este proceso puede tardar unos segundos."
                ):
                    original_resized = input_image.resize((512, 512))
                    original_array = np.array(original_resized)
                    restored_array = np.array(st.session_state.restored_image)

                    ssim_val = ssim(
                        original_array, restored_array, data_range=255, channel_axis=2
                    )
                    psnr_val = psnr(original_array, restored_array, data_range=255)

                    original_tensor = lpips.im2tensor(original_array).to(device)
                    restored_tensor = lpips.im2tensor(restored_array).to(device)
                    lpips_val = lpips_model(original_tensor, restored_tensor).item()

                st.success("Análisis completado.")

                col1, col2, col3 = st.columns(3)
                col1.metric(label="SSIM", value=f"{ssim_val:.4f}")
                col2.metric(label="PSNR", value=f"{psnr_val:.2f} dB")
                col3.metric(label="LPIPS", value=f"{lpips_val:.4f}")

                with st.expander("📝 ¿Cómo interpretar este análisis?"):
                    st.info(
                        """
                        Estas métricas **no miden la calidad** de la restauración, sino la **magnitud de la transformación** aplicada por el modelo. Un cambio más grande (valores más bajos de SSIM/PSNR y más altos de LPIPS) indica una intervención más profunda del modelo.
                        """
                    )
                    st.markdown(
                        """
                        #### **SSIM (Índice de Similitud Estructural)**
                        - **Qué mide:** Compara la estructura, el contraste y la luminancia. Su rango es de -1 a 1.
                        - **Interpretación aquí:** Un valor de **1** significa que las imágenes son idénticas. Un valor **más bajo** indica una alteración significativa de la textura y apariencia.

                        ---
                        #### **PSNR (Relación Señal-Ruido Pico)**
                        - **Qué mide:** La diferencia a nivel de píxeles. Se mide en decibelios (dB).
                        - **Interpretación aquí:** Un valor **más bajo** sugiere cambios más profundos en los colores y detalles. Un valor muy alto (ej. > 40 dB) indicaría una transformación mínima.
                        
                        ---
                        #### **LPIPS (Distancia Perceptual)**
                        - **Qué mide:** Usa una red neuronal para imitar qué tan diferentes percibe un humano dos imágenes.
                        - **Interpretación aquí:** Un valor **más alto** (ej. > 0.4) indica cambios notorios y perceptibles. Un valor cercano a **0** significaría que son casi idénticas a la vista.
                        """
                    )
            except Exception as e:
                st.error(f"No se pudo completar el análisis cuantitativo: {e}")

st.markdown(
    """
    <hr style="margin-top:50px; margin-bottom:10px;">
    <div style="text-align: center; color: gray; font-size: 14px;">
        🚧 Esta aplicación sigue en desarrollo.<br>
        Desarrollada como parte de un trabajo final de investigación.<br>
        Puede contener errores.
    </div>
    """,
    unsafe_allow_html=True,
)
