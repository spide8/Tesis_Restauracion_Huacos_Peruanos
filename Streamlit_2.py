import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from collections import OrderedDict
import time
import lpips
from pytorch_fid import fid_score
import os
import numpy as np
import shutil  # Importado para limpiar archivos temporales

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
# 2. FUNCIONES AUXILIARES Y CARGA DE MODELOS (MODIFICADO)
# ==============================================================================


# Usamos @st.cache_resource para cargar TODOS los modelos pesados una sola vez.
# Esta función ahora devuelve ambos modelos para una carga centralizada.
@st.cache_resource
def load_all_models():
    """
    Carga el generador y el modelo LPIPS, los mantiene en caché para evitar
    descargas y cargas repetidas en cada ejecución.
    """
    # Cargar el modelo generador principal
    generator_model = GeneratorResNet()
    checkpoint = torch.load(
        "generator_checkpoint.pth", map_location=torch.device("cpu"), weights_only=False
    )
    generator_weights = checkpoint["netG"]
    new_state_dict = OrderedDict()
    for k, v in generator_weights.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    generator_model.load_state_dict(new_state_dict)
    generator_model.eval()

    # Cargar el modelo para la métrica LPIPS
    # Nota: La advertencia sobre 'pretrained' que puedes ver en los logs
    # proviene de esta librería (lpips), no de tu código. Es seguro ignorarla.
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


def calculate_metrics(original_img_pil, restored_img_pil, lpips_model, device):
    """
    Calcula las métricas FID y LPIPS.
    Maneja la creación y eliminación de directorios temporales de forma segura.
    """
    temp_dir = "temp_for_metrics"
    original_dir = os.path.join(temp_dir, "original")
    restored_dir = os.path.join(temp_dir, "restored")

    # Crear directorios temporales
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(restored_dir, exist_ok=True)

    # Guardar imágenes
    original_img_pil.save(os.path.join(original_dir, "img.png"))
    restored_img_pil.save(os.path.join(restored_dir, "img.png"))

    fid_value = "N/A"
    try:
        # La métrica FID está diseñada para comparar distribuciones de MUCHAS imágenes.
        # Calcularla con una sola imagen por carpeta arrojará una advertencia
        # y el resultado (infinito) no es matemáticamente significativo.
        # Lo calculamos de todos modos, pero lo manejaremos con cuidado.
        fid_value = fid_score.calculate_fid_given_paths(
            [original_dir, restored_dir], batch_size=1, device=device, dims=2048
        )
        # Si el valor es infinito, lo mostramos como tal.
        if np.isinf(fid_value):
            fid_value = "∞ (No significativo)"
        else:
            fid_value = f"{fid_value:.2f}"

    except Exception:
        # Si hay un error, simplemente mostramos N/A
        pass

    # Calcular LPIPS
    original_tensor = lpips.im2tensor(np.array(original_img_pil)).to(device)
    restored_tensor = lpips.im2tensor(np.array(restored_img_pil)).to(device)
    lpips_value = lpips_model(original_tensor, restored_tensor).item()

    # Limpiar archivos y directorios temporales
    shutil.rmtree(temp_dir)

    return fid_value, lpips_value


# ==============================================================================
# 3. CONFIGURACIÓN DE LA PÁGINA E INTERFAZ (MODIFICADO)
# ==============================================================================
st.set_page_config(
    page_title="Proyecto Huacos - Restaurador", page_icon="🏺", layout="centered"
)

# --- Título con Recuadro Blanco ---
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

# --- Carga de Modelos ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    with st.spinner(
        "Cargando modelos de IA, por favor espere... (Esto puede tardar en el primer inicio)"
    ):
        # Llamamos a nuestra función cacheada una sola vez
        model, lpips_model = load_all_models()
        model.to(device)
        lpips_model.to(device)
except Exception as e:
    st.error(f"Error fatal al cargar los modelos: {e}")
    st.exception(e)  # Muestra el stack trace completo para depuración
    st.stop()

# --- Carga de Archivo ---
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

    # CORRECCIÓN: Se reemplazó use_container_width=True por width='stretch'
    st.image(input_image, caption="Imagen Original", width="stretch")

    if st.button("✨ Restaurar Huaco", use_container_width=True):
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
        # CORRECCIÓN: Se reemplazó use_container_width=True por width='stretch'
        st.image(
            st.session_state.restored_image,
            caption="Imagen Restaurada",
            width="stretch",
        )
        buf = io.BytesIO()
        st.session_state.restored_image.save(buf, format="PNG")
        st.download_button(
            label="📥 Descargar Imagen Restaurada",
            data=buf.getvalue(),
            file_name="huaco_restaurado.png",
            mime="image/png",
            use_container_width=True,
        )
        st.markdown("---")
        st.subheader("📊 Evaluación Cuantitativa del Cambio")

        try:
            with st.spinner("Calculando métricas de evaluación..."):
                fid_val, lpips_val = calculate_metrics(
                    input_image.resize((512, 512)),
                    st.session_state.restored_image,
                    lpips_model,
                    device,
                )
            col1, col2 = st.columns(2)
            col1.metric(label="FID (Fréchet Inception Distance)", value=fid_val)
            col2.metric(label="LPIPS (Distancia Perceptual)", value=f"{lpips_val:.4f}")

            # --- Recuadro de Métricas Actualizado con explicación sobre FID ---
            with st.expander("📝 ¿Cómo interpretar estas métricas?"):
                st.info(
                    """
                    Estas métricas miden la **magnitud del cambio** entre la imagen original y la restaurada.
                    
                    **FID (Fréchet Inception Distance):**
                    - **¿Qué es?** Mide la diferencia entre las características de dos **grupos** de imágenes. **Menor es mejor**.
                    - **Interpretación aquí:** El cálculo de FID con una sola imagen no es matemáticamente robusto (por eso puede dar `∞`). Sin embargo, un valor numérico alto sugiere que el modelo realizó cambios significativos y distinguibles.
                    
                    **LPIPS (Learned Perceptual Image Patch Similarity):**
                    - **¿Qué es?** Mide qué tan diferentes se ven dos imágenes para un humano. **Menor es mejor (más similar)**.
                    - **Interpretación aquí:** Un LPIPS más alto (ej. > 0.4) indica que los cambios son notorios. Un valor bajo (ej. < 0.1) significa que las imágenes son casi idénticas.
                    
                    **En resumen:** No buscamos valores de cero. Valores más altos reflejan una transformación más profunda por parte del modelo.
                    """
                )
        except Exception as e:
            st.warning(f"No se pudieron calcular las métricas: {e}")

# --- Footer Actualizado ---
st.markdown(
    """
    <hr style="margin-top:50px; margin-bottom:10px;">
    <div style="text-align: center; color: gray; font-size: 14px;">
        🚧 Esta aplicación sigue en desarrollo.<br>
        Desarrollada por un estudiante de la Universidad de Lima como parte de su trabajo final de investigación.<br>
        Puede contener errores.
    </div>
    """,
    unsafe_allow_html=True,
)
