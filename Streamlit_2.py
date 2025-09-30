import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import ImageOps
import io


# =====================
# 1. Definición del modelo
# =====================
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(in_features, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(in_features, affine=True)
        )

    def forward(self, x):
        return x + self.block(x)

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x): return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, use_upsample=True):
        super().__init__()
        layers = []
        if use_upsample:
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
            ]
        else:
            layers += [nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)]

        layers += [nn.InstanceNorm2d(out_channels, affine=True), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_res=2):
        super().__init__()
        # Encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.res1 = nn.Sequential(*[ResidualBlock(64) for _ in range(n_res)])

        self.down2 = UNetDown(64, 128)
        self.res2 = nn.Sequential(*[ResidualBlock(128) for _ in range(n_res)])

        self.down3 = UNetDown(128, 256)
        self.res3 = nn.Sequential(*[ResidualBlock(256) for _ in range(n_res)])

        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.res4 = nn.Sequential(*[ResidualBlock(512) for _ in range(n_res)])

        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.res5 = nn.Sequential(*[ResidualBlock(512) for _ in range(n_res)])

        # Decoder
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 256, dropout=0.5)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        # Final
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, out_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.res1(self.down1(x))
        d2 = self.res2(self.down2(d1))
        d3 = self.res3(self.down3(d2))
        d4 = self.res4(self.down4(d3))
        d5 = self.res5(self.down5(d4))

        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.final(u4)


# =====================
# 2. Cargar modelo
# =====================
@st.cache_resource
def load_model(checkpoint_path):
    model = GeneratorResUNet()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    model.eval()
    return model


# =====================
# 3. Preprocesamiento y Postprocesamiento
# =====================
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return transform(image).unsqueeze(0)

def postprocess_tensor(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor * 0.5) + 0.5
    image = transforms.ToPILImage()(tensor)
    return image


# =====================
# 4. Interfaz Streamlit
# =====================
st.set_page_config(
    page_title="Restaurador de Huacos",  # Título de la pestaña
    page_icon="🏺",                      # Emoji o ruta a imagen (ej. "favicon.png")
    layout="centered"
)
st.markdown(
    """
    <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: #333333; margin-bottom: 10px;">
            🏺 Simulador de restauración digital de huacos peruanos
        </h1>
        <p style="color: #555555; font-size: 18px; margin: 0;">
            Sube una imagen de un huaco deteriorado o desgastado para restaurarlo.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


model = load_model("generator_checkpoint.pth")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Huaco deteriorado", width=400)

    st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: white;
        color: black;
        border: 1px solid black;
    }
    div.stButton > button:first-child:hover {
        background-color: #f0f0f0;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
    )


    if st.button("Restaurar Imagen"):
        progress_bar = st.progress(0)
        with st.spinner("Procesando con el modelo GAN..."):
            input_tensor = preprocess_image(input_image)
            progress_bar.progress(25)

            with torch.no_grad():
                output_tensor = model(input_tensor)
            progress_bar.progress(75)

            restored_image = postprocess_tensor(output_tensor)
            progress_bar.progress(100)

        
        def pad_to_square(img, size=(400, 400), color=(255, 255, 255)):
            
            return ImageOps.pad(img, size, color=color)

        target_size = (400, 400) 
        img_original_resized = ImageOps.contain(input_image, target_size)
        img_restored_resized = ImageOps.contain(restored_image, target_size)

        # Mostrar imágenes lado a lado
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_original_resized, caption="Original", width=400)
        with col2:
            st.image(img_restored_resized, caption="Restaurado", width=400)

        buf = io.BytesIO()
        restored_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # CSS para centrar y cambiar estilo del botón
        st.markdown(
            """
            <style>
            div.stDownloadButton {text-align: center;}
            div.stDownloadButton > button:first-child {
                background-color: white;
                color: black;
                border: 1px solid black;
            }
            div.stDownloadButton > button:first-child:hover {
                background-color: #f0f0f0;
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Botón de descarga centrado
        st.download_button(
            label="Descargar imagen restaurada",
            data=byte_im,
            file_name="huaco_restaurado.png",
            mime="image/png",
            key="download_btn"
        ) 

        # Calcular métricas
        img1 = np.array(input_image.resize((512, 512)))
        img2 = np.array(restored_image.resize((512, 512)))
        psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
        ssim = structural_similarity(img1, img2, channel_axis=2)


        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px;">
                <div style="font-size: 34px; font-weight: bold; margin-bottom: 15px;">
                    📊 Métricas de Restauración
                </div>
                <div style="font-size: 24px;">
                    <b>PSNR:</b> {psnr:.2f} dB
                </div>
                <div style="font-size: 24px;">
                    <b>SSIM:</b> {ssim:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <hr style="margin-top:50px; margin-bottom:10px;">

            <div style="text-align: center; color: gray; font-size: 14px;">
                🚧 Esta aplicación sigue en desarrollo.<br>
                Desarrollada por un estudiante de la Universidad de Lima como parte de su trabajo final de investigación.<br>
                Puede contener errores.
            </div>
            """,
            unsafe_allow_html=True
        )
