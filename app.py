"""
============================================================================
 Segmentation de tumeurs cérébrales 3D — Application Streamlit
============================================================================
 - Choix du modèle : U-Net 3D  ou  HMSA-UNet3D
 - Upload d'un patient (4 modalités .nii.gz + masque seg optionnel)
 - Affichage : 4 modalités + vérité terrain + prédiction (classes colorées)
 - Curseur de coupe, métriques Dice par classe, téléchargement du masque prédit

 Lancement :  streamlit run app.py
============================================================================
"""

import os
import io
import tempfile
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import streamlit as st

# Rendu 3D (optionnel) : plotly + scikit-image
try:
    import plotly.graph_objects as go
    from skimage import measure
    PLOTLY_3D = True
except Exception:
    PLOTLY_3D = False

# ---------------------------------------------------------------------------
# Constantes (doivent correspondre à l'entraînement)
# ---------------------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = (128, 128, 128)
MODALITIES  = ["flair", "t1", "t1ce", "t2"]
REGIONS     = ["WT", "TC", "ET"]
CLASS_CMAP  = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.6), (0, 1, 0, 0.6), (1, 1, 0, 0.6)])
CLASS_LEGEND = {"Core / nécrose": "red", "Œdème": "limegreen", "Enhancing": "yellow"}

# Type de sortie de chaque modèle :
#   "sigmoid3" -> 3 canaux WT/TC/ET (seuil 0.5)   |  "softmax4" -> 4 classes (argmax)
MODEL_OUTPUT = {"U-Net 3D": "sigmoid3", "HMSA-UNet3D": "softmax4"}

# Type de prétraitement de chaque modèle (DOIT correspondre à l'entraînement) :
#   "resize" -> interpolation trilinéaire en 128^3   |   "crop" -> crop/pad centré 128^3
MODEL_PREPROC = {"U-Net 3D": "resize", "HMSA-UNet3D": "crop"}

# Hyperparamètres du HMSA — DOIVENT être identiques à l'entraînement (sinon poids incompatibles)
HMSA_BASE      = 32
HMSA_NUM_HEADS = 4
HMSA_CLASSES   = 4

# ===========================================================================
# 1) ARCHITECTURES
# ===========================================================================

# ---- U-Net 3D (identique au notebook d'entraînement) ----------------------
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout), nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x): return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, f=16):
        super().__init__()
        self.e1 = DoubleConv(in_ch, f)
        self.e2 = DoubleConv(f, f * 2)
        self.e3 = DoubleConv(f * 2, f * 4)
        self.e4 = DoubleConv(f * 4, f * 8)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(f * 8, f * 16)
        self.u4 = nn.ConvTranspose3d(f * 16, f * 8, 2, stride=2); self.d4 = DoubleConv(f * 16, f * 8)
        self.u3 = nn.ConvTranspose3d(f * 8, f * 4, 2, stride=2);  self.d3 = DoubleConv(f * 8, f * 4)
        self.u2 = nn.ConvTranspose3d(f * 4, f * 2, 2, stride=2);  self.d2 = DoubleConv(f * 4, f * 2)
        self.u1 = nn.ConvTranspose3d(f * 2, f, 2, stride=2);      self.d1 = DoubleConv(f * 2, f)
        self.out = nn.Conv3d(f, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(b),  e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


# ===========================================================================
#  CODE HMSA-UNet3D (intégré depuis le notebook d'entraînement)
# ===========================================================================
class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d(1); self.mx = nn.AdaptiveMaxPool3d(1)
        h = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv3d(channels, h, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv3d(h, channels, 1, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.mlp(self.avg(x)) + self.mlp(self.mx(x)))


class SpatialAttention3D(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, k, padding=k // 2, bias=False); self.sig = nn.Sigmoid()
    def forward(self, x):
        a = torch.mean(x, 1, keepdim=True); m, _ = torch.max(x, 1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([a, m], 1)))


class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, k=7):
        super().__init__(); self.ca = ChannelAttention3D(channels, reduction); self.sa = SpatialAttention3D(k)
    def forward(self, x): return self.sa(self.ca(x))


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, use_cbam=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False); self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True); self.use_cbam = use_cbam
        if use_cbam: self.cbam = CBAM3D(out_ch)
        self.res = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        idt = self.res(x)
        o = self.relu(self.bn1(self.conv1(x))); o = self.bn2(self.conv2(o))
        if self.use_cbam: o = self.cbam(o)
        return self.relu(o + idt)


class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1), nn.BatchNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1), nn.BatchNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1), nn.BatchNorm3d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class SelfAttention3D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__(); assert channels % num_heads == 0
        self.h = num_heads; self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False); self.proj = nn.Conv3d(channels, channels, 1)
    def forward(self, x):
        B, C, D, H, W = x.shape; n = D * H * W; ch = C // self.h
        qkv = self.qkv(self.norm(x)).view(B, 3, self.h, ch, n)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        att = (torch.matmul(q.transpose(-2, -1), k) / (ch ** 0.5)).softmax(-1)
        out = torch.matmul(v, att.transpose(-2, -1)).reshape(B, C, D, H, W)
        return x + self.proj(out)


class HMSA_UNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base=32, deep_supervision=True, num_heads=4):
        super().__init__(); self.deep_supervision = deep_supervision
        c1, c2, c3, c4, c5 = base, base*2, base*4, base*8, base*16
        self.enc1 = ConvBlock3D(in_channels, c1); self.enc2 = ConvBlock3D(c1, c2)
        self.enc3 = ConvBlock3D(c2, c3);          self.enc4 = ConvBlock3D(c3, c4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(c4, c5); self.self_attn = SelfAttention3D(c5, num_heads)
        self.up4 = nn.ConvTranspose3d(c5, c4, 2, 2); self.ag4 = AttentionGate3D(c4, c4, c4//2); self.dec4 = ConvBlock3D(c4*2, c4)
        self.up3 = nn.ConvTranspose3d(c4, c3, 2, 2); self.ag3 = AttentionGate3D(c3, c3, c3//2); self.dec3 = ConvBlock3D(c3*2, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, 2); self.ag2 = AttentionGate3D(c2, c2, c2//2); self.dec2 = ConvBlock3D(c2*2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, 2); self.ag1 = AttentionGate3D(c1, c1, c1//2); self.dec1 = ConvBlock3D(c1*2, c1)
        self.out_conv = nn.Conv3d(c1, num_classes, 1)
        if deep_supervision:
            self.aux2 = nn.Conv3d(c2, num_classes, 1); self.aux3 = nn.Conv3d(c3, num_classes, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.self_attn(self.bottleneck(self.pool(e4)))
        d4 = self.up4(b);  d4 = self.dec4(torch.cat([self.ag4(d4, e4), d4], 1))
        d3 = self.up3(d4); d3 = self.dec3(torch.cat([self.ag3(d3, e3), d3], 1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([self.ag2(d2, e2), d2], 1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([self.ag1(d1, e1), d1], 1))
        out = self.out_conv(d1)
        if self.deep_supervision and self.training:
            return out, self.aux2(d2), self.aux3(d3)
        return out

HMSA_READY = True   # code HMSA intégré
# ===========================================================================
#  FIN DE LA ZONE HMSA-UNet3D
# ===========================================================================


def build_model(name):
    """Instancie l'architecture choisie."""
    if name == "U-Net 3D":
        return UNet3D(in_ch=4, out_ch=3, f=16)
    if name == "HMSA-UNet3D":
        if not HMSA_READY:
            raise RuntimeError(
                "Le code HMSA-UNet3D n'est pas encore intégré. Colle ConvBlock3D, "
                "SelfAttention3D, AttentionGate3D et HMSA_UNet3D dans app.py, puis HMSA_READY = True."
            )
        # mêmes arguments qu'à l'entraînement
        return HMSA_UNet3D(in_channels=4, num_classes=HMSA_CLASSES,   # noqa: F821
                           base=HMSA_BASE, deep_supervision=True, num_heads=HMSA_NUM_HEADS)
    raise ValueError(name)


@st.cache_resource(show_spinner=False)
def load_model(name, weights_bytes):
    """Charge le modèle + poids (mis en cache pour ne pas recharger à chaque interaction)."""
    model = build_model(name).to(DEVICE)
    state = torch.load(io.BytesIO(weights_bytes), map_location=DEVICE)
    # Le checkpoint peut être un dict complet (epoch, history, cfg...) avec les poids
    # rangés sous une clé. On extrait les poids quelle que soit la convention.
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    model.load_state_dict(state)
    model.eval()
    return model


# ===========================================================================
# 2) PRÉTRAITEMENT / POST-TRAITEMENT (identiques au notebook)
# ===========================================================================
def zscore_per_modality(vol):
    m = vol > 0
    if m.sum() == 0:
        return vol
    mu, sd = vol[m].mean(), vol[m].std() + 1e-8
    out = (vol - mu) / sd
    out[~m] = 0.0
    return out


def make_regions(seg):
    return np.stack([(seg > 0), (seg == 1) | (seg == 4), (seg == 4)]).astype(np.float32)


def znorm_stack(image):
    """z-score par modalité sur les voxels non nuls. image : (C,H,W,D)."""
    out = np.zeros_like(image, np.float32)
    for c in range(image.shape[0]):
        v = image[c]; m = v > 0
        if m.sum() > 0:
            out[c] = np.where(m, (v - v[m].mean()) / (v[m].std() + 1e-8), 0.0)
    return out


def crop_or_pad_center(image, size):
    """Crop/pad centré (comme l'entraînement HMSA). image : (C,H,W,D). Renvoie (out, info)."""
    _, H, W, D = image.shape; th, tw, td = size
    def fx(L, T):
        if L >= T:
            s = (L - T) // 2; return slice(s, s + T), 0, 0          # crop
        b = (T - L) // 2; return slice(0, L), b, T - L - b          # pad
    sh, pbh, pah = fx(H, th); sw, pbw, paw = fx(W, tw); sd, pbd, pad_ = fx(D, td)
    out = image[:, sh, sw, sd]
    out = np.pad(out, ((0, 0), (pbh, pah), (pbw, paw), (pbd, pad_)))
    info = {"orig": (H, W, D), "sl": (sh, sw, sd), "pad": ((pbh, pah), (pbw, paw), (pbd, pad_))}
    return out.astype(np.float32), info


def invert_crop_or_pad(vol, info):
    """Replace un volume 128^3 dans l'espace d'origine (inverse de crop_or_pad_center)."""
    (pbh, pah), (pbw, paw), (pbd, pad_) = info["pad"]
    th, tw, td = vol.shape
    core = vol[pbh:th - pah if pah else th,
               pbw:tw - paw if paw else tw,
               pbd:td - pad_ if pad_ else td]
    full = np.zeros(info["orig"], dtype=vol.dtype)
    sh, sw, sd = info["sl"]
    full[sh, sw, sd] = core
    return full


def regions_to_label(mask3):
    wt, tc, et = mask3[0] > 0.5, mask3[1] > 0.5, mask3[2] > 0.5
    lab = np.zeros_like(mask3[0])
    lab[wt] = 2; lab[tc] = 1; lab[et] = 3      # œdème / core / enhancing
    return lab


def central_tumor_slice(label3d):
    z = np.where((label3d > 0).sum(axis=(0, 1)) > 0)[0]
    return int((z.min() + z.max()) // 2) if len(z) else IMG_SIZE[2] // 2


def tumor_mesh(binary_vol, color, name, opacity=0.5):
    """Surface 3D (Mesh3d) d'un masque binaire via marching cubes. None si trop petit."""
    if not PLOTLY_3D or binary_vol.sum() < 30:
        return None
    verts, faces, _, _ = measure.marching_cubes(binary_vol.astype(np.float32),
                                                level=0.5, step_size=2)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True, flatshading=True,
    )


def save_nii_from_file(uploaded):
    """Écrit un fichier uploadé sur disque temporaire et renvoie le chemin."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
    tmp.write(uploaded.getbuffer()); tmp.close()
    return tmp.name


def detect_modality(filename):
    name = filename.lower().replace(".nii.gz", "").replace(".nii", "")
    return name.split("_")[-1]


@torch.no_grad()
def run_inference(model, model_name, image_raw, seg_raw=None):
    """Prétraitement propre à chaque modèle, inférence, post-traitement.
       Renvoie (regions3, label3d, img128, gt3, restore)
         regions3 : (3,128^3) binaire WT/TC/ET   | label3d : (128^3) {0,1,2,3}
         img128   : image fournie au modèle       | gt3 : (3,128^3) ou None
         restore  : info pour replacer la prédiction dans l'espace d'origine."""
    if MODEL_PREPROC[model_name] == "resize":          # U-Net 3D : z-norm puis interpolation
        img = znorm_stack(image_raw)
        img_t = F.interpolate(torch.from_numpy(img)[None].float(), size=IMG_SIZE,
                              mode="trilinear", align_corners=False)
        restore = ("resize", None)
        gt3 = None
        if seg_raw is not None:
            g = make_regions(seg_raw)
            gt3 = F.interpolate(torch.from_numpy(g)[None], size=IMG_SIZE, mode="nearest")[0].numpy()
    else:                                              # HMSA : crop/pad centré puis z-norm
        proc, info = crop_or_pad_center(image_raw, IMG_SIZE)
        proc = znorm_stack(proc)
        img_t = torch.from_numpy(proc)[None].float()
        restore = ("crop", info)
        gt3 = None
        if seg_raw is not None:
            seg_c, _ = crop_or_pad_center(seg_raw[None].astype(np.float32), IMG_SIZE)
            gt3 = make_regions(seg_c[0])

    out = model(img_t.to(DEVICE))
    if isinstance(out, (tuple, list)):                 # deep supervision -> sortie principale
        out = out[0]

    if MODEL_OUTPUT[model_name] == "sigmoid3":
        regions = (torch.sigmoid(out)[0] > 0.5).float().cpu().numpy()     # (3,128^3)
        label = regions_to_label(regions)
    else:  # "softmax4" : argmax -> {0=fond,1=nécrose,2=œdème,3=enhancing}
        label = torch.argmax(out[0], dim=0).cpu().numpy().astype(np.float32)
        wt, tc, et = label > 0, (label == 1) | (label == 3), label == 3
        regions = np.stack([wt, tc, et]).astype(np.float32)
    return regions, label, img_t[0].cpu().numpy(), gt3, restore


def dice_per_class(pred3, gt3, eps=1e-6):
    out = {}
    for i, r in enumerate(REGIONS):
        p, g = pred3[i] > 0.5, gt3[i] > 0.5
        inter = (p & g).sum()
        out[r] = (2 * inter + eps) / (p.sum() + g.sum() + eps)
    return out


# ===========================================================================
# 3) INTERFACE
# ===========================================================================
st.set_page_config(page_title="Segmentation tumeurs cérébrales 3D",
                   page_icon="🧠", layout="wide")

st.markdown("""
<style>
/* ===== Base ===== */
.block-container {padding-top: 1.2rem; max-width: 1320px;}
.stApp {background: linear-gradient(180deg,#f2f6fc 0%, #f8fafc 100%);}
html, body, [class*="css"] {font-family: 'Segoe UI', system-ui, sans-serif;}

/* ===== En-tête / logo (clair) ===== */
.app-header {
  display:flex; align-items:center; justify-content:space-between; gap:1rem;
  background: linear-gradient(135deg,#eef5ff 0%, #f3f0ff 50%, #e8fbff 100%);
  border:1px solid #e3e9f5;
  padding: 1.6rem 2.1rem; border-radius: 20px; color:#0f172a; margin-bottom: 1.4rem;
  box-shadow: 0 12px 30px rgba(59,130,246,.12);
}
.brand {display:flex; align-items:center; gap:1.1rem;}
.brand .logo {
  width:78px; height:78px; border-radius:20px; flex:0 0 auto;
  background:#ffffff; border:1px solid #e3e9f5;
  display:flex; align-items:center; justify-content:center;
  box-shadow: 0 6px 16px rgba(59,130,246,.16);
}
.brand h1 {color:#0f2a5e; margin:0; font-size:2rem; font-weight:800; letter-spacing:.4px;}
.brand p {margin:.25rem 0 0; color:#5b6b86; font-size:.95rem; font-weight:400;}
.htags {display:flex; flex-direction:column; gap:.4rem; align-items:flex-end;}
.htag {background:#eef5ff; border:1px solid #cfe0ff; color:#2563eb;
       padding:4px 12px; border-radius:20px; font-size:.78rem; font-weight:700;}

/* ===== Sidebar (claire) ===== */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#ffffff 0%, #f5f9ff 100%);
  border-right:1px solid #e6ecf6;
}
[data-testid="stSidebar"] * {color:#334155;}
.side-brand {display:flex; align-items:center; gap:.6rem; padding:.2rem 0 1rem;
             border-bottom:1px solid #e6ecf6; margin-bottom:1rem;}
.side-brand .dot {width:34px;height:34px;border-radius:10px;
  background:linear-gradient(135deg,#60a5fa,#3b82f6);display:flex;align-items:center;justify-content:center;}
.side-brand span {font-weight:700; font-size:1.05rem; color:#0f2a5e;}
.side-label {text-transform:uppercase; letter-spacing:1.2px; font-size:.72rem;
  color:#2563eb; font-weight:700; margin:1.1rem 0 .4rem;}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
[data-testid="stSidebar"] .stTextInput input {
  background:#ffffff; border:1px solid #dbe4f3; border-radius:10px; color:#334155;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background:#f4f8ff; border:1px dashed #c7d7f5; border-radius:12px; padding:.4rem;
}
.badge {display:inline-flex; align-items:center; gap:.5rem; background:#f1f6ff;
  border:1px solid #d6e2f8; border-radius:20px; padding:6px 14px; font-size:.85rem; font-weight:600; color:#334155;}
.badge .live {width:9px;height:9px;border-radius:50%;}

/* ===== Cartes / titres ===== */
.section-title {font-weight:700; color:#0f2a5e; font-size:1.15rem; margin:.4rem 0 .7rem;
  display:flex; align-items:center; gap:.5rem;}
.section-title:before {content:""; width:5px; height:20px; border-radius:4px;
  background:linear-gradient(180deg,#60a5fa,#3b82f6); display:inline-block;}

/* ===== Métriques ===== */
[data-testid="stMetric"] {
  background:#fff; border:1px solid #e2e8f0; border-radius:16px; padding:14px 18px;
  box-shadow:0 2px 10px rgba(15,23,42,.05);
}
[data-testid="stMetricValue"] {color:#2563eb; font-weight:800;}

/* ===== Boutons / onglets ===== */
.stButton>button {border-radius:12px; font-weight:700; padding:.55rem 1.3rem; border:1px solid #e2e8f0;}
.stButton>button[kind="primary"] {background:linear-gradient(135deg,#60a5fa,#2563eb); border:none; color:#fff;}
.stTabs [data-baseweb="tab-list"] {gap:.4rem;}
.stTabs [data-baseweb="tab"] {font-weight:700; border-radius:10px 10px 0 0;}
.stTabs [aria-selected="true"] {color:#2563eb;}

/* ===== Légende ===== */
.legend-pill {display:inline-block; padding:4px 14px; border-radius:16px;
  color:#fff; font-size:.8rem; margin-right:6px; font-weight:700;}
</style>
""", unsafe_allow_html=True)

# Logo SVG : cerveau avec une tumeur (nodule rouge)
BRAIN_SVG = """
<svg width="54" height="54" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M32 11c-4.4 0-8 2.4-9.1 5.8-3.6.2-6.6 2.8-6.6 6.7 0 1.7.5 3.2 1.5 4.4-1.7 1.2-2.8 3.1-2.8 5.3
           0 2.6 1.4 4.7 3.4 5.8-.2.8-.3 1.5-.3 2.3 0 4.2 3.2 7.4 7.4 7.4 1 3.1 3.8 5.1 7.1 5.1"
        fill="#dbeafe" stroke="#3b82f6" stroke-width="1.6" stroke-linejoin="round"/>
  <path d="M32 11c4.4 0 8 2.4 9.1 5.8 3.6.2 6.6 2.8 6.6 6.7 0 1.7-.5 3.2-1.5 4.4 1.7 1.2 2.8 3.1 2.8 5.3
           0 2.6-1.4 4.7-3.4 5.8.2.8.3 1.5.3 2.3 0 4.2-3.2 7.4-7.4 7.4-1 3.1-3.8 5.1-7.1 5.1"
        fill="#eff6ff" stroke="#3b82f6" stroke-width="1.6" stroke-linejoin="round"/>
  <path d="M32 11v42" stroke="#3b82f6" stroke-width="1.3" opacity=".45"/>
  <path d="M23 20c2 1 3 2.5 3 4.5M20 33c2 0 3.5 1 4.5 2.6M24 44c1.5-.6 3-.4 4.3.4"
        stroke="#3b82f6" stroke-width="1.2" stroke-linecap="round" opacity=".4"/>
  <circle cx="38.5" cy="30" r="8.5" fill="#ef4444" opacity=".16"/>
  <path d="M38.7 24.4c2.7-.3 5.1 1.4 5.4 3.9.2 1.8 1.2 1.7 1.3 3.4.1 2.5-2.1 4.7-4.7 4.8-2 .1-2.8 1-4.5.4
           -2.4-.8-3.7-2.8-3.5-5.1.1-1.5 1-1.9 1.3-3.2.4-2.1 2-4 4.7-4.2z"
        fill="#f43f5e" stroke="#b91c1c" stroke-width="1" stroke-linejoin="round"/>
  <circle cx="37.4" cy="29.4" r="1.7" fill="#fecaca"/>
</svg>
"""

st.markdown(f"""
<div class="app-header">
  <div class="brand">
    <div class="logo">{BRAIN_SVG}</div>
    <div>
      <h1>NeuroSeg&nbsp;3D</h1>
      <p>Segmentation de tumeurs cérébrales par apprentissage profond · BraTS&nbsp;2021</p>
    </div>
  </div>
  <div class="htags">
    <span class="htag">U-Net 3D</span>
    <span class="htag">HMSA-UNet3D</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---- Barre latérale : configuration ----
with st.sidebar:
    st.markdown(f"""
    <div class="side-brand">
      <div class="dot">{BRAIN_SVG.replace('width="54" height="54"', 'width="22" height="22"')}</div>
      <span>NeuroSeg 3D</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-label">🧠 Modèle</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Modèle", ["U-Net 3D", "HMSA-UNet3D"],
                              label_visibility="collapsed")

    st.markdown('<div class="side-label">📦 Poids du modèle (.pth)</div>', unsafe_allow_html=True)
    default_path = {"U-Net 3D": "models/best_unet3d.pth",
                    "HMSA-UNet3D": "models/hmsa_unet3d_best.pth"}[model_name]
    path_in = st.text_input("Chemin local", value=default_path, label_visibility="collapsed")
    weights_up = st.file_uploader("Téléverser un .pth", type=["pth", "pt"],
                                  label_visibility="collapsed")

    weights_bytes = None
    if weights_up is not None:
        weights_bytes = weights_up.getvalue()
        st.success("Poids chargés depuis l'upload.")
    elif path_in and os.path.exists(path_in):
        with open(path_in, "rb") as fh:
            weights_bytes = fh.read()
        st.success(f"Poids trouvés : {os.path.basename(path_in)}")
    else:
        st.info("Indique un chemin valide ou téléverse le fichier .pth.")

    st.markdown('<div class="side-label">🖥️ Système</div>', unsafe_allow_html=True)
    on_gpu = DEVICE.type == "cuda"
    st.markdown(
        f'<div class="badge"><span class="live" style="background:'
        f'{"#22c55e" if on_gpu else "#94a3b8"}"></span>'
        f'{"GPU · CUDA" if on_gpu else "CPU"}</div>',
        unsafe_allow_html=True)

# ---- Zone principale : upload du patient ----
st.markdown('<div class="section-title">📂 Charger un patient</div>', unsafe_allow_html=True)
files = st.file_uploader(
    "Téléverse les fichiers .nii.gz (flair, t1, t1ce, t2 — et seg si tu veux comparer)",
    type=["gz", "nii"], accept_multiple_files=True,
)

mod_files, seg_file = {}, None
if files:
    for f in files:
        m = detect_modality(f.name)
        if m in MODALITIES:
            mod_files[m] = f
        elif m == "seg":
            seg_file = f
    cols = st.columns(5)
    for i, m in enumerate(MODALITIES):
        cols[i].metric(m.upper(), "✅" if m in mod_files else "❌")
    cols[4].metric("SEG (GT)", "✅" if seg_file else "—")

ready = len(mod_files) == 4 and weights_bytes is not None
run = st.button("🚀 Lancer la segmentation", type="primary", disabled=not ready)
if files and len(mod_files) < 4:
    st.warning("Il manque des modalités. Vérifie que les 4 fichiers (flair, t1, t1ce, t2) sont bien nommés.")

# ---- Inférence ----
if run:
    try:
        model = load_model(model_name, weights_bytes)
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        st.stop()

    with st.spinner("Prétraitement et prédiction en cours…"):
        # Chargement BRUT des 4 modalités (la normalisation se fait dans run_inference,
        # car elle diffère selon le modèle : avant resize pour U-Net, après crop pour HMSA)
        vols, ref_affine, orig_shape = [], None, None
        for m in MODALITIES:
            nii = nib.load(save_nii_from_file(mod_files[m]))
            if ref_affine is None:
                ref_affine, orig_shape = nii.affine, nii.shape
            vols.append(nii.get_fdata().astype(np.float32))
        image_raw = np.stack(vols)                                 # (4,H,W,D) brut

        seg_raw = None
        if seg_file is not None:
            seg_raw = nib.load(save_nii_from_file(seg_file)).get_fdata().astype(np.float32)

        pred3, pred_lab, img128, gt3, restore = run_inference(
            model, model_name, image_raw, seg_raw)

    st.session_state["result"] = dict(
        img128=img128, pred3=pred3, pred_lab=pred_lab, gt3=gt3, restore=restore,
        affine=ref_affine, orig_shape=orig_shape, model_name=model_name,
    )

# ---- Affichage des résultats ----
if "result" in st.session_state:
    R = st.session_state["result"]
    img128, pred3, gt3 = R["img128"], R["pred3"], R["gt3"]
    pred_lab = R["pred_lab"]
    gt_lab = regions_to_label(gt3) if gt3 is not None else None

    st.markdown('<div class="section-title">📊 Résultats</div>', unsafe_allow_html=True)

    # Métriques
    vox = {r: int((pred3[i] > 0.5).sum()) for i, r in enumerate(REGIONS)}
    if gt3 is not None:
        dices = dice_per_class(pred3, gt3)
        cols = st.columns(4)
        for i, r in enumerate(REGIONS):
            cols[i].metric(f"Dice {r}", f"{dices[r]:.3f}")
        cols[3].metric("Dice moyen", f"{np.mean(list(dices.values())):.3f}")
    else:
        cols = st.columns(3)
        for i, r in enumerate(REGIONS):
            cols[i].metric(f"Volume {r} (voxels)", f"{vox[r]:,}")
        st.info("Aucun masque seg fourni : la vérité terrain et le Dice ne sont pas affichés.")

    # Curseur de coupe partagé (par défaut : coupe centrale de la tumeur)
    ref_lab = gt_lab if gt_lab is not None else pred_lab
    default_s = central_tumor_slice(ref_lab)
    s = st.slider("Coupe axiale", 0, IMG_SIZE[2] - 1, default_s)

    tab1, tab2, tab3 = st.tabs(
        ["🎨 Vue multi-classes", "⚪ Tumeur seule (binaire)", "🧊 Vue 3D"])

    # ----- Onglet 1 : vue clinique multi-classes -----
    with tab1:
        legend = " ".join(
            f"<span class='legend-pill' style='background:{c}'>{k}</span>"
            for k, c in CLASS_LEGEND.items())
        st.markdown("**Classes :** " + legend, unsafe_allow_html=True)

        n = 4 + (1 if gt_lab is not None else 0) + 1
        fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
        for c, m in enumerate(MODALITIES):
            ax[c].imshow(np.rot90(img128[c, :, :, s]), cmap="gray")
            ax[c].set_title(m.upper()); ax[c].axis("off")
        k = 4
        if gt_lab is not None:
            ax[k].imshow(np.rot90(img128[0, :, :, s]), cmap="gray")
            ax[k].imshow(np.rot90(gt_lab[:, :, s]), cmap=CLASS_CMAP, vmin=0, vmax=3, interpolation="nearest")
            ax[k].set_title("Vérité terrain"); ax[k].axis("off"); k += 1
        ax[k].imshow(np.rot90(img128[0, :, :, s]), cmap="gray")
        ax[k].imshow(np.rot90(pred_lab[:, :, s]), cmap=CLASS_CMAP, vmin=0, vmax=3, interpolation="nearest")
        ax[k].set_title(f"Prédiction — {R['model_name']}"); ax[k].axis("off")
        fig.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig)

    # ----- Onglet 2 : tumeur seule (binaire) -----
    with tab2:
        region_name = st.radio(
            "Région à isoler",
            ["WT — tumeur entière", "TC — cœur tumoral", "ET — tumeur rehaussée"],
            horizontal=True)
        ridx = {"WT": 0, "TC": 1, "ET": 2}[region_name.split(" ")[0]]
        pmask = pred3[ridx]                                    # masque binaire prédit (128^3)

        st.caption(f"Région **{REGIONS[ridx]}** — volume prédit : "
                   f"**{int((pmask > 0.5).sum()):,} voxels**")

        ncol = 3 if gt3 is not None else 2
        fig2, ax2 = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.4))
        fig2.patch.set_alpha(0)

        # 1) Masque binaire net (tumeur blanche sur fond noir) -> forme/structure
        ax2[0].imshow(np.rot90(pmask[:, :, s]), cmap="gray", vmin=0, vmax=1)
        ax2[0].set_title("Masque binaire (prédiction)", color="#1f2a44")
        ax2[0].axis("off")

        # 2) Tumeur isolée sur l'IRM (le reste est masqué) -> texture interne
        iso = img128[0, :, :, s] * (pmask[:, :, s] > 0.5)
        ax2[1].imshow(np.rot90(iso), cmap="magma")
        ax2[1].set_title("Tumeur isolée (IRM)", color="#1f2a44")
        ax2[1].axis("off")

        # 3) Masque binaire réel (si disponible) -> comparaison
        if gt3 is not None:
            ax2[2].imshow(np.rot90((gt3[ridx][:, :, s] > 0.5).astype(float)), cmap="gray", vmin=0, vmax=1)
            ax2[2].set_title("Masque binaire (vérité terrain)", color="#1f2a44")
            ax2[2].axis("off")

        plt.tight_layout()
        st.pyplot(fig2)

        st.caption("Astuce : déplace le curseur de coupe au-dessus pour parcourir la tumeur en profondeur.")

    # ----- Onglet 3 : rendu 3D interactif de la tumeur -----
    with tab3:
        if not PLOTLY_3D:
            st.warning("Le rendu 3D nécessite deux librairies. Installe-les puis relance :\n\n"
                       "`pip install plotly scikit-image`")
        else:
            st.caption("Tourne, zoome et explore la morphologie de la tumeur (surfaces 3D).")
            cls_map = {"Enhancing": (3, "yellow"),
                       "Core / nécrose": (1, "red"),
                       "Œdème": (2, "limegreen")}
            c1, c2 = st.columns([3, 1])
            with c1:
                chosen = st.multiselect("Classes à afficher", list(cls_map.keys()),
                                        default=list(cls_map.keys()))
            with c2:
                opacity = st.slider("Opacité", 0.1, 1.0, 0.55, 0.05)
            show_gt3d = st.checkbox("Superposer la vérité terrain (transparent)",
                                    value=False) if gt3 is not None else False

            with st.spinner("Construction des surfaces 3D…"):
                meshes = []
                for cname in chosen:
                    val, col = cls_map[cname]
                    mesh = tumor_mesh(pred_lab == val, col, f"Préd · {cname}", opacity)
                    if mesh is not None:
                        meshes.append(mesh)
                if show_gt3d and gt_lab is not None:
                    g = tumor_mesh(gt_lab > 0, "lightgray", "Vérité terrain (WT)", 0.15)
                    if g is not None:
                        meshes.append(g)

            if meshes:
                fig3 = go.Figure(data=meshes)
                fig3.update_layout(
                    scene=dict(aspectmode="data",
                               xaxis=dict(visible=False), yaxis=dict(visible=False),
                               zaxis=dict(visible=False), bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=0, r=0, t=10, b=0), height=600,
                    legend=dict(orientation="h", y=-0.02),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("Volumes prédits — " + " · ".join(
                    f"{REGIONS[i]} : {int((pred3[i] > 0.5).sum()):,} vox" for i in range(3)))
            else:
                st.info("Pas assez de voxels prédits pour générer une surface 3D "
                        "(sélectionne au moins une classe avec de la tumeur).")

    # Téléchargement du masque prédit (ramené à l'espace d'origine du patient)
    st.markdown('<div class="section-title">💾 Export</div>', unsafe_allow_html=True)
    lab128 = np.zeros(IMG_SIZE, dtype=np.float32)
    lab128[pred3[0] > 0.5] = 2          # œdème
    lab128[pred3[1] > 0.5] = 1          # core / nécrose
    lab128[pred3[2] > 0.5] = 4          # enhancing (label BraTS = 4)

    mode, info = R["restore"]
    if mode == "resize":                # U-Net : on ré-interpole vers la taille d'origine
        lab_orig = F.interpolate(torch.from_numpy(lab128)[None, None],
                                 size=R["orig_shape"], mode="nearest")[0, 0].numpy().astype(np.uint8)
    else:                               # HMSA : on replace le crop dans le volume d'origine
        lab_orig = invert_crop_or_pad(lab128, info).astype(np.uint8)

    out_nii = nib.Nifti1Image(lab_orig, R["affine"])
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz").name
    nib.save(out_nii, tmp_path)
    with open(tmp_path, "rb") as fh:
        st.download_button("⬇️ Télécharger le masque prédit (.nii.gz)",
                           fh.read(), file_name="prediction_seg.nii.gz")
else:
    st.info("Charge un patient et clique sur « Lancer la segmentation » pour voir les résultats.")
