# Application — Segmentation 3D de tumeurs cérébrales (BraTS)

Interface Streamlit pour tester tes modèles entraînés (U-Net 3D et HMSA-UNet3D)
sur un patient : choix du modèle, upload des IRM, visualisation vérité terrain +
prédiction colorée, et export du masque prédit.

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Placer les poids

Crée un dossier `models/` à côté de `app.py` et mets-y tes deux fichiers :

```
models/
 ├─ best_unet3d.pth
 └─ best_hmsa_unet3d.pth
```

Ce sont les `state_dict` sauvegardés pendant l'entraînement
(`torch.save(model.state_dict(), ...)`). Tu peux aussi les téléverser
directement depuis la barre latérale de l'app.

## 3. Intégrer ton HMSA-UNet3D

Le U-Net 3D est déjà inclus. Pour le second modèle :

1. Ouvre `app.py`.
2. Repère la zone **« COLLE ICI TA CLASSE HMSA-UNet3D »**.
3. Colle ta classe `HMSA_UNet3D` (exactement celle de l'entraînement).
4. Mets `HMSA_READY = True`.

⚠️ La classe doit être identique à l'entraînement, sinon les poids ne se
chargeront pas. Garde la signature `in_ch=4, out_ch=3` (ou adapte `build_model`).

## 4. Lancer

```bash
streamlit run app.py
```

L'app s'ouvre dans le navigateur. Téléverse les 4 fichiers `.nii.gz`
(`flair, t1, t1ce, t2`) — et le `seg` si tu veux comparer à la vérité terrain et
voir le Dice par classe — puis clique sur **Lancer la segmentation**.

### Sur Kaggle / Colab (sans navigateur local)

```bash
pip install -r requirements.txt
npm install -g localtunnel        # ou utilise ngrok / cloudflared
streamlit run app.py &> log.txt &
npx localtunnel --port 8501
```

## Couleurs des classes
- 🔴 rouge = core / nécrose (label 1)
- 🟢 vert = œdème (label 2)
- 🟡 jaune = enhancing (label 4)
