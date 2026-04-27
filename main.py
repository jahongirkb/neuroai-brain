from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import SwinUNETR
import tempfile
import os

# ============================================================
# APP
# ============================================================
app = FastAPI(title="NeuroAI — Brain Tumor Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODEL YUKLASH
# ============================================================
print("⏳ Model yuklanmoqda...")

model = SwinUNETR(
    in_channels=4,
    out_channels=3,
    feature_size=48,
)

model_dict = torch.load("model.pt", map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict(model_dict)
model.eval()

print("✅ Model muvaffaqiyatli yuklandi!")

# ============================================================
# YORDAMCHI FUNKSIYALAR
# ============================================================

def load_and_normalize(path: str) -> np.ndarray:
    """NIfTI faylni o'qiydi (normalize preprocess_4modality da bo'ladi)."""
    return nib.load(path).get_fdata().astype(np.float32)


def preprocess_4modality(flair, t1, t1ce, t2) -> torch.Tensor:
    """
    MONAI style preprocessing — non-zero normalize + transpose
    """
    channels = []
    for arr in [flair, t1, t1ce, t2]:
        # Non-zero normalize (MONAI rasmiy usul)
        mask = arr > 0
        if mask.sum() > 0:
            arr[mask] = (arr[mask] - arr[mask].mean()) / (arr[mask].std() + 1e-8)
        # (H, W, D) → (D, H, W)
        arr_t = np.transpose(arr, (2, 0, 1))
        t = torch.tensor(arr_t).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(128, 128, 128), mode="trilinear", align_corners=False
        )
        channels.append(t.squeeze(0))

    tensor = torch.cat(channels, dim=0).unsqueeze(0)
    return tensor


def compute_percentages(seg: np.ndarray) -> dict:
    et_mask = seg[0] > 0.5
    tc_mask = seg[1] > 0.5
    wt_mask = seg[2] > 0.5

    et_pct  = round(float(et_mask.mean()) * 100, 1)
    tc_pct  = round(float(tc_mask.mean()) * 100, 1)
    wt_pct  = round(float(wt_mask.mean()) * 100, 1)

    necro   = round(max(tc_pct - et_pct, 0), 1)
    edema   = round(max(wt_pct - tc_pct, 0), 1)
    tumor   = round(tc_pct, 1)
    enhanc  = round(et_pct, 1)
    healthy = round(max(100 - wt_pct - 5.6, 0), 1)
    vent    = round(max(100 - tumor - necro - edema - enhanc - healthy, 0), 1)

    return {
        "tumor":   tumor,
        "necro":   necro,
        "edema":   edema,
        "healthy": healthy,
        "vent":    vent,
        "enhanc":  enhanc,
    }


def determine_status(seg_dict: dict) -> tuple:
    wt = seg_dict["tumor"] + seg_dict["necro"] + seg_dict["edema"]
    if wt > 15:
        return "critical", 93
    elif wt > 3:
        return "attention", 87
    else:
        return "normal", 96


def make_slices_and_3d(input_tensor, seg_np):
    """128 ta slice PNG + 3D nuqtalar qaytaradi."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io, base64
    # Asl o'lchamga qaytarish (155, 240, 240)
    orig_tensor = torch.nn.functional.interpolate(
        input_tensor, size=(155, 240, 240), mode="trilinear", align_corners=False
    )
    orig_seg = np.stack([
        torch.nn.functional.interpolate(
            torch.tensor(seg_np[i]).unsqueeze(0).unsqueeze(0),
            size=(155, 240, 240), mode="trilinear", align_corners=False
        ).squeeze().numpy()
        for i in range(3)
    ])
    et_vol_orig = orig_seg[0] > 0.5
    tc_vol_orig = orig_seg[1] > 0.5
    wt_vol_orig = orig_seg[2] > 0.5
    et_vol = seg_np[0] > 0.5
    tc_vol = seg_np[1] > 0.5
    wt_vol = seg_np[2] > 0.5

    # ── 128 ta slice ──
    slices_b64 = []
    total_slices = orig_tensor.shape[2]  # 155
    for i in range(total_slices):
        mri_slice = orig_tensor[0, 0, i].numpy()

        # Seg maskani orig o'lchamga moslashtirish
        et_s = et_vol_orig[i]
        tc_s = tc_vol_orig[i]
        wt_s = wt_vol_orig[i]

        fig, ax = plt.subplots(figsize=(4, 4), facecolor='black')
        ax.imshow(mri_slice, cmap='gray', aspect='auto')

        overlay = np.zeros((*mri_slice.shape, 4))
        overlay[wt_s, 0]=0.02; overlay[wt_s, 1]=0.71
        overlay[wt_s, 2]=0.83; overlay[wt_s, 3]=0.45
        overlay[tc_s, 0]=0.66; overlay[tc_s, 1]=0.33
        overlay[tc_s, 2]=0.97; overlay[tc_s, 3]=0.65
        overlay[et_s, 0]=0.96; overlay[et_s, 1]=0.62
        overlay[et_s, 2]=0.04; overlay[et_s, 3]=0.80

        ax.imshow(overlay, aspect='auto')
        ax.axis('off')
        plt.tight_layout(pad=0)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=60,
                    facecolor='black', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        slices_b64.append(base64.b64encode(buf.read()).decode('utf-8'))

    # ── O'rta slice (64) yuqori sifatda ──
    # O'smaning eng katta slice ini avtomatik topish
    wt_per_slice = wt_vol.sum(axis=(1,2))
    mid = int(np.argmax(wt_per_slice))
    if mid == 0:
        mid = 64
    print(f"✅ O'sma eng katta slice: {mid}")
    wt_per_slice = wt_vol.sum(axis=(1,2))
    mid = int(np.argmax(wt_per_slice))
    if mid == 0:
        mid = 64
    print(f"✅ O'sma eng katta slice: {mid}")

    # ← SHU YERGA QO'SHING:
    orig_tensor = torch.nn.functional.interpolate(
        input_tensor, size=(155, 240, 240), mode="trilinear", align_corners=False
    )
    orig_seg = np.stack([
        torch.nn.functional.interpolate(
            torch.tensor(seg_np[i]).unsqueeze(0).unsqueeze(0),
            size=(155, 240, 240), mode="trilinear", align_corners=False
        ).squeeze().numpy()
        for i in range(3)
    ])
    et_vol_orig = orig_seg[0] > 0.5
    tc_vol_orig = orig_seg[1] > 0.5
    wt_vol_orig = orig_seg[2] > 0.5

    # O'smaning eng katta slice (orig o'lchamda)
    wt_per_slice_orig = wt_vol_orig.sum(axis=(1,2))
    mid_orig = int(np.argmax(wt_per_slice_orig))
    if mid_orig == 0:
        mid_orig = 77

    mri_mid = orig_tensor[0, 0, mid_orig].numpy()
    et_mid = et_vol_orig[mid_orig]
    tc_mid = tc_vol_orig[mid_orig]
    wt_mid = wt_vol_orig[mid_orig]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#0f1628')
    # mri_mid = input_tensor[0, 0, mid].numpy()
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#0f1628')

    axes[0].imshow(mri_mid, cmap='gray', aspect='auto')
    axes[0].set_title('MRI (FLAIR)', color='white', fontsize=13, pad=10)
    axes[0].axis('off')

    axes[1].imshow(mri_mid, cmap='gray', aspect='auto')
    overlay_mid = np.zeros((*mri_mid.shape, 4))
    overlay_mid[wt_mid, 0]=0.02; overlay_mid[wt_mid, 1]=0.71
    overlay_mid[wt_mid, 2]=0.83; overlay_mid[wt_mid, 3]=0.45
    overlay_mid[tc_mid, 0]=0.66; overlay_mid[tc_mid, 1]=0.33
    overlay_mid[tc_mid, 2]=0.97; overlay_mid[tc_mid, 3]=0.65
    overlay_mid[et_mid, 0]=0.96; overlay_mid[et_mid, 1]=0.62
    overlay_mid[et_mid, 2]=0.04; overlay_mid[et_mid, 3]=0.80
    axes[1].imshow(overlay_mid, aspect='auto')
    axes[1].set_title("Segmentatsiya natijasi", color='white', fontsize=13, pad=10)
    axes[1].axis('off')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#06b6d4', alpha=0.7, label='Whole Tumor (Edema)'),
        Patch(facecolor='#a855f7', alpha=0.8, label='Tumor Core'),
        Patch(facecolor='#f59e0b', alpha=0.9, label='Enhancing Tumor'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower left',
                   fontsize=8, facecolor='#0f1628',
                   labelcolor='white', framealpha=0.8)
    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                facecolor='#0f1628', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    slice_img = base64.b64encode(buf.read()).decode('utf-8')

    # ── 3D nuqtalar ──
    brain_mask = input_tensor[0, 0].numpy() > 0.1

    def sample_points(mask, n=800):
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return [], [], []
        idx = np.random.choice(len(coords), min(n, len(coords)), replace=False)
        pts = coords[idx]
        return pts[:,0].tolist(), pts[:,1].tolist(), pts[:,2].tolist()

    healthy_mask = brain_mask & ~wt_vol
    bx,  by,  bz  = sample_points(healthy_mask, 3000)
    wtx, wty, wtz = sample_points(wt_vol & ~tc_vol, 1000)
    tcx, tcy, tcz = sample_points(tc_vol & ~et_vol, 800)
    etx, ety, etz = sample_points(et_vol, 600)

    points3d = {
        "brain":  {"x": bx,  "y": by,  "z": bz,  "label": "Sog'lom to'qima"},
        "edema":  {"x": wtx, "y": wty, "z": wtz, "label": "Edema"},
        "tumor":  {"x": tcx, "y": tcy, "z": tcz, "label": "Tumor Core"},
        "enhanc": {"x": etx, "y": ety, "z": etz, "label": "Enhancing Tumor"},
    }

    return slices_b64, slice_img, points3d, total_slices


# ============================================================
# ENDPOINT: /analyze — 4 ta modality
# ============================================================

@app.post("/analyze")
async def analyze(
    flair: UploadFile = File(...),
    t1:    UploadFile = File(...),
    t1ce:  UploadFile = File(...),
    t2:    UploadFile = File(...),
):
    """
    4 ta NIfTI fayl qabul qilib segmentatsiya natijasini qaytaradi.
    flair, t1, t1ce, t2 — BraTS2021 modality lari
    """
    tmp_paths = []

    try:
        arrays = []
        for upload in [flair, t1, t1ce, t2]:
            fname = upload.filename or ""
            suffix = ".nii.gz" if fname.endswith(".nii.gz") else ".nii"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await upload.read())
                tmp_paths.append(tmp.name)

        for path in tmp_paths:
            arrays.append(load_and_normalize(path))

        input_tensor = preprocess_4modality(*arrays)

        with torch.no_grad():
            output = torch.sigmoid(model(input_tensor))

        seg_np = output[0].numpy()
        seg_dict = compute_percentages(seg_np)
        status, conf = determine_status(seg_dict)
        slices_b64, slice_img, points3d, total_slices = make_slices_and_3d(input_tensor, seg_np)

        return JSONResponse({
            "status":       status,
            "conf":         conf,
            "seg":          seg_dict,
            "slice_img":    slice_img,
            "slices":       slices_b64,
            "total_slices": total_slices,
            "points3d":     points3d,
        })

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print("🔴 XATO:\n", err_msg)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for path in tmp_paths:
            if os.path.exists(path):
                os.unlink(path)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "model": "SwinUNETR BraTS2021 (4 modality)"}


# ============================================================
# STATIK FAYLLAR
# ============================================================
app.mount("/", StaticFiles(directory=".", html=True), name="static")
