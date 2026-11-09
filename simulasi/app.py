"""
AI Rekomendasi - Simulasi Rekomendasi Mata Kuliah Pilihan
Teknologi: Python, Streamlit, Pandas, Scikit-learn

Jalankan:  streamlit run app.py
"""
import html

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- DATA MASTER
NILAI = {
    "alpro": "Algoritma & Pemrograman",
    "basdat": "Basis Data",
    "stat": "Statistika",
    "jarkom": "Jaringan Komputer",
    "rpl": "Rekayasa Perangkat Lunak",
    "mat": "Matematika Diskrit",
}
MINAT = ["AI", "Data", "Web", "Keamanan", "Jaringan", "Mobile", "Desain"]

MATKUL = pd.DataFrame(
    [
        ("Machine Learning", {"alpro": 2, "stat": 3, "mat": 2}, ["AI", "Data"]),
        ("Data Mining", {"basdat": 3, "stat": 3, "alpro": 1}, ["Data", "AI"]),
        ("Pengembangan Web Lanjut", {"alpro": 2, "basdat": 2, "rpl": 2}, ["Web", "Desain"]),
        ("Keamanan Informasi", {"jarkom": 3, "mat": 2, "alpro": 1}, ["Keamanan", "Jaringan"]),
        ("Jaringan Nirkabel & IoT", {"jarkom": 3, "alpro": 2}, ["Jaringan", "Mobile"]),
        ("Pemrograman Mobile", {"alpro": 3, "rpl": 2}, ["Mobile", "Desain"]),
        ("Interaksi Manusia & Komputer", {"rpl": 3, "stat": 1}, ["Desain", "Web"]),
        ("Big Data Analytics", {"basdat": 3, "stat": 2, "jarkom": 1}, ["Data", "Jaringan"]),
    ],
    columns=["matkul", "bobot_nilai", "tag_minat"],
)


# ------------------------------------------------------------ SKOR BERBASIS ATURAN
def skor_nilai(nilai, bobot):
    return sum(nilai[k] * w for k, w in bobot.items()) / sum(bobot.values())


def skor_minat(minat_terpilih, tag):
    return 100 * sum(t in minat_terpilih for t in tag) / len(tag)


# ------------------------------------------------------------ DATASET SIMULASI + MODEL ML
@st.cache_data
def buat_dataset(n=1500, seed=42):
    """Dataset historis simulasi. Ganti dengan data asli (CSV) jika tersedia."""
    rng = np.random.default_rng(seed)
    baris = []
    for _ in range(n):
        nilai = {k: int(np.clip(rng.normal(72, 13), 30, 100)) for k in NILAI}
        minat = {m: int(rng.random() < 0.3) for m in MINAT}
        terpilih = [m for m, v in minat.items() if v]
        skor = [
            0.6 * skor_nilai(nilai, r.bobot_nilai)
            + 0.4 * skor_minat(terpilih, r.tag_minat)
            + rng.normal(0, 6)
            for _, r in MATKUL.iterrows()
        ]
        baris.append({**nilai, **minat, "matkul_dipilih": MATKUL.matkul[int(np.argmax(skor))]})
    return pd.DataFrame(baris)


@st.cache_resource
def latih_model():
    df = buat_dataset()
    fitur = list(NILAI) + MINAT
    X, y = df[fitur], df["matkul_dipilih"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    akurasi = accuracy_score(y_te, model.predict(X_te))
    pentingnya = pd.Series(model.feature_importances_, index=fitur).sort_values(ascending=False)
    return model, fitur, akurasi, pentingnya, df


def penjelasan(r, nilai, minat_terpilih):
    """Daftar alasan (HTML) mengapa mata kuliah ini direkomendasikan."""
    e = html.escape
    poin = []

    # 1) Alasan dari nilai
    urut = sorted(r["bobot"].items(), key=lambda kv: -kv[1])[:2]
    utama = [f"{NILAI[k]} ({nilai[k]})" for k, _ in urut]
    teks = f"Mata kuliah ini paling bergantung pada {' dan '.join(utama)}. Rata-rata nilai prasyarat berbobotmu {r['sn']:.0f}."
    lemah = [NILAI[k] for k, _ in urut if nilai[k] < 65]
    kuat = [NILAI[k] for k, _ in urut if nilai[k] >= 75]
    if kuat:
        teks += f" Nilai {' dan '.join(kuat)} sudah kuat."
    if lemah:
        teks += f" Nilai {' dan '.join(lemah)} masih perlu ditingkatkan agar lebih siap."
    poin.append(("Nilai", teks))

    # 2) Alasan dari minat
    cocok = [t for t in r["tag"] if t in minat_terpilih]
    belum = [t for t in r["tag"] if t not in minat_terpilih]
    if cocok and not belum:
        teks = f"Semua bidang mata kuliah ini ({', '.join(cocok)}) sesuai dengan minatmu."
    elif cocok:
        teks = f"Bidang {', '.join(cocok)} sesuai minatmu, sedangkan {', '.join(belum)} belum kamu pilih."
    else:
        teks = f"Bidang mata kuliah ini ({', '.join(r['tag'])}) belum sesuai dengan minat yang kamu pilih."
    poin.append(("Minat", teks))

    # 2) Alasan dari model AI
    poin.append((
        "Model AI",
        f"Berdasarkan pola data mahasiswa sebelumnya, model menempatkan mata kuliah ini di peringkat "
        f"{r['rank_ai']} dari {len(MATKUL)} dengan probabilitas {r['ai']:.0f}%.",
    ))
    return "".join(f"<li><b>{e(j)}:</b> {e(t)}</li>" for j, t in poin)


# ------------------------------------------------------------------------- UI
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif}
.block-container{max-width:780px;padding-top:2.2rem;padding-bottom:3rem}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0}

.hero{background:linear-gradient(120deg,#0b4f6c 0%,#0e7490 55%,#0d9488 100%);color:#fff;border-radius:22px;
 padding:34px 32px;margin-bottom:24px;box-shadow:0 12px 32px rgba(14,116,144,.28)}
.hero small{display:block;letter-spacing:.14em;font-size:.72rem;font-weight:600;opacity:.8;margin-bottom:10px}
.hero h1{color:#fff;font-size:2rem;font-weight:800;line-height:1.2;margin:0 0 10px;padding:0}
.hero p{margin:0;opacity:.92;font-size:1rem;max-width:560px;line-height:1.55}

[data-testid="stForm"]{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:26px 26px 22px;
 box-shadow:0 4px 18px rgba(15,23,42,.06)}
.step{display:flex;align-items:center;gap:12px;margin:6px 0 4px;font-weight:700;font-size:1.12rem;color:#0f172a}
.step b{background:#0d9488;color:#fff;border-radius:10px;width:30px;height:30px;display:inline-flex;
 align-items:center;justify-content:center;font-size:.9rem}
div[data-baseweb="input"],div[data-baseweb="select"]>div{border-radius:10px!important}
div[data-testid="stFormSubmitButton"]>button{border-radius:12px;font-weight:700;font-size:1rem;
 padding:.7rem 1rem;width:100%;background:#0d9488;border:0;color:#fff;margin-top:6px}
div[data-testid="stFormSubmitButton"]>button:hover{background:#0f766e;color:#fff}
[data-testid="stExpander"]{background:#fff;border:1px solid #e2e8f0!important;border-radius:14px}

.sect{margin:30px 0 14px;font-weight:800;font-size:1.3rem;color:#0f172a}
.sect span{display:block;font-size:.88rem;font-weight:400;color:#64748b;margin-top:2px}
.res{display:flex;gap:16px;background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:18px 20px;
 margin-bottom:14px;box-shadow:0 4px 16px rgba(15,23,42,.05)}
.res.best{border:2px solid #0d9488;background:linear-gradient(180deg,#f0fdfa,#fff);
 box-shadow:0 10px 28px rgba(13,148,136,.18)}
.rank{flex:0 0 40px;height:40px;border-radius:12px;background:#e2e8f0;color:#334155;font-weight:800;
 display:flex;align-items:center;justify-content:center;font-size:1.1rem}
.best .rank{background:#0d9488;color:#fff}
.body{flex:1;min-width:0}
.top{display:flex;justify-content:space-between;align-items:center;gap:10px}
.lbl{font-size:.7rem;font-weight:700;letter-spacing:.1em;color:#0d9488;margin-bottom:2px}
.nm{font-weight:700;font-size:1.1rem;color:#0f172a}
.pct{font-weight:800;font-size:1.5rem;color:#0d9488;line-height:1}
.pct small{font-size:.7rem;font-weight:600;color:#64748b;display:block;text-align:right;margin-top:3px}
.bar{height:9px;border-radius:8px;background:#e2e8f0;overflow:hidden}
.bar i{display:block;height:100%;border-radius:8px;background:linear-gradient(90deg,#0e7490,#14b8a6)}
.main-bar{margin:12px 0 10px}
.chip{display:inline-block;font-size:.76rem;font-weight:500;padding:3px 11px;border-radius:999px;margin-right:6px;
 background:#e2e8f0;color:#475569}
.chip.on{background:#ccfbf1;color:#0f766e;font-weight:600}
.mini{display:grid;grid-template-columns:52px 1fr 34px;align-items:center;gap:10px;margin-top:9px;
 font-size:.8rem;color:#64748b}
.mini .bar{height:6px}
.mini em{font-style:normal;font-weight:600;color:#334155;text-align:right}
.why{margin-top:14px;padding-top:12px;border-top:1px solid #e2e8f0;font-size:.9rem;color:#475569;line-height:1.55}
.why b{color:#0f172a}.why ul{margin:6px 0 0;padding-left:18px}.why li{margin-bottom:5px}
.note{font-size:.8rem;color:#94a3b8;text-align:center;margin-top:26px}
@media(max-width:600px){.hero{padding:26px 22px}.hero h1{font-size:1.6rem}.res{padding:16px}}
</style>
"""

st.set_page_config(page_title="Rekomendasi Mata Kuliah Pilihan")
st.markdown(CSS, unsafe_allow_html=True)
model, fitur, akurasi, pentingnya, df_latih = latih_model()

st.markdown(
    '<div class="hero"><small>SISTEM PENDUKUNG KEPUTUSAN</small>'
    "<h1>Rekomendasi Mata Kuliah Pilihan</h1>"
    "<p>Isi nilai dan minatmu. Sistem akan menghitung dan memilihkan mata kuliah pilihan "
    "yang paling cocok untukmu.</p></div>",
    unsafe_allow_html=True,
)

with st.form("form"):
    st.markdown('<div class="step"><b>1</b>Nilai mata kuliah dasar</div>', unsafe_allow_html=True)
    st.caption("Masukkan nilai akhir (0-100) untuk tiap mata kuliah.")
    kol = st.columns(2)
    nilai = {}
    for i, (k, label) in enumerate(NILAI.items()):
        nilai[k] = kol[i % 2].number_input(label, 0, 100, 75, step=1)

    st.markdown('<div class="step" style="margin-top:14px"><b>2</b>Bidang yang kamu minati</div>',
                unsafe_allow_html=True)
    minat_terpilih = st.multiselect("Pilih satu atau lebih", MINAT, placeholder="Klik untuk memilih minat")

    with st.expander("Pengaturan lanjutan (boleh dilewati)"):
        w_nilai = st.slider("Pengaruh nilai dibanding minat (%)", 0, 100, 60) / 100
        w_ai = st.slider("Porsi model AI dalam hasil (%)", 0, 100, 50) / 100

    kirim = st.form_submit_button("Lihat Rekomendasi", type="primary")

if kirim:
    if not minat_terpilih:
        st.warning("Pilih minimal satu minat dulu ya.")
        st.stop()

    x = pd.DataFrame([{**nilai, **{m: int(m in minat_terpilih) for m in MINAT}}])[fitur]
    proba = pd.Series(model.predict_proba(x)[0], index=model.classes_)
    p_max = max(proba.max(), 1e-9)

    baris = []
    for _, r in MATKUL.iterrows():
        sn = skor_nilai(nilai, r.bobot_nilai)
        sm = skor_minat(minat_terpilih, r.tag_minat)
        aturan = w_nilai * sn + (1 - w_nilai) * sm
        akhir = (1 - w_ai) * aturan + w_ai * (proba[r.matkul] / p_max * 100)
        baris.append({"matkul": r.matkul, "tag": r.tag_minat, "sn": sn, "sm": sm,
                      "ai": proba[r.matkul] * 100, "akhir": akhir, "bobot": r.bobot_nilai})
    hasil = pd.DataFrame(baris).sort_values("akhir", ascending=False).reset_index(drop=True)
    hasil["rank_ai"] = hasil["ai"].rank(ascending=False, method="min").astype(int)

    st.markdown('<div class="sect">Hasil rekomendasi<span>Tiga mata kuliah yang paling cocok '
                "berdasarkan nilai dan minatmu.</span></div>", unsafe_allow_html=True)
    for i in range(3):
        r = hasil.loc[i]
        chips = "".join(
            f'<span class="chip {"on" if t in minat_terpilih else ""}">{html.escape(t)}</span>'
            for t in r["tag"]
        )
        label = '<div class="lbl">REKOMENDASI UTAMA</div>' if i == 0 else ""
        st.markdown(
            f'<div class="res {"best" if i == 0 else ""}"><div class="rank">{i + 1}</div><div class="body">'
            f'<div class="top"><div>{label}<div class="nm">{html.escape(r["matkul"])}</div></div>'
            f'<div class="pct">{r["akhir"]:.0f}%<small>kecocokan</small></div></div>'
            f'<div class="bar main-bar"><i style="width:{min(r["akhir"], 100):.0f}%"></i></div>'
            f"<div>{chips}</div>"
            f'<div class="mini"><span>Nilai</span><div class="bar"><i style="width:{min(r["sn"], 100):.0f}%"></i></div>'
            f'<em>{r["sn"]:.0f}</em></div>'
            f'<div class="mini"><span>Minat</span><div class="bar"><i style="width:{min(r["sm"], 100):.0f}%"></i></div>'
            f'<em>{r["sm"]:.0f}</em></div>'
            f'<div class="why"><b>Mengapa direkomendasikan?</b><ul>{penjelasan(r, nilai, minat_terpilih)}</ul></div>'
            "</div></div>",
            unsafe_allow_html=True,
        )

    with st.expander("Lihat detail perhitungan"):
        tabel = hasil.rename(
            columns={"matkul": "Mata Kuliah", "sn": "Skor Nilai", "sm": "Skor Minat",
                     "ai": "Probabilitas AI (%)", "akhir": "Skor Akhir"}
        ).drop(columns="tag")
        st.dataframe(tabel.round(1))
        st.caption("Skor akhir = gabungan skor aturan (nilai + minat) dan probabilitas model "
                   "Random Forest (scikit-learn).")

st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
with st.expander("Tentang model AI (untuk penilaian)"):
    st.metric("Akurasi model pada data uji", f"{akurasi * 100:.1f}%")
    st.write("Fitur yang paling berpengaruh:")
    st.bar_chart(pentingnya.head(8))
    st.caption("Model dilatih dari dataset simulasi yang dibangkitkan acak, bukan data asli.")
    st.dataframe(df_latih.head(30))

st.markdown('<div class="note">Simulasi rekomendasi mata kuliah pilihan | Python, Streamlit, Pandas, Scikit-learn</div>',
            unsafe_allow_html=True)