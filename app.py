import json
from base64 import b64encode
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"
DATASET_URL = "https://www.kaggle.com/datasets/mobeenfatimah/hivaids-clinical-global-dataset"
YES_NO = {0: "Tidak", 1: "Ya"}
EXAMPLE_A = (560, 80, 0, 0)
EXAMPLE_B = (110, 54000, 1, 1)


@st.cache_data
def asset_data(name: str, modified_at: int) -> str:
    return b64encode((ASSETS / name).read_bytes()).decode("ascii")


def show_cover() -> None:
    temple_photo = asset_data("parthenon.webp", (ASSETS / "parthenon.webp").stat().st_mtime_ns)
    statue_art = asset_data("asklepios.webp", (ASSETS / "asklepios.webp").stat().st_mtime_ns)
    sanctuary_art = asset_data("sanctuary.webp", (ASSETS / "sanctuary.webp").stat().st_mtime_ns)
    st.html(f"""
    <div class="lab-covers">
      <header class="lab-cover" id="awal">
        <img class="lab-temple" src="data:image/webp;base64,{temple_photo}"
             alt="Ilustrasi fasad Parthenon dari depan, dari atap sampai dasar tangga" fetchpriority="high">
        <div class="lab-cover-content">
          <nav class="lab-masthead" aria-label="Navigasi utama">
            <div class="lab-nav-links">
              <a href="#asklepios">Asklepios</a>
              <a href="./#ruang-lab">Simulasi</a>
            </div>
          </nav>
          <div class="lab-hero-copy">
            <h1>HIV Outcomes <em>Lab</em></h1>
            <p class="lab-deck">Studi mortalitas lima tahun pada data HIV sintetis.</p>
          </div>
        </div>
      </header>
      <section class="lab-story" id="asklepios" aria-labelledby="asklepios-title">
        <img class="lab-story-backdrop" src="data:image/webp;base64,{sanctuary_art}"
             alt="" loading="lazy">
        <span class="lab-story-inscription" aria-hidden="true">Asklepios</span>
        <div class="lab-story-art">
          <img class="lab-statue" src="data:image/webp;base64,{statue_art}"
               alt="Ilustrasi potret patung marmer Asklepios dengan tongkat berular"
               width="1024" height="1536">
        </div>
        <div class="lab-story-copy">
          <h2 id="asklepios-title">Asklepios</h2>
          <p>Dalam mitologi Yunani, Asklepios dikenal sebagai dewa penyembuhan. Patung marmer dan tongkat berularnya menjadi inspirasi visual untuk lab ini.</p>
          <a class="lab-story-link" href="./#ruang-lab">Lihat eksperimen <span aria-hidden="true">↘</span></a>
        </div>
      </section>
    </div>
    """)


@st.cache_data
def load_metrics(path: Path, modified_at: int) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(path: Path, modified_at: int, expected_hash: str):
    model_bytes = path.read_bytes()
    if sha256(model_bytes).hexdigest() != expected_hash:
        raise ValueError("Model dan metrik berasal dari versi berbeda.")
    return joblib.load(BytesIO(model_bytes))


def percent(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}".replace(".", ",")


def count(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def fill_example(cd4: int, viral_load: int, tb: int, resistance: int) -> None:
    st.session_state.update(cd4=cd4, viral_load=viral_load, tb=tb, resistance=resistance)


def show_simulation(model, metrics: dict, artifact_version: tuple[int, int]) -> None:
    if "cd4" not in st.session_state:
        fill_example(*EXAMPLE_A)

    with st.container(key="lab_simulation"):
        st.html("""
          <div class="lab-workspace-title">
            <h2>Simulasi prediksi</h2>
            <p>Pilih profil contoh atau isi empat nilai untuk profil fiktif. Model menghitung skor berdasarkan pola dalam data sintetis.</p>
          </div>
        """)
        inputs, output = st.columns([5, 4], gap="medium")

        with inputs, st.container(key="lab_inputs"):
            with st.container(horizontal=True, key="lab_presets"):
                st.button("Profil contoh A", on_click=fill_example, args=EXAMPLE_A)
                st.button("Profil contoh B", on_click=fill_example, args=EXAMPLE_B)

            with st.form("clinical_profile", border=False):
                cd4 = st.number_input(
                    "Jumlah CD4 saat ini (sel/µL)", min_value=5, max_value=1276, value=None,
                    step=1, key="cd4", persist_state="session",
                )
                viral_load = st.number_input(
                    "Viral load saat ini (salinan/mL)", min_value=0, max_value=750000, value=None,
                    step=1, key="viral_load", persist_state="session",
                )
                tb = st.selectbox(
                    "Koinfeksi tuberkulosis (TB)", (0, 1), index=None, format_func=YES_NO.get,
                    key="tb", persist_state="session",
                )
                resistance = st.selectbox(
                    "Mutasi resistansi obat", (0, 1), index=None, format_func=YES_NO.get,
                    key="resistance", persist_state="session",
                )
                submitted = st.form_submit_button("Jalankan prediksi", type="primary")

        threshold = metrics["threshold"]
        if submitted and None in (cd4, viral_load, tb, resistance):
            inputs.error("Isi keempat nilai sebelum menjalankan prediksi.")
        elif submitted:
            profile = pd.DataFrame([{
                "Current_CD4_Count": cd4,
                "Current_Viral_Load": viral_load,
                "TB_Coinfection": tb,
                "Drug_Resistance_Mutation": resistance,
            }])
            score = float(model.predict_proba(profile)[0, 1])
            st.session_state.last_prediction = artifact_version, score, cd4, viral_load, tb, resistance

        with output, st.container(key="lab_result"):
            st.markdown("### Hasil simulasi")
            prediction = st.session_state.get("last_prediction")
            score = 0.0
            if prediction is None or prediction[0] != artifact_version:
                st.html('<div class="lab-empty-score" aria-hidden="true">—</div>')
                st.write("Jalankan prediksi untuk menampilkan skor dan kelas model.")
            else:
                _, score, submitted_cd4, submitted_vl, submitted_tb, submitted_resistance = prediction
                st.metric("Skor model untuk kelas 1", percent(score, digits=2))
                predicted_class = int(score >= threshold)
                outcome = "Mortalitas dalam 5 tahun" if predicted_class else "Tanpa mortalitas dalam 5 tahun"
                st.html(f"""
                  <p class="lab-result-class" role="status"><span class="lab-class-badge">Kelas {predicted_class}</span>
                    · {outcome}</p>
                """)
            st.html(f"""
              <div class="lab-score-track" aria-hidden="true"
                   style="--lab-score: {score:.2%}; --lab-threshold: {threshold:.2%}">
                <span></span>
              </div>
            """)
            st.caption(f"Model {metrics['selected_model']}. Kelas 1 mulai dari {percent(threshold, digits=2)}.")
            if prediction is not None and prediction[0] == artifact_version:
                st.caption(
                    f"Input terakhir · CD4 {count(submitted_cd4)} sel/µL · "
                    f"Viral load {count(submitted_vl)} salinan/mL · "
                    f"TB {YES_NO[submitted_tb].lower()} · Mutasi {YES_NO[submitted_resistance].lower()}"
                )


def show_evaluation(metrics: dict) -> None:
    from html import escape
    from math import ceil

    selected_model = metrics["selected_model"]
    test = metrics["test"]
    primary = (("Average Precision", "average_precision"), ("ROC AUC", "roc_auc"), ("F1", "f1"))
    secondary = (
        ("Presisi", "precision"), ("Recall", "recall"), ("Akurasi", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
    )
    primary_metrics = "".join(
        f"<div><dt>{label}</dt><dd>{percent(test[key])}</dd></div>" for label, key in primary
    )
    secondary_metrics = "".join(
        f"<div><dt>{label}</dt><dd>{percent(test[key])}</dd></div>" for label, key in secondary
    )

    rows = sorted(metrics["models"], key=lambda row: row["average_precision"], reverse=True)
    chart_max = max(.1, ceil(rows[0]["average_precision"] * 10) / 10)
    comparison_rows = "".join(
        f"""
          <tr class="{'lab-eval-selected' if row['name'] == selected_model else ''}">
            <th scope="row">{escape(row['name'])}</th>
            <td>
              <span class="lab-eval-ap">{percent(row['average_precision'])}</span>
              <span class="lab-eval-bar" aria-hidden="true">
                <span style="width: {row['average_precision'] / chart_max:.2%}"></span>
              </span>
            </td>
            <td>{percent(row['roc_auc'])}</td>
          </tr>
        """
        for row in rows
    )
    matrix = test["confusion_matrix"]
    largest = max(1, max(max(row) for row in matrix))
    matrix_rows = "".join(
        f'<tr><th scope="row">Aktual {index}</th>'
        + "".join(
            f'<td style="--lab-heat: {.08 + .47 * (value / largest) ** .5:.3f}">{count(value)}</td>'
            for value in row
        )
        + "</tr>"
        for index, row in enumerate(matrix)
    )
    st.html(f"""
      <div class="lab-eval">
        <section class="lab-eval-lead" aria-labelledby="lab-eval-model">
          <div>
            <p class="lab-eval-label">Model terpilih</p>
            <h2 id="lab-eval-model">{escape(selected_model)}</h2>
            <p>Average Precision tertinggi pada data validasi. Kinerja akhir diukur pada
              {count(metrics['split']['test'])} rekaman uji.</p>
          </div>
          <dl class="lab-eval-primary" aria-label="Metrik utama pada data uji">
            {primary_metrics}
          </dl>
        </section>
        <div class="lab-eval-detail">
          <section aria-labelledby="lab-eval-comparison">
            <h2 id="lab-eval-comparison">Perbandingan model</h2>
            <p>Data validasi · Skala batang 0–{percent(chart_max, digits=0)}</p>
            <table class="lab-eval-ranking" aria-labelledby="lab-eval-comparison">
              <thead>
                <tr><th scope="col">Model</th><th scope="col">Average Precision</th><th scope="col">ROC AUC</th></tr>
              </thead>
              <tbody>{comparison_rows}</tbody>
            </table>
            <p class="lab-eval-caption">Average Precision merangkum presisi dan recall pada berbagai ambang skor.</p>
          </section>
          <section aria-labelledby="lab-eval-confusion">
            <h2 id="lab-eval-confusion">Confusion matrix</h2>
            <table class="lab-eval-matrix" aria-labelledby="lab-eval-confusion">
              <thead>
                <tr><th scope="col">Kelas</th><th scope="col">Prediksi 0</th><th scope="col">Prediksi 1</th></tr>
              </thead>
              <tbody>{matrix_rows}</tbody>
            </table>
            <p class="lab-eval-caption">Warna yang lebih pekat menunjukkan jumlah yang lebih besar.</p>
          </section>
        </div>
        <section class="lab-eval-more" aria-labelledby="lab-eval-additional">
          <h2 id="lab-eval-additional">Metrik tambahan pada data uji</h2>
          <dl>{secondary_metrics}</dl>
        </section>
      </div>
    """)


def show_about(metrics: dict) -> None:
    dataset = metrics["dataset"]
    split = metrics["split"]
    split_total = sum(split.values())
    groups = (
        ("train", "Latih", "Melatih model"),
        ("validation", "Validasi", "Memilih model dan ambang"),
        ("test", "Uji", "Mengukur kinerja akhir"),
    )
    split_bar = "".join(
        f'<span data-split="{key}" style="flex: {split[key] / split_total:.6f}"></span>'
        for key, _, _ in groups
    )
    split_legend = "".join(
        f"""
          <div data-split="{key}">
            <dt>{label}</dt>
            <dd><strong>{count(split[key])}</strong> <span>{percent(split[key] / split_total, digits=0)}</span></dd>
            <dd>{purpose}</dd>
          </div>
        """
        for key, label, purpose in groups
    )

    features = (
        ("I", "Jumlah CD4", "Jumlah sel CD4 per mikroliter pada saat pengamatan."),
        ("II", "Viral load", "Jumlah salinan HIV per mililiter pada saat pengamatan."),
        ("III", "Koinfeksi TB", "Status koinfeksi tuberkulosis, ya atau tidak."),
        ("IV", "Mutasi resistansi obat", "Status mutasi terkait resistansi obat, ya atau tidak."),
    )
    feature_rows = "".join(
        f"""
          <li>
            <span class="lab-dataset-index" aria-hidden="true">{index}</span>
            <div><h3>{name}</h3><p>{description}</p></div>
          </li>
        """
        for index, name, description in features
    )

    st.html(f"""
      <div class="lab-dataset">
        <div class="lab-dataset-overview">
          <section class="lab-dataset-profile" aria-labelledby="lab-data-profile">
            <span class="lab-dataset-chip">Data sintetis</span>
            <h2 id="lab-data-profile">HIV/AIDS Clinical Global Dataset</h2>
            <dl class="lab-dataset-facts">
              <div><dt>Rekaman</dt><dd>{count(dataset["rows"])}</dd></div>
              <div><dt>Kolom</dt><dd>{count(dataset["columns"])}</dd></div>
              <div><dt>Label kelas 1</dt><dd>{percent(dataset["positive_rate"], digits=2)}</dd></div>
            </dl>
          </section>
          <section class="lab-dataset-partition" aria-labelledby="lab-data-partition">
            <h2 id="lab-data-partition">Pembagian dataset</h2>
            <div class="lab-dataset-split" aria-hidden="true">{split_bar}</div>
            <dl class="lab-dataset-legend">{split_legend}</dl>
          </section>
        </div>

        <section class="lab-dataset-features" aria-labelledby="lab-data-features">
          <header>
            <h2 id="lab-data-features">Empat fitur model</h2>
            <p>Empat pengukuran dan status klinis dari satu profil pada saat pengamatan.</p>
          </header>
          <ol class="lab-dataset-register">{feature_rows}</ol>
        </section>

        <section class="lab-dataset-target" aria-labelledby="lab-data-target">
          <h2 id="lab-data-target">Label hasil</h2>
          <dl class="lab-dataset-classes">
            <div><dt>Kelas 0</dt><dd>Tanpa mortalitas dalam lima tahun</dd></div>
            <div><dt>Kelas 1</dt><dd>Mortalitas dalam lima tahun</dd></div>
          </dl>
        </section>

        <footer class="lab-dataset-sources">
          <a href="{DATASET_URL}" target="_blank" rel="noopener noreferrer">Dataset Kaggle <span aria-hidden="true">↗</span></a>
        </footer>
      </div>
    """)


def load_experiment():
    model_path = ARTIFACTS / "model.joblib"
    metrics_path = ARTIFACTS / "metrics.json"
    try:
        artifact_version = model_path.stat().st_mtime_ns, metrics_path.stat().st_mtime_ns
        metrics = load_metrics(metrics_path, artifact_version[1])
        model = load_model(model_path, artifact_version[0], metrics["model_sha256"])
    except (OSError, EOFError, KeyError, ValueError, ModuleNotFoundError):
        st.error("Artefak model belum tersedia atau tidak dapat dibuka. Jalankan pelatihan dari folder proyek.")
        st.code("python train.py", language="bash")
        st.stop()
    return model, metrics, artifact_version


def simulation_page() -> None:
    model, metrics, artifact_version = load_experiment()
    dataset = metrics["dataset"]
    relief_name = "simulation-relief.webp"
    relief_art = asset_data(relief_name, (ASSETS / relief_name).stat().st_mtime_ns)
    st.html(f"""
        <section class="lab-intro" id="ruang-lab" aria-labelledby="ruang-lab-title">
          <img class="lab-intro-relief" src="data:image/webp;base64,{relief_art}"
               alt="" width="1752" height="898" loading="lazy">
          <div class="lab-intro-content">
            <h1 id="ruang-lab-title">Ruang <em>simulasi</em></h1>
            <dl class="lab-summary" aria-label="Ringkasan data">
              <div><dt>Rekaman sintetis</dt><dd>{count(dataset['rows'])}</dd></div>
              <div><dt>Fitur model</dt><dd>{len(metrics['features'])}</dd></div>
              <div><dt>Label kelas 1</dt><dd>{percent(dataset['positive_rate'], digits=2)}</dd></div>
            </dl>
          </div>
        </section>
    """)

    with st.container(key="lab_workbench"):
        show_simulation(model, metrics, artifact_version)


def evaluation_page() -> None:
    st.html(ASSETS / "evaluation.css")
    _, metrics, _ = load_experiment()
    with st.container(key="lab_insights"):
        st.title("Evaluasi model")
        show_evaluation(metrics)


def data_page() -> None:
    st.html(ASSETS / "dataset.css")
    _, metrics, _ = load_experiment()
    with st.container(key="lab_insights"):
        st.title("Tentang data")
        show_about(metrics)


def main() -> None:
    st.set_page_config(page_title="Asklepios · HIV Outcomes Lab", layout="wide", initial_sidebar_state="auto")
    st.html(ASSETS / "lab.css")
    st.html(ASSETS / "dashboard.css")
    st.html(ASSETS / "simulation-workspace.css")

    pages = [
        st.Page(simulation_page, title="Simulasi", icon=":material/science:", default=True),
        st.Page(evaluation_page, title="Evaluasi model", icon=":material/monitoring:", url_path="evaluasi"),
        st.Page(data_page, title="Tentang data", icon=":material/database:", url_path="data"),
        st.Page(show_cover, title="Beranda", icon=":material/home:", url_path="beranda"),
    ]
    current_page = st.navigation(pages, position="hidden")
    with st.sidebar:
        st.html('<div class="lab-dashboard-title">HIV Outcomes <em>Lab</em></div>')
        for page in pages:
            st.page_link(page, width="stretch")

    with st.container(key="lab_dashboard"):
        current_page.run()


if __name__ == "__main__":
    main()
