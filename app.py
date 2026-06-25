import os
import cv2
import uuid
import shutil
import threading
import json
import warnings
import matplotlib

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd



from werkzeug.utils      import secure_filename
from src.generic_predict import produce_output

from flask               import (
                                 Flask,
                                 render_template,
                                 request,
                                 redirect,
                                 url_for,
                                 flash,
                                 send_from_directory,
                                 jsonify,
                                 session
                                )
from src.utils           import (
                                 read_png,
                                 create_image_df,
                                 hist_analysis,
                                 coverage_analysis,
                                 congruence_analysis,
                                 completeness_analysis,
                                 consistency_analysis
                                )

warnings.filterwarnings("ignore")
matplotlib.use('Agg')

app            = Flask(__name__,
                       template_folder='flask_files/templates',
                       static_folder='flask_files/static'
                      )
app.secret_key = 'mammoqc_secret_key'

# Configuration
BASE_DIR         = os.getcwd()
UPLOAD_FOLDER    = os.path.join(BASE_DIR, 'flask_files', 'static', 'uploads')
OUTPUT_FOLDER    = os.path.join(BASE_DIR, 'flask_files', 'static', 'outputs')
TMP_DATA_DIR     = os.path.join(BASE_DIR, 'data', 'tmp')
PRESET_REAL_CSV  = os.path.join(BASE_DIR, 'data', 'real_patch_appearance.csv')
PRESET_SYNTH_CSV = os.path.join(BASE_DIR, 'data', 'kde_patch_appearance.csv')
USER_REAL_CSV    = os.path.join(BASE_DIR, 'data', 'real_patch_appearance.csv')
USER_SYNTH_CSV   = os.path.join(BASE_DIR, 'data', 'kde_patch_appearance.csv')


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TMP_DATA_DIR,  exist_ok=True)

# =============================================
# Progress tracking
# =============================================
progress_store = {}
progress_lock  = threading.Lock()


def cleanup_tmp():
    """Clean up temporary input files, preview images, and job outputs."""
    # Clear uploaded input files
    if os.path.exists(TMP_DATA_DIR):
        for filename in os.listdir(TMP_DATA_DIR):
            file_path = os.path.join(TMP_DATA_DIR, filename)

            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning {file_path}: {e}")

    # Clear preview images
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)

            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning {file_path}: {e}")

    # Clear job output directories
    if os.path.exists(OUTPUT_FOLDER):
        for item in os.listdir(OUTPUT_FOLDER):
            item_path = os.path.join(OUTPUT_FOLDER, item)

            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                elif os.path.isfile(item_path):
                    os.remove(item_path)
            except Exception as e:
                print(f"Error cleaning {item_path}: {e}")

def generate_previews(df, n=4, prefix="img", synth_df=None):
    """
    Generate resized preview images.

    If synth_df is provided → returns two aligned lists: (real_previews, synth_previews)
    Otherwise → returns single list (original behavior)
    """
    samples        = df.sample(n=min(n, len(df)))
    real_previews  = []
    synth_previews = []

    for idx, row in samples.iterrows():
        img_path  = row['filepath']
        real_stem = os.path.splitext(os.path.basename(str(img_path)))[0]

        img_array = read_png(img_path)

        target_size  = 250
        h, w         = img_array.shape[:2]
        scale        = target_size / max(h, w)
        img_resized  = cv2.resize(img_array,
                                 (int(w * scale), int(h * scale)),
                                 interpolation = cv2.INTER_LINEAR
                                 )

        preview_name = f"{prefix}_preview_{uuid.uuid4().hex}.png"
        preview_path = os.path.join(UPLOAD_FOLDER, preview_name)

        plt.imsave(preview_path, img_resized)
        real_previews.append(preview_name)

        if synth_df is not None:
            mask     = synth_df['filepath'].astype(str).str.contains(real_stem, regex=False, na=False)
            matching = synth_df[mask]

            if len(matching) > 0:
                synth_row  = matching.sample(1).iloc[0]
                synth_path = synth_row['filepath']

                s_img = read_png(synth_path)

                sh, sw    = s_img.shape[:2]
                s_scale   = target_size / max(sh, sw)
                s_resized = cv2.resize(s_img,
                                       (int(sw * s_scale),
                                       int(sh * s_scale)),
                                       interpolation=cv2.INTER_LINEAR
                                       )

                s_name    = f"synth_preview_{uuid.uuid4().hex}.png"
                s_path    = os.path.join(UPLOAD_FOLDER, s_name)

                plt.imsave(s_path, s_resized)
                synth_previews.append(s_name)
            else:
                synth_previews.append(None)

    if synth_df is not None:
        print(f"DEBUG >>> {prefix}: Generated {len(real_previews)} paired previews")
        return real_previews, synth_previews
    else:
        print(f"DEBUG >>> {prefix}: Sampled   {len(samples)} rows, appended {len(real_previews)} previews")
        return real_previews

@app.route('/')
def index():
    cleanup_tmp()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    upload_type = request.form.get('upload_type')

    if upload_type == 'preset':
        real_csv  = PRESET_REAL_CSV
        synth_csv = PRESET_SYNTH_CSV

    elif upload_type == 'directory':
        dir_path = request.form.get('dir_path')

        if not dir_path or not os.path.isdir(dir_path):
            flash('Invalid directory path provided.', 'error')
            return redirect(url_for('index'))

        for f in os.listdir(dir_path):
            if f.lower().endswith('.dcm') or f.lower().endswith('.dicom') or f.lower().endswith('.png'):
                shutil.copy2(os.path.join(dir_path, f), os.path.join(TMP_DATA_DIR, f))

        create_image_df(TMP_DATA_DIR)
        real_csv  = USER_REAL_CSV
        synth_csv = USER_SYNTH_CSV

    elif upload_type == 'files':
        files = request.files.getlist('dicom_files')

        if not files or files[0].filename == '':
            flash('No files selected.', 'error')
            return redirect(url_for('index'))

        for file in files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(TMP_DATA_DIR, filename))

        create_image_df(TMP_DATA_DIR)
        real_csv  = USER_REAL_CSV
        synth_csv = USER_SYNTH_CSV

    else:
        flash('Invalid upload type.', 'error')
        return redirect(url_for('index'))

    session['real_csv']  = real_csv
    session['synth_csv'] = synth_csv
    session['is_preset'] = (upload_type == 'preset')

    try:
        real_df                                   = pd.read_csv(real_csv)
        synth_df                                  = pd.read_csv(synth_csv)
        real_preview_images, synth_preview_images = generate_previews(real_df, n=5, prefix="real", synth_df=synth_df)

        return render_template(
                               'index.html',
                               real_preview_images  = real_preview_images,
                               synth_preview_images = synth_preview_images,
                               dataset_ready        = True,
                               total_images         = len(real_df),
                               total_synth_images   = len(synth_df)
                              )

    except Exception as e:
        flash(f'Error generating preview: {str(e)}', 'error')

        return redirect(url_for('index'))


@app.route('/generate_report', methods=['POST'])
def generate_report():
    real_csv  = session['real_csv']
    synth_csv = session['synth_csv']

    if not real_csv:
        flash('No real dataset selected. Please upload images first.', 'error')
        return redirect(url_for('index'))

    elif not synth_csv:
        flash('No synthetic dataset selected. Please upload images first.', 'error')
        return redirect(url_for('index'))

    job_id = str(uuid.uuid4())

    with progress_lock:
        progress_store[job_id] = {'status': 'processing', 'progress': 0, 'error': None}

    def run_inference(real_csv, synth_csv, is_preset):
        try:
            # ----------------------------------------------------------
            # 1. Run inference
            # ----------------------------------------------------------
            job_output_dir = os.path.join(OUTPUT_FOLDER, job_id)

            def inference_progress_callback(fraction_done):
                """
                Called by generate_predictions after every batch.
                fraction_done: float in [0.0, 1.0]
                """
                pct = int(fraction_done * 100)
                with progress_lock:
                    progress_store[job_id]['progress'] = pct

            produce_output( real_csv          = real_csv,
                            synth_csv         = synth_csv,
                            output_dir        = job_output_dir,
                            progress_callback = inference_progress_callback
                           )

            # ----------------------------------------------------------
            # 2. Generate report/output/metrics
            # ----------------------------------------------------------

            # fig, metrics_data = results_analysis(job_output_dir)
            histo_fig, metrics_data = hist_analysis('./data/features')
            histo_fig.savefig(os.path.join(job_output_dir, 'histo_fig.png'))

            coverage_fig            = coverage_analysis('./data/features')
            coverage_fig.savefig(os.path.join(job_output_dir, 'coverage_fig.png'))

            congruence_fig          = congruence_analysis('./data/features')
            congruence_fig.savefig(os.path.join(job_output_dir, 'congruence_fig.png'))

            try:
                cons_df_race, cons_fig_race = consistency_analysis(
                                                                    real_meta_csv,
                                                                    group_by="Race",
                                                                    label="report"
                                                                  )
                if cons_fig_race:
                    cons_fig_race.savefig(os.path.join(job_output_dir, 'consistency_race_fig.png'))

                if comp_df is not None and not comp_df.empty:
                    comp_df.to_json(os.path.join(job_output_dir,      'completeness.json'),         orient='records')
                if cons_df_hosp is not None and not cons_df_hosp.empty:
                    cons_df_hosp.to_json(os.path.join(job_output_dir, 'consistency_race.json'), orient='records')

            except Exception as e:
                print(f"[Job {job_id}] Completeness/Consistency analysis skipped or failed: {e}")

            plt.close('all')

            # ----------------------------------------------------------
            # 3. Save metrics as JSON
            # ----------------------------------------------------------
            if metrics_data:
                serializable_metrics = {}

                for k, v in metrics_data.items():
                    if isinstance(v, np.ndarray):
                        serializable_metrics[k] = v.tolist()
                    else:
                        serializable_metrics[k] = v

                with open(os.path.join(job_output_dir, 'metrics.json'), 'w') as f:
                    json.dump(serializable_metrics, f)

            with progress_lock:
                progress_store[job_id] = {
                                          'status': 'completed',
                                          'progress': 100,
                                          'output_dir': job_id
                                          }

        except Exception as e:
            print(f"[Job {job_id}] Error: {e}")

            with progress_lock:
                progress_store[job_id] = {
                                          'status': 'error',
                                          'progress': 0,
                                          'error': str(e)
                                         }

    is_preset = session.get('is_preset', False)
    thread    = threading.Thread(target=run_inference, args=(real_csv, synth_csv, is_preset))
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/progress/<job_id>')
def progress(job_id):
    with progress_lock:
        return jsonify(progress_store.get(job_id, {'status': 'not_found'}))


@app.route('/results/<job_id>')
def results(job_id):
    with progress_lock:
        job = progress_store.get(job_id)

    if not job or job['status'] != 'completed':
        return redirect(url_for('index'))

    histo_fig         = f"{job_id}/histo_fig.png"
    coverage_fig      = f"{job_id}/coverage_fig.png"
    congruence_fig    = f"{job_id}/congruence_fig.png"

    metrics_data      = None

    metrics_json_path = os.path.join(OUTPUT_FOLDER, job_id, 'metrics.json')

    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, 'r') as f:
            metrics_data = json.load(f)

    return render_template('results.html',
                           job_id=job_id,
                           histo_fig=histo_fig,
                           coverage_fig=coverage_fig,
                           congruence_fig=congruence_fig,
                           metrics_data=metrics_data
                           )


@app.route('/results/<job_id>/histogram')
def results_histogram(job_id):
    with progress_lock:
        job = progress_store.get(job_id)

    if not job or job['status'] != 'completed':
        flash('Results not ready yet.', 'error')

        return redirect(url_for('index'))

    histo_fig = f"{job_id}/histo_fig.png"

    return render_template('results_histogram.html',
                           job_id=job_id,
                           histo_fig=histo_fig
                           )


@app.route('/results/<job_id>/coverage')
def results_coverage(job_id):
    with progress_lock:
        job = progress_store.get(job_id)

    if not job or job['status'] != 'completed':
        flash('Results not ready yet.', 'error')

        return redirect(url_for('index'))

    coverage_fig = f"{job_id}/coverage_fig.png"

    return render_template('results_coverage.html',
                           job_id=job_id,
                           coverage_fig=coverage_fig
                           )

@app.route('/results/<job_id>/congruence')
def results_congruence(job_id):
    with progress_lock:
        job = progress_store.get(job_id)

    if not job or job['status'] != 'completed':
        flash('Results not ready yet.', 'error')

        return redirect(url_for('index'))

    congruence_fig = f"{job_id}/congruence_fig.png"

    return render_template('results_congruence.html',
                           job_id         = job_id,
                           congruence_fig = congruence_fig
                           )

@app.route('/results/<job_id>/completeness')
def results_completeness(job_id):
    with progress_lock:
        job = progress_store.get(job_id)

    if not job or job['status'] != 'completed':
        flash('Results not ready yet.', 'error')

        return redirect(url_for('index'))

    completeness_fig = f"{job_id}/completeness_fig.png"

    return render_template('results_completeness.html',
                           job_id           = job_id,
                           completeness_fig = completeness_fig
                           )


@app.route('/results/<job_id>/consistency')
def results_consistency(job_id):
    with progress_lock:
        job = progress_store.get(job_id)

    if not job or job['status'] != 'completed':
        flash('Results not ready yet.', 'error')
        return redirect(url_for('index'))

    return render_template('results_consistency.html',
                           job_id                   = job_id,
                           consistency_race_fig    = f"{job_id}/consistency_race_fig.png",
                           consistency_vendor_fig  = f"{job_id}/consistency_vendor_fig.png")

@app.route('/download/<job_id>')
def download_results(job_id):
    """Zip the job output directory and send it as a downloadable file."""
    job_output_dir = os.path.join(OUTPUT_FOLDER, job_id)

    if not os.path.exists(job_output_dir):
        flash('Output files not found for this job.', 'error')
        return redirect(url_for('index'))

    zip_buffer = os.path.join(OUTPUT_FOLDER, f"{job_id}.zip")
    try:
        shutil.make_archive(
                            base_name = os.path.join(OUTPUT_FOLDER, job_id),
                            format    = 'zip',
                            root_dir  = OUTPUT_FOLDER,
                            base_dir  = job_id
                            )

        return send_from_directory(
                                   OUTPUT_FOLDER,
                                   f"{job_id}.zip",
                                   as_attachment = True,
                                   download_name = f"{job_id}.zip"
                                  )

    except Exception as e:
        flash(f'Error creating zip file: {str(e)}', 'error')

        return redirect(url_for('results', job_id=job_id))
    finally:
        if os.path.exists(zip_buffer):
            try:
                os.remove(zip_buffer)
            except Exception:
                pass


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/static/outputs/<path:filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050, use_reloader=False)
