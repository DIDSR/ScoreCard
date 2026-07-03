import os
import cv2
import glob
import uuid
import shutil
import threading
import json
import warnings
import matplotlib

import matplotlib.pyplot   as plt
import numpy               as np
import pandas              as pd


from   copy                import deepcopy
from   werkzeug.utils      import secure_filename
from   src.generic_predict import produce_output

from   flask               import (
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
from   src.utils           import (
                                   read_png,
                                   create_image_df,
                                  )

from   src.analyses        import (
                                   hist_analysis,
                                   coverage_analysis,
                                   congruence_analysis,
                                   completeness_analysis,
                                   consistency_analysis,
                                   constraint_patch_analysis,
                                  )
from   src.analysis_registry import (
                                   ANALYSIS_REGISTRY,
                                   discover_schema,
                                   default_config,
                                   enabled_flags,
                                   parse_config_from_form,
                                   feature_options_for_analysis,
                                  )

warnings.filterwarnings("ignore")
matplotlib.use('Agg')

app            = Flask(__name__,
                       template_folder='flask_files/templates',
                       static_folder='flask_files/static'
                      )
app.secret_key = 'mammoqc_secret_key'

# Configuration
BASE_DIR                  = os.getcwd()
UPLOAD_FOLDER             = os.path.join(BASE_DIR, 'flask_files', 'static', 'uploads')
OUTPUT_FOLDER             = os.path.join(BASE_DIR, 'flask_files', 'static', 'outputs')
TMP_DATA_DIR              = os.path.join(BASE_DIR, 'data', 'tmp')
PRESET_REAL_CSV           = os.path.join(BASE_DIR, 'data', 'real_patch_appearance.csv')
PRESET_SYNTH_CSV          = os.path.join(BASE_DIR, 'data', 'kde_patch_appearance.csv')
PRESET_METADATA_REAL_CSV  = os.path.join(BASE_DIR, 'data', 'real_patch_appearance_with_metadata.csv')
PRESET_METADATA_SYNTH_CSV = os.path.join(BASE_DIR, 'data', 'kde_patch_appearance_with_metadata.csv')
USER_REAL_CSV             = os.path.join(BASE_DIR, 'data', 'real_patch_appearance.csv')
USER_SYNTH_CSV            = os.path.join(BASE_DIR, 'data', 'kde_patch_appearance.csv')
USER_METADATA_REAL_CSV    = os.path.join(BASE_DIR, 'data', 'uploaded_metadata_real.csv')
USER_METADATA_SYNTH_CSV   = os.path.join(BASE_DIR, 'data', 'uploaded_metadata_synth.csv')

NO_IMAGES_MESSAGE         = "No images available, utilizing default CSVs"
FEATURES_DIR              = './data/features'

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

    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)

            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)

            except Exception as e:
                print(f"Error cleaning {file_path}: {e}")

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

def _save_uploaded_csv(file_storage, dest_path):
    """Save an uploaded CSV/JSON metadata file and return its path."""
    filename = secure_filename(file_storage.filename)
    if not filename:
        return None
    ext = os.path.splitext(filename)[1].lower()

    if ext not in {'.csv', '.json'}:
        raise ValueError(f"Unsupported metadata file type: {ext or '(none)'}")
    file_storage.save(dest_path)
    return dest_path


def _resolve_metadata_csvs(real_csv, synth_csv, is_preset=False):
    """
    Determine metadata CSV paths for Completeness and Consistency.

    Preset loads dedicated metadata-rich CSVs. Custom uploads may provide
    separate metadata files; otherwise patch CSVs are used as a fallback.
    """
    if is_preset:
        return PRESET_METADATA_REAL_CSV, PRESET_METADATA_SYNTH_CSV

    metadata_real   = None
    metadata_synth  = None
    real_meta_file  = request.files.get('real_metadata')
    synth_meta_file = request.files.get('synth_metadata')

    if real_meta_file and real_meta_file.filename:
        metadata_real  = _save_uploaded_csv(real_meta_file, USER_METADATA_REAL_CSV)

    if synth_meta_file and synth_meta_file.filename:
        metadata_synth = _save_uploaded_csv(synth_meta_file, USER_METADATA_SYNTH_CSV)

    return metadata_real or real_csv, metadata_synth or synth_csv


def _discover_dataset_schema(real_csv, synth_csv, metadata_real_csv=None, metadata_synth_csv=None):
    meta_real  = metadata_real_csv  or session.get('metadata_real_csv')  or real_csv
    meta_synth = metadata_synth_csv or session.get('metadata_synth_csv') or synth_csv

    return discover_schema(
        real_csv,
        synth_csv,
        FEATURES_DIR,
        metadata_real_csv  = meta_real,
        metadata_synth_csv = meta_synth,
    )


def _images_available(df):
    """Return True when the dataframe has at least one readable image filepath."""
    if df is None or df.empty or 'filepath' not in df.columns:
        return False

    return df['filepath'].astype(str).apply(os.path.isfile).any()

def _prepare_dataset_preview(real_csv, synth_csv, n=5):
    """
    Load CSV datasets and generate preview thumbnails when image files exist.

    If images are missing or preview generation fails, return empty previews with
    a user-facing message while keeping the CSV datasets ready for analysis.
    """
    real_df  = pd.read_csv(real_csv)
    synth_df = pd.read_csv(synth_csv)

    fallback = {
        'real_df'             : real_df,
        'synth_df'            : synth_df,
        'real_preview_images' : [],
        'synth_preview_images': [],
        'images_available'    : False,
        'preview_message'     : NO_IMAGES_MESSAGE,
    }

    if not _images_available(real_df) and not _images_available(synth_df):
        return fallback

    try:
        real_previews, synth_previews = generate_previews(
            real_df, n=n, prefix="real", synth_df=synth_df
        )

        if not real_previews:
            return fallback

        paired_real  = []
        paired_synth = []

        for real_img, synth_img in zip(real_previews, synth_previews):
            if real_img and synth_img:
                paired_real.append(real_img)
                paired_synth.append(synth_img)

        if paired_real:
            real_previews, synth_previews = paired_real, paired_synth

        return {
            'real_df'             : real_df,
            'synth_df'            : synth_df,
            'real_preview_images' : real_previews,
            'synth_preview_images': synth_previews,
            'images_available'    : True,
            'preview_message'     : None,
        }

    except Exception as exc:
        print(f"Preview generation skipped: {exc}")
        return fallback


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
        img_path     = row['filepath']
        real_stem    = os.path.splitext(os.path.basename(str(img_path)))[0]
        img_array    = read_png(img_path)

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

def _render_index_page(dataset_ready=False, preview=None, schema=None, analysis_config=None):
    ctx = {
        'dataset_ready'               : dataset_ready,
        'analysis_registry'           : ANALYSIS_REGISTRY,
        'schema'                      : schema or {},
        'analysis_config'             : analysis_config or {},
        'feature_options_for_analysis': feature_options_for_analysis,
    }

    if dataset_ready and preview:
        ctx.update({
            'real_preview_images' : preview['real_preview_images'],
            'synth_preview_images': preview['synth_preview_images'],
            'images_available'    : preview['images_available'],
            'preview_message'     : preview['preview_message'],
            'total_images'        : len(preview['real_df']),
            'total_synth_images'  : len(preview['synth_df']),
        })

    return render_template('index.html', **ctx)


@app.route('/')
def index():
    cleanup_tmp()

    return _render_index_page()

@app.route('/upload', methods=['POST'])
def upload():
    upload_type = request.form.get('upload_type')
    is_preset   = (upload_type == 'preset')

    if is_preset:
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

    try:
        metadata_real_csv, metadata_synth_csv = _resolve_metadata_csvs(
            real_csv, 
            synth_csv, 
            is_preset = is_preset
        )
    except ValueError as e:
        flash(str(e), 'error')

        return redirect(url_for('index'))

    session['real_csv']           = real_csv
    session['synth_csv']          = synth_csv
    session['metadata_real_csv']  = metadata_real_csv
    session['metadata_synth_csv'] = metadata_synth_csv
    session['is_preset']          = is_preset

    try:
        preview                    = _prepare_dataset_preview(real_csv, synth_csv, n=5)
        schema                     = _discover_dataset_schema(real_csv, synth_csv, metadata_real_csv, metadata_synth_csv)
        session['analysis_config'] = default_config(schema)
        session['dataset_schema']  = schema

        return _render_index_page(
            dataset_ready=True,
            preview=preview,
            schema=schema,
            analysis_config=session['analysis_config'],
        )

    except Exception as e:
        flash(f'Error loading datasets: {str(e)}', 'error')

        return redirect(url_for('index'))


@app.route('/configure', methods=['POST'])
def configure():
    if not session.get('real_csv') or not session.get('synth_csv'):
        flash('Load datasets before configuring analyses.', 'error')
        return redirect(url_for('index'))

    real_csv  = session['real_csv']
    synth_csv = session['synth_csv']
    schema    = session.get('dataset_schema') or _discover_dataset_schema(real_csv, synth_csv)

    parsed, errors = parse_config_from_form(
        request.form,
        schema,
        existing_config=session.get('analysis_config'),
    )

    if errors:
        for err in errors:
            flash(err, 'error')
        preview = _prepare_dataset_preview(real_csv, synth_csv, n=5)
        return _render_index_page(
            dataset_ready=True,
            preview=preview,
            schema=schema,
            analysis_config=parsed,
        )

    session['analysis_config'] = parsed
    flash('Analysis configuration saved.', 'success')

    preview = _prepare_dataset_preview(real_csv, synth_csv, n=5)
    return _render_index_page(
        dataset_ready  = True,
        preview        = preview,
        schema         = schema,
        analysis_config= parsed,
    )


def _save_fig_dict(figs, out_dir, prefix):
    """Save a dict of {metric: fig} as individual PNGs."""
    if not figs:
        return
    if isinstance(figs, dict):
        for metric, fig in figs.items():
            safe = str(metric).replace(" ", "_").replace("/", "-").replace("%", "pct")
            path = os.path.join(out_dir, f"{prefix}_{safe}.png")

            try:
                fig.savefig(path, bbox_inches="tight", pad_inches=0.12)

            except Exception as e:
                print(f"[Job] Failed to save {path}: {e}")
    else:
        try:
            figs.savefig(
                os.path.join(out_dir, f"{prefix}.png"),
                bbox_inches="tight",
                pad_inches=0.12,
            )
        except Exception:
            pass


def _list_per_metric_images(job_output_dir, prefix):
    pattern = os.path.join(job_output_dir, f"{prefix}_*.png")
    files   = sorted(glob.glob(pattern))
    rel_dir = os.path.relpath(job_output_dir, OUTPUT_FOLDER)
    items   = []

    for f in files:
        fname    = os.path.basename(f)
        rel_path = os.path.join(rel_dir, fname).replace("\\", "/")
        base     = fname.replace(f"{prefix}_", "").replace(".png", "").replace("_", " ")

        items.append((base, rel_path))

    return items


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

    metadata_real_csv  = session.get('metadata_real_csv') or real_csv
    metadata_synth_csv = session.get('metadata_synth_csv') or synth_csv

    if session.get('analysis_config'):
        analysis_config = deepcopy(session['analysis_config'])
    else:
        schema = _discover_dataset_schema(real_csv, synth_csv, metadata_real_csv, metadata_synth_csv)
        analysis_config = default_config(schema)

    config_flags = enabled_flags(analysis_config)

    with progress_lock:
        progress_store[job_id] = {
            'status'         : 'processing',
            'progress'       : 0,
            'error'          : None,
            'config'         : config_flags,
            'analysis_config': analysis_config,
        }

    def run_inference(real_csv, synth_csv, metadata_real_csv, metadata_synth_csv, is_preset, cfg):
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

            metrics_data = None

            if cfg.get('histogram', {}).get('enabled'):
                histo_fig, metrics_data = hist_analysis(
                    FEATURES_DIR,
                    feature_names=cfg['histogram'].get('features'),
                )
                histo_fig.savefig(
                    os.path.join(job_output_dir, 'histo_fig.png'),
                    bbox_inches="tight",
                    pad_inches=0.12,
                )

            if cfg.get('coverage', {}).get('enabled'):
                coverage_figs = coverage_analysis(
                    FEATURES_DIR,
                    feature_names=cfg['coverage'].get('features'),
                    metrics_to_compute=cfg['coverage'].get('metrics'),
                )
                _save_fig_dict(coverage_figs, job_output_dir, prefix="coverage")

            if cfg.get('congruence', {}).get('enabled'):
                congruence_figs = congruence_analysis(
                    cfg['congruence']['metrics'],
                    results_dir=FEATURES_DIR,
                    feature_names=cfg['congruence'].get('features'),
                )
                _save_fig_dict(congruence_figs, job_output_dir, prefix="congruence")

            if cfg.get('constraint', {}).get('enabled'):
                violation_df, constraint_fig = constraint_patch_analysis(
                    FEATURES_DIR,
                    features_to_check=cfg['constraint'].get('features'),
                )
                if constraint_fig is not None:
                    constraint_fig.savefig(
                        os.path.join(job_output_dir, 'constraint.png'),
                        bbox_inches="tight",
                        pad_inches=0.12,
                    )
                if violation_df is not None and not violation_df.empty:
                    violation_df.to_json(
                        os.path.join(job_output_dir, 'constraint.json'),
                        orient='records',
                    )

            if cfg.get('consistency', {}).get('enabled'):
                try:
                    cons_cfg = cfg['consistency']
                    cons_df, cons_figs = consistency_analysis(
                        metadata_real_csv,
                        metadata_synth_csv,
                        group_by=cons_cfg.get('group_by', 'Race'),
                        metric_cols=cons_cfg.get('fields'),
                        label="report",
                        metrics_to_plot=cons_cfg.get('metrics'),
                    )
                    prefix = f"consistency_{cons_cfg.get('group_by', 'group').replace(' ', '_')}"
                    _save_fig_dict(cons_figs, job_output_dir, prefix=prefix)
                except Exception as e:
                    print(f"[Job {job_id}] Consistency analysis skipped or failed: {e}")

            if cfg.get('completeness', {}).get('enabled'):
                try:
                    comp_cfg = cfg['completeness']
                    comp_df, comp_figs = completeness_analysis(
                        metadata_real_csv,
                        metadata_synth_csv,
                        required_fields=comp_cfg.get('fields'),
                        label="report",
                        metrics_to_include=comp_cfg.get('metrics'),
                    )
                    _save_fig_dict(comp_figs, job_output_dir, prefix="completeness")

                    if comp_df is not None and not comp_df.empty:
                        comp_df.to_json(
                            os.path.join(job_output_dir, 'completeness.json'),
                            orient='records',
                        )
                except Exception as e:
                    print(f"[Job {job_id}] Completeness analysis skipped or failed: {e}")

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
                progress_store[job_id].update({
                    'status'    : 'completed',
                    'progress'  : 100,
                    'output_dir': job_id
                })

        except Exception as e:
            print(f"[Job {job_id}] Error: {e}")

            with progress_lock:
                progress_store[job_id].update({
                                               'status'  : 'error',
                                               'progress': 0,
                                               'error'   : str(e)
                                              })

    is_preset = session.get('is_preset', False)
    thread = threading.Thread(
        target=run_inference,
        args=(
            real_csv,
            synth_csv,
            metadata_real_csv,
            metadata_synth_csv,
            is_preset,
            analysis_config,
        ),
    )
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

    config            = job.get('config', {})

    histo_fig    = f"{job_id}/histo_fig.png"
    metrics_data = None

    metrics_json_path = os.path.join(OUTPUT_FOLDER, job_id, 'metrics.json')

    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, 'r') as f:
            metrics_data = json.load(f)

    return render_template('results.html',
                           job_id       = job_id,
                           histo_fig    = histo_fig,
                           metrics_data = metrics_data,
                           config       = config
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

    job_output_dir = os.path.join(OUTPUT_FOLDER, job_id)
    coverage_images = _list_per_metric_images(job_output_dir, "coverage")

    return render_template('results_coverage.html',
                           job_id=job_id,
                           coverage_images=coverage_images
                           )

@app.route('/results/<job_id>/congruence')
def results_congruence(job_id):
    with progress_lock:
        job = progress_store.get(job_id)

    if not job or job['status'] != 'completed':
        flash('Results not ready yet.', 'error')
        return redirect(url_for('index'))

    job_output_dir = os.path.join(OUTPUT_FOLDER, job_id)
    congruence_images = _list_per_metric_images(job_output_dir, "congruence")

    return render_template('results_congruence.html',
                           job_id=job_id,
                           congruence_images=congruence_images
                           )

@app.route('/results/<job_id>/completeness')
def results_completeness(job_id):
    with progress_lock:
        job = progress_store.get(job_id)

    if not job or job['status'] != 'completed':
        flash('Results not ready yet.', 'error')
        return redirect(url_for('index'))

    job_output_dir = os.path.join(OUTPUT_FOLDER, job_id)
    completeness_images = _list_per_metric_images(job_output_dir, "completeness")

    return render_template('results_completeness.html',
                           job_id=job_id,
                           completeness_images=completeness_images
                           )


@app.route('/results/<job_id>/constraint')
def results_constraint(job_id):
    with progress_lock:
        job = progress_store.get(job_id)

    if not job or job['status'] != 'completed':
        flash('Results not ready yet.', 'error')

        return redirect(url_for('index'))

    constraint_fig       = f"{job_id}/constraint.png"
    constraint_json_path = os.path.join(OUTPUT_FOLDER, job_id, 'constraint.json')
    violation_data       = None

    if os.path.exists(constraint_json_path):
        with open(constraint_json_path, 'r') as f:
            violation_data = json.load(f)

    return render_template(
        'results_constraint.html',
        job_id         = job_id,
        constraint_fig = constraint_fig,
        violation_data = violation_data,
    )


@app.route('/results/<job_id>/consistency')
def results_consistency(job_id):
    with progress_lock:
        job = progress_store.get(job_id)

    if not job or job['status'] != 'completed':
        flash('Results not ready yet.', 'error')
        return redirect(url_for('index'))

    job_output_dir     = os.path.join(OUTPUT_FOLDER, job_id)
    consistency_images = []

    for f in sorted(glob.glob(os.path.join(job_output_dir, "consistency_*.png"))):
        fname    = os.path.basename(f)
        rel_path = os.path.join(job_id, fname).replace("\\", "/")
        title    = fname.replace("consistency_", "").replace(".png", "").replace("_", " ").title()

        consistency_images.append((title, rel_path))

    return render_template(
        'results_consistency.html',
        job_id             = job_id,
        consistency_images = consistency_images,
    )

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