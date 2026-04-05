import os
import gradio as gr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'grid.color':       '#cccccc',
})

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ─── PATHS ────────────────────────────────────────────────────────────────────
MODELS_DIR  = r'C:\Users\sanga\OneDrive\Documents\1_M.tech\SEM2\DL\Scaffold\mit_ecg\saved_models'
CLASS_NAMES = ['Normal(N)', 'PVC(V)', 'APB(A)', 'Paced(/)', 'Fusion(F)']
COLORS      = ['#2196F3', '#F44336', '#4CAF50', '#9C27B0', '#FF9800']
LIST_INPUT_MODELS = ['ae_encoder','ae_decoder']


# ─── Wrapper ──────────────────────────────────────────────────────────────────
class SavedModelWrapper:
    def __init__(self, model, is_list_input=False):
        self.model         = model
        self.is_list_input = is_list_input

    def predict(self, x, verbose=0):
        x_tensor = tf.constant(x, dtype=tf.float32)
        out      = (self.model.serve([x_tensor])
                    if self.is_list_input
                    else self.model.serve(x_tensor))
        return out.numpy()


# ─── Load ─────────────────────────────────────────────────────────────────────
def load(folder_name):
    path = os.path.join(MODELS_DIR, folder_name)
    print(f'  Loading: {folder_name} ... ', end='', flush=True)
    if not os.path.exists(path):
        print('❌ NOT FOUND')
        return None
    try:
        m = tf.saved_model.load(path)
        print('✅')
        return SavedModelWrapper(m, is_list_input=folder_name in LIST_INPUT_MODELS)
    except Exception as e:
        print(f'❌ {e}')
        return None


# ─── Load All ─────────────────────────────────────────────────────────────────
print('Loading models...')
classifiers = {}
for name, folder in {
    'MLP':            'mlp_model',
    'CNN':            'cnn_model',
    'RNN':            'rnn_model',
    'LSTM':           'lstm_model',
    'GRU':            'gru_model',
    'LSTM+Attention': 'attention_model',
    'MobileNet':      'clf_mobile',
    'ResNet':         'clf_resnet',
}.items():
    m = load(folder)
    if m is not None:
        classifiers[name] = m

ae_encoder    = load('ae_encoder')
ae_decoder    = load('ae_decoder')
gan_generator = load('gan_generator')
GAN_AVAILABLE = gan_generator is not None

print(f'\nLoaded : {list(classifiers.keys())}')
print(f'AE     : {"✅" if ae_encoder else "❌"}')
print(f'GAN    : {"✅" if GAN_AVAILABLE else "❌"}')


# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_signal(signal_str, model_name):
    vals = np.array([float(v.strip()) for v in signal_str.split(',')],
                    dtype=np.float32)
    if model_name == 'MLP':
        vals = np.pad(vals, (0, max(0, 360  - len(vals))))[:360]
        return vals.reshape(1, 360)
    elif model_name == 'CNN':
        vals = np.pad(vals, (0, max(0, 360  - len(vals))))[:360]
        return vals.reshape(1, 360, 1)
    elif model_name in ['RNN', 'LSTM', 'GRU', 'LSTM+Attention']:
        vals = np.pad(vals, (0, max(0, 2048 - len(vals))))[:2048]
        return vals.reshape(1, 32, 64)
    elif model_name == 'MobileNet':
        vals = np.pad(vals, (0, max(0, 1280 - len(vals))))[:1280]
        return vals.reshape(1, 1280)
    elif model_name == 'ResNet':
        vals = np.pad(vals, (0, max(0, 2048 - len(vals))))[:2048]
        return vals.reshape(1, 2048)
    else:
        vals = np.pad(vals, (0, max(0, 360  - len(vals))))[:360]
        return vals.reshape(1, 360)


def make_sample_signal(model_name='MLP'):
    n_map = {
        'MLP': 360, 'CNN': 360,
        'RNN': 2048, 'LSTM': 2048, 'GRU': 2048, 'LSTM+Attention': 2048,
        'MobileNet': 1280, 'ResNet': 2048,
    }
    n   = n_map.get(model_name, 360)
    t   = np.linspace(0, 4 * np.pi, n)
    ecg = (np.sin(t) + 0.3*np.sin(3*t) +
           0.1*np.sin(5*t) + 0.05*np.random.randn(n))
    ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min())
    return ', '.join([f'{v:.4f}' for v in ecg])


# ─── Tab 1: Classify ──────────────────────────────────────────────────────────
def classify_ecg(signal_str, model_name):
    try:
        if model_name not in classifiers:
            return f'❌ {model_name} not loaded', None

        x    = parse_signal(signal_str, model_name)
        pred = classifiers[model_name].predict(x)[0]
        cls  = int(np.argmax(pred))

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f'ECG Classification — {model_name}',
                     fontsize=15, fontweight='bold')

        vals = np.array([float(v.strip()) for v in signal_str.split(',')],
                        dtype=np.float32)[:360]
        axes[0].plot(vals, color='#2196F3', linewidth=2.0)
        axes[0].fill_between(range(len(vals)), vals, alpha=0.15, color='#2196F3')
        axes[0].set_title('Input ECG Signal', fontweight='bold', pad=10)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Amplitude')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        bars = axes[1].barh(CLASS_NAMES, pred, color=COLORS,
                            height=0.5, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, pred):
            axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                         f'{val*100:.1f}%', va='center',
                         fontsize=11, fontweight='bold')
        axes[1].set_xlim(0, 1.2)
        axes[1].set_xlabel('Probability')
        axes[1].set_title(
            f'Prediction: {CLASS_NAMES[cls]}  ({pred[cls]*100:.1f}%)',
            fontweight='bold', pad=10, color=COLORS[cls])
        axes[1].axvline(0.5, color='grey', linestyle='--', alpha=0.4)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout(pad=2.5)

        result = f"""
### ✅ Classification Result

| Field | Value |
|-------|-------|
| **Model** | {model_name} |
| **Prediction** | {CLASS_NAMES[cls]} |
| **Confidence** | {pred[cls]*100:.1f}% |

#### All Class Probabilities
| Class | Probability |
|-------|------------|
| Normal(N) | {pred[0]*100:.1f}% |
| PVC(V)    | {pred[1]*100:.1f}% |
| APB(A)    | {pred[2]*100:.1f}% |
| Paced(/)  | {pred[3]*100:.1f}% |
| Fusion(F) | {pred[4]*100:.1f}% |
        """
        return result, fig

    except Exception as e:
        return f'❌ Error: {str(e)}', None


# ─── Tab 2: Compare All ───────────────────────────────────────────────────────
def compare_all(signal_str):
    try:
        import pandas as pd
        rows, preds = [], {}

        for name, model in classifiers.items():
            try:
                x           = parse_signal(signal_str, name)
                pred        = model.predict(x)[0]
                cls         = int(np.argmax(pred))
                preds[name] = pred
                rows.append({
                    'Model':      name,
                    'Prediction': CLASS_NAMES[cls],
                    'Confidence': f'{pred[cls]*100:.1f}%',
                    'Normal %':   f'{pred[0]*100:.1f}',
                    'PVC %':      f'{pred[1]*100:.1f}',
                    'APB %':      f'{pred[2]*100:.1f}',
                    'Paced %':    f'{pred[3]*100:.1f}',
                    'Fusion %':   f'{pred[4]*100:.1f}',
                })
            except Exception as e:
                print(f'  {name} skipped: {e}')

        if not rows:
            return 'No models could process this signal', None

        df          = pd.DataFrame(rows)
        model_names = list(preds.keys())
        confs       = [float(r['Confidence'].replace('%', '')) for r in rows]
        bar_colors  = ['#2196F3','#F44336','#4CAF50',
                       '#9C27B0','#FF9800','#FF5722','#00BCD4','#8BC34A']

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('All Models Comparison',
                     fontsize=15, fontweight='bold')

        bars = axes[0].bar(model_names, confs,
                           color=bar_colors[:len(model_names)],
                           edgecolor='white', linewidth=1.5, width=0.6)
        axes[0].set_title('Model Confidence (%)', fontweight='bold', pad=10)
        axes[0].set_ylabel('Confidence %')
        axes[0].set_ylim(0, 120)
        axes[0].tick_params(axis='x', rotation=35)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        for bar, v in zip(bars, confs):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                         v + 1, f'{v:.1f}%',
                         ha='center', fontsize=10, fontweight='bold')

        prob = np.array([preds[n] for n in model_names])
        im   = axes[1].imshow(prob, cmap='Blues',
                               aspect='auto', vmin=0, vmax=1)
        axes[1].set_xticks(range(5))
        axes[1].set_xticklabels(CLASS_NAMES, rotation=25, fontsize=10)
        axes[1].set_yticks(range(len(model_names)))
        axes[1].set_yticklabels(model_names, fontsize=10)
        axes[1].set_title('Probability Heatmap', fontweight='bold', pad=10)
        plt.colorbar(im, ax=axes[1], fraction=0.03)
        for i in range(len(model_names)):
            for j in range(5):
                axes[1].text(j, i, f'{prob[i,j]:.2f}',
                             ha='center', va='center', fontsize=9,
                             fontweight='bold',
                             color='white' if prob[i,j] > 0.5 else '#333')

        plt.tight_layout(pad=2.5)
        return df, fig

    except Exception as e:
        return str(e), None


# ─── Tab 3: GAN ───────────────────────────────────────────────────────────────
def generate_ecg(n_samples, seed):
    if not GAN_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, '⚠️  GAN model not loaded',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    try:
        np.random.seed(int(seed))
        n    = int(n_samples)
        z    = np.random.normal(0, 1, (n, 128)).astype(np.float32)
        imgs = gan_generator.predict(z)
        imgs = np.clip((imgs + 1) / 2, 0, 1)

        cols  = min(n, 4)
        rows  = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        fig.suptitle('GAN Generated ECG CWT Scalograms',
                     fontsize=15, fontweight='bold')
        axes = np.array(axes).flatten() if n > 1 else [axes]
        for i in range(n):
            axes[i].imshow(imgs[i], interpolation='bilinear')
            axes[i].set_title(f'Generated #{i+1}',
                              fontsize=11, fontweight='bold')
            axes[i].axis('off')
        for j in range(n, len(axes)):
            axes[j].axis('off')
        plt.tight_layout(pad=2.0)
        return fig

    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        ax.axis('off')
        return fig


# ─── Tab 4: AE ────────────────────────────────────────────────────────────────
def reconstruct_ecg(signal_str):
    if ae_encoder is None or ae_decoder is None:
        return None, '❌ Autoencoder not loaded'
    try:
        vals        = np.array([float(v.strip()) for v in signal_str.split(',')],
                               dtype=np.float32)
        needed      = 64 * 64 * 3
        vals_padded = np.pad(vals, (0, max(0, needed - len(vals))))[:needed]
        scalogram   = vals_padded.reshape(1, 64, 64, 3).astype(np.float32)
        scalogram   = ((scalogram - scalogram.min()) /
                       (scalogram.max() - scalogram.min() + 1e-8))

        lat        = ae_encoder.predict(scalogram)
        recon      = ae_decoder.predict(lat)
        orig_flat  = scalogram[0].flatten()[:360]
        recon_flat = recon[0].flatten()[:360]
        diff       = orig_flat - recon_flat
        mse        = float(np.mean(np.square(diff)))

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Autoencoder — ECG Reconstruction',
                     fontsize=15, fontweight='bold')

        axes[0].plot(orig_flat, color='#2196F3', linewidth=2.0)
        axes[0].fill_between(range(len(orig_flat)), orig_flat,
                             alpha=0.15, color='#2196F3')
        axes[0].set_title('Original ECG Signal', fontweight='bold', pad=10)
        axes[0].set_ylabel('Amplitude')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        axes[1].plot(recon_flat, color='#F44336', linewidth=2.0, linestyle='--')
        axes[1].fill_between(range(len(recon_flat)), recon_flat,
                             alpha=0.15, color='#F44336')
        axes[1].set_title('Reconstructed ECG Signal', fontweight='bold', pad=10)
        axes[1].set_ylabel('Amplitude')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        axes[2].plot(orig_flat,  color='#2196F3', linewidth=2.0,
                     label='Original', alpha=0.85)
        axes[2].plot(recon_flat, color='#F44336', linewidth=2.0,
                     linestyle='--', label='Reconstructed', alpha=0.85)
        axes[2].fill_between(range(len(diff)), diff,
                             alpha=0.3, color='#FF9800', label='Difference')
        axes[2].set_title('Overlay + Difference', fontweight='bold', pad=10)
        axes[2].set_xlabel('Sample Index')
        axes[2].set_ylabel('Amplitude')
        axes[2].legend(loc='upper right', fontsize=11)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)

        plt.tight_layout(pad=2.5)

        q = ('✅ Excellent' if mse < 0.01
             else '✅ Good'  if mse < 0.05
             else '⚠️ Fair')

        return fig, f"""
### 🔄 Autoencoder Results

| Metric | Value |
|--------|-------|
| **Latent Dimension** | {lat.shape[1]} |
| **Reconstruction MSE** | {mse:.6f} |
| **Compression** | 64×64×3 → {lat.shape[1]} → 64×64×3 |
| **Quality** | {q} |
        """
    except Exception as e:
        return None, f'❌ Error: {str(e)}'


# ─── CSS ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Segoe UI', sans-serif !important;
}
h1 {
    text-align: center !important;
    font-size: 2em !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
}
footer { display: none !important; }
"""

INFO = """
# 📊 Model Reference

## Dataset — MIT-BIH Arrhythmia
| Property | Value |
|----------|-------|
| Samples | 92,524 ECG beats |
| Classes | Normal, PVC, APB, Paced, Fusion |
| Signal | 360 samples per beat |
| Scalograms | 64×64×3 images |

## Input Shapes
| Model | Input |
|-------|-------|
| MLP | (360,) |
| CNN | (360,1) |
| RNN / LSTM / GRU / Attention | (32,64) |
| MobileNet | (1280,) |
| ResNet | (2048,) |
| AE Encoder | (64,64,3) |
| GAN | (128,) |

## Performance
| Model | Accuracy | F1 |
|-------|----------|----|
| MLP | 95% | 0.94 |
| CNN | 97% | 0.96 |
| RNN | 93% | 0.92 |
| LSTM | 96% | 0.95 |
| GRU | 95% | 0.94 |
| **LSTM+Attention** | **98%** | **0.97** |
"""


# ─── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title='ECG Deep Learning Suite',
               css=CUSTOM_CSS) as demo:

    gr.Markdown('# 🫀 ECG Deep Learning Suite')
    gr.Markdown(
        '<p style="text-align:center;color:#666;font-size:1.05em;">'
        'MIT-BIH Arrhythmia &nbsp;|&nbsp; '
        'MLP · CNN · RNN · LSTM · GRU · LSTM+Attention · MobileNet · ResNet · AE · GAN'
        '</p>'
    )

    with gr.Tabs():

        # Tab 1
        with gr.Tab('🔍 Classify ECG'):
            gr.Markdown('#### Select model → Load Sample → Classify')
            with gr.Row():
                with gr.Column(scale=3):
                    sig1 = gr.Textbox(
                        label='ECG Signal (comma separated)',
                        lines=6,
                        placeholder='0.1234, 0.4567, 0.7890, ...'
                    )
                with gr.Column(scale=1):
                    model_drop   = gr.Dropdown(
                        choices=list(classifiers.keys()),
                        value=list(classifiers.keys())[0],
                        label='Select Model'
                    )
                    gr.Markdown('')
                    sample_btn1  = gr.Button('📥 Load Sample')
                    gr.Markdown('')
                    classify_btn = gr.Button('▶ Classify', variant='primary')

            classify_out = gr.Markdown()
            classify_plt = gr.Plot()

            sample_btn1.click(make_sample_signal,
                              inputs=[model_drop], outputs=[sig1])
            classify_btn.click(classify_ecg,
                               inputs=[sig1, model_drop],
                               outputs=[classify_out, classify_plt])

        # Tab 2
        with gr.Tab('📊 Compare All Models'):
            gr.Markdown('#### Load Sample → Compare All Models at once')
            with gr.Row():
                with gr.Column(scale=3):
                    sig2 = gr.Textbox(
                        label='ECG Signal (comma separated)',
                        lines=6,
                        placeholder='0.1234, 0.4567, ...'
                    )
                with gr.Column(scale=1):
                    gr.Markdown('')
                    sample_btn2 = gr.Button('📥 Load Sample')
                    gr.Markdown('')
                    compare_btn = gr.Button('▶ Compare All', variant='primary')

            compare_tbl = gr.Dataframe(label='Results Table')
            compare_plt = gr.Plot()

            sample_btn2.click(lambda: make_sample_signal('MLP'), outputs=[sig2])
            compare_btn.click(compare_all,
                              inputs=[sig2],
                              outputs=[compare_tbl, compare_plt])

        # Tab 3
        with gr.Tab('🎨 Generate ECG (GAN)'):
            gr.Markdown('#### No input needed — GAN generates from random noise')
            with gr.Row():
                with gr.Column(scale=1):
                    n_slider   = gr.Slider(1, 8, value=4, step=1,
                                            label='Number of images')
                    seed_input = gr.Number(value=42, label='Random seed')
                    gen_btn    = gr.Button('▶ Generate', variant='primary')
            gen_plot = gr.Plot()
            gen_btn.click(generate_ecg,
                          inputs=[n_slider, seed_input],
                          outputs=[gen_plot])

        # Tab 4
        with gr.Tab('🔄 Reconstruct (AE)'):
            gr.Markdown('#### Load Sample → Encode + Reconstruct ECG')
            with gr.Row():
                with gr.Column(scale=3):
                    sig3 = gr.Textbox(
                        label='ECG Signal (comma separated)',
                        lines=6,
                        placeholder='0.1234, 0.4567, ...'
                    )
                with gr.Column(scale=1):
                    gr.Markdown('')
                    sample_btn3 = gr.Button('📥 Load Sample')
                    gr.Markdown('')
                    ae_btn      = gr.Button('▶ Reconstruct', variant='primary')

            ae_info = gr.Markdown()
            ae_plot = gr.Plot()

            sample_btn3.click(lambda: make_sample_signal('MLP'), outputs=[sig3])
            ae_btn.click(reconstruct_ecg,
                         inputs=[sig3],
                         outputs=[ae_plot, ae_info])

        # Tab 5
        with gr.Tab('ℹ️ Model Info'):
            gr.Markdown(INFO)


if __name__ == '__main__':
    demo.launch(share=True)