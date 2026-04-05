import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras

MODELS_DIR = r'C:\Users\sanga\OneDrive\Documents\1_M.tech\SEM2\DL\Scaffold\mit_ecg\models'
EXPORT_DIR = r'C:\Users\sanga\OneDrive\Documents\1_M.tech\SEM2\DL\Scaffold\mit_ecg\saved_models'
os.makedirs(EXPORT_DIR, exist_ok=True)

files = {
    'mlp_model':       'mlp_model.keras',
    'cnn_model':       'cnn_model.keras',
    'rnn_model':       'rnn_model.keras',
    'lstm_model':      'lstm_model.keras',
    'gru_model':       'gru_model.keras',
    'attention_model': 'attention_model.keras',
    'ae_encoder':      'ae_encoder.keras',
    'ae_decoder':      'ae_decoder.keras',
    'gan_generator':   'generator.keras',      # ← your file is generator.keras
    'clf_mobile':      'clf_mobile.keras',     # ← bonus models
    'clf_resnet':      'clf_resnet.keras',     # ← bonus models
}

for export_name, filename in files.items():
    src = os.path.join(MODELS_DIR, filename)
    dst = os.path.join(EXPORT_DIR, export_name)

    if not os.path.exists(src):
        print(f'❌ NOT FOUND : {filename}')
        continue

    if os.path.exists(dst):
        print(f'⏭️  SKIP (exists): {export_name}')
        continue

    try:
        print(f'Converting {filename} ... ', end='', flush=True)
        m = keras.models.load_model(src)
        m.export(dst)
        print(f'✅')
    except Exception as e:
        print(f'❌ Failed: {e}')

print('\n✅ All done!')
print(f'Saved to: {EXPORT_DIR}')
for f in os.listdir(EXPORT_DIR):
    print(f'   {f}')