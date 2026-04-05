# # import tensorflow as tf
# # from tensorflow import keras

# # models_dir = 'models'

# # names = ['mlp_ecg', 'cnn_ecg', 'rnn_ecg', 'lstm_ecg',
# #          'gru_ecg', 'lstm_attention_ecg']

# # for name in names:
# #     m = keras.models.load_model(f'{models_dir}/{name}.keras')
# #     print(f'{name:25s} input={m.input_shape}  output={m.output_shape}')

# # enc = keras.models.load_model(f'{models_dir}/ae_encoder.keras')
# # print(f'ae_encoder               input={enc.input_shape}')

# # gan = keras.models.load_model(f'{models_dir}/gan_generator.keras')
# # print(f'gan_generator            input={gan.input_shape}')

# import os
# MODELS_DIR = r'C:\Users\sanga\OneDrive\Documents\1_M.tech\SEM2\DL\Scaffold\mit_ecg\models'
# for f in os.listdir(MODELS_DIR):
#     print(repr(f))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

SAVED_DIR = r'C:\Users\sanga\OneDrive\Documents\1_M.tech\SEM2\DL\Scaffold\mit_ecg\saved_models'

folders = os.listdir(SAVED_DIR)
for folder in sorted(folders):
    path = os.path.join(SAVED_DIR, folder)
    try:
        m = tf.saved_model.load(path)
        # Get input signature
        sig = m.signatures if hasattr(m, 'signatures') else None
        infer = m.serve.input_signature
        print(f'{folder:20s} : {infer}')
    except Exception as e:
        print(f'{folder:20s} : ERROR {e}')