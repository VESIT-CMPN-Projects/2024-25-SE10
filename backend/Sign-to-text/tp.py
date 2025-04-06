import h5py
import json

def get_tf_version_from_h5(model_path):
    with h5py.File(model_path, 'r') as f:
        if 'keras_version' in f.attrs:
            keras_version = f.attrs['keras_version'].decode('utf-8')
            print(f"Keras version: {keras_version}")
        if 'model_config' in f.attrs:
            model_config = f.attrs['model_config'].decode('utf-8')
            model_config = json.loads(model_config)
            if 'backend' in model_config:
                backend_info = model_config['backend']
                print(f"Backend info: {backend_info}")

if __name__ == "__main__":
    get_tf_version_from_h5("./Model/retrained_model.h5")