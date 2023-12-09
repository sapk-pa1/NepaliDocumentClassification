from tensorflow.keras.models import load_model 

MODEL_PATH = r"models\resnet50.h5"
def load_resnet50_doc_class():
    model = load_model(MODEL_PATH)
    return model 
    