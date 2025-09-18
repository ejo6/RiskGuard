import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

cli = InferenceServerClient(url="localhost:8000")

def infer(model, n_features):
    # Randomly fill in
    x = np.random.rand(1, n_features).astype(np.float32)  # shape [1, F]
    inp = InferInput("float_input", x.shape, "FP32")
    inp.set_data_from_numpy(x, binary_data=True)

    outputs = [
        InferRequestedOutput("probabilities", binary_data=True),
        InferRequestedOutput("label", binary_data=True),
    ]

    result = cli.infer(model_name=model, inputs=[inp], outputs=outputs)
    probs = result.as_numpy("probabilities")
    label = result.as_numpy("label")
    print(model, "->", label.ravel(), probs)


infer("lgbm_top25", 25)
infer("lgbm_top50", 50)
infer("lgbm_top100", 100)
