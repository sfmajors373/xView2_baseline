import tritonhttpclient


class DamageClassificationModel(object):
    '''
    Class to load the damage classification model from triton and run inference
    '''

    def __init__(self, triton_url='triton-docker:8000'):
        self.input_name = 'input_1'
        self.output_name = 'dense_3'
        self.model_name = 'damage-classification'
        self.model_version = '1'
        self.label = 0
        self.input_size = (64, 128, 128, 3)
        self.triton_client = tritonhttpclient.InferenceServerClient(url=triton_url, verbose=False)

    def predict(self, img):
        input0 = tritonhttpclient.InferInput(self.input_name, self.input_size, 'FP32')
        input0.set_data_from_numpy(img, binary_data=False)
        output = tritonhttpclient.InferRequestedOutput(self.output_name, binary_data=False)
        response = self.triton_client.infer(self.model_name,
                                            model_version=self.model_version,
                                            inputs=[input0],
                                            outputs=[output])
        print('!!!!!!!!!!!!!!!!!!!! RESPONSE: ', response)
        logits = response.as_numpy(self.output_name)
        return logits
