class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, tf.argmax(predictions[0])]
    
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        
        # Ensure both arrays are of type float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if heatmap.dtype != np.float32:
            heatmap = heatmap.astype(np.float32)
            
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
###
def show_cnn_grad_cam_results(image, mode="PET", case='default'):
    original_height, original_width = image.shape[:2]
    pixel_size = 200
    model_layer = ''

    if mode == 'PET':
        cnn_model = keras.models.load_model("trained_models/pet_vgg16.keras")
        pixel_size = 100
        model_layer = 'block5_conv3'

    if mode == 'CT':
        cnn_model = keras.models.load_model("trained_models/ct_vgg16.keras")
        pixel_size = 100
        model_layer = 'block5_conv3'

    if mode == 'Polar Maps':
        cnn_model = keras.models.load_model("trained_models/polar_maps.keras")
        pixel_size = 300
        model_layer = 'conv2d_2'

    if mode == 'SPECT':
        cnn_model = keras.models.load_model("trained_models/spect_model")
        pixel_size = 250
        model_layer = 'conv2d_3'

    res_image = cv2.resize(image, (pixel_size, pixel_size))
    res_image = res_image.astype('float32') / 255
    res_image = np.expand_dims(res_image, axis=0)

    preds = cnn_model.predict(res_image)

    if mode == 'SPECT':
        classification_preds = (cnn_model.predict(res_image) > 0.5).astype(int)

        if classification_preds[0, 0] == 1:
            print("\n The model predicts that this image instance exhibits signs indicative of an infarction with a probability ", preds[0, 0])
        if classification_preds[0, 1] == 1:
            print("\n The model predicts that this image instance exhibits signs indicative of ischemic case with a probability ", preds[0, 1])
        if classification_preds[0, 2] == 1:
            print("\n The model predicts that this image instance exhibits signs indicative of normal case with a probability ", preds[0, 2])
    else:
        binary_preds = (preds > 0.5).astype(int)
        if binary_preds == 0:
            print("\nThe image is predicted as Benign with a probability ", preds)
        else:
            print("\nThe image is predicted as Malignant with a probability ", preds)

    i = np.argmax(preds[0])

    icam = GradCAM(cnn_model, i, model_layer)
    heatmap = icam.compute_heatmap(res_image)
    heatmap = cv2.resize(heatmap, (pixel_size, pixel_size))

    (heatmap, output) = icam.overlay_heatmap(heatmap, res_image[0], alpha=0.5)

    heatmap = cv2.resize(output, (original_width, original_height))

    # Normalize heatmap to [0, 1]
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max()

    if case != 'Multimodal':
        if mode == 'Polar Maps':
            cad_pred_print(binary_preds, preds)
        elif mode == 'SPECT':
            spect_pred_print(classification_preds, preds)
        else:
            nsclc_pred_print(binary_preds, preds)
    else:
        return preds, heatmap

    st.markdown("")
    st.markdown("")
    col1, col2, col3 = st.columns([3, 1, 3])
    with col1:
        col1.header("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
    with col3:
        col3.header("XAI Image")
        st.image(heatmap, caption="GRAD-CAM Image", use_column_width=True)

    return