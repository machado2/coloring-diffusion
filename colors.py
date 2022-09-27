import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from keras import layers

"""
## Hyperparameterers
"""

# data
dataset_name = "oxford_flowers102"
dataset_repetitions = 5
num_epochs = 1  # train for at least 50 epochs for good results
image_size = 64
# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# optimization
batch_size = 64
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4

def preprocess_image(data):
    # center crop image
    height = tf.shape(data["image"])[0]
    width = tf.shape(data["image"])[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        data["image"],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def prepare_dataset(split):
    return (
        tfds.load(dataset_name, split=split, shuffle_files=True)
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(dataset_repetitions)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


# load dataset
train_dataset = prepare_dataset("train[:80%]+validation[:80%]+test[:80%]")
val_dataset = prepare_dataset("train[80%:]+validation[80%:]+test[80%:]")

"""
## Kernel inception distance

[Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401) is an image quality
metric which was proposed as a replacement for the popular
[Frechet Inception Distance (FID)](https://arxiv.org/abs/1706.08500).
I prefer KID to FID because it is simpler to
implement, can be estimated per-batch, and is computationally lighter. More details
[here](https://keras.io/examples/generative/gan_ada/#kernel-inception-distance).

In this example, the images are evaluated at the minimal possible resolution of the
Inception network (75x75 instead of 299x299), and the metric is only measured on the
validation set for computational efficiency. We also limit the number of sampling steps
at evaluation to 5 for the same reason.

Since the dataset is relatively small, we go over the train and validation splits
multiple times per epoch, because the KID estimation is noisy and compute-intensive, so
we want to evaluate only after many iterations, but for many iterations.

"""


class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()



def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")

class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, image_size, image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()


"""
## Training
"""

# create and compile the model
model = DiffusionModel(image_size, widths, block_depth)
# below tensorflow 2.9:
# pip install tensorflow_addons
# import tensorflow_addons as tfa
# optimizer=tfa.optimizers.AdamW
model.compile(
    optimizer=keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_absolute_error,
)
# pixelwise mean absolute error is used as loss

# save the best model based on the validation KID metric
checkpoint_path = "checkpoints/diffusion_model"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,
)

# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(train_dataset)

# run training and plot generated images periodically
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
        checkpoint_callback,
    ],
)

"""
## Inference
"""

# load the best model and generate images
model.load_weights(checkpoint_path)
model.plot_images()

"""
## Results

By running the training for at least 50 epochs (takes 2 hours on a T4 GPU and 30 minutes
on an A100 GPU), one can get high quality image generations using this code example.

The evolution of a batch of images over a 80 epoch training (color artifacts are due to
GIF compression):

![flowers training gif](https://i.imgur.com/FSCKtZq.gif)

Images generated using between 1 and 20 sampling steps from the same initial noise:

![flowers sampling steps gif](https://i.imgur.com/tM5LyH3.gif)

Interpolation (spherical) between initial noise samples:

![flowers interpolation gif](https://i.imgur.com/hk5Hd5o.gif)

Deterministic sampling process (noisy images on top, predicted images on bottom, 40
steps):

![flowers deterministic generation gif](https://i.imgur.com/wCvzynh.gif)

Stochastic sampling process (noisy images on top, predicted images on bottom, 80 steps):

![flowers stochastic generation gif](https://i.imgur.com/kRXOGzd.gif)

Trained model and demo available on HuggingFace:

| Trained Model | Demo |
| :--: | :--: |
| [![model badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-DDIM-black.svg)](https://huggingface.co/keras-io/denoising-diffusion-implicit-models) | [![spaces badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-DDIM-black.svg)](https://huggingface.co/spaces/keras-io/denoising-diffusion-implicit-models) |
"""

"""
## Lessons learned

During preparation for this code example I have run numerous experiments using
[this repository](https://github.com/beresandras/clear-diffusion-keras).
In this section I list
the lessons learned and my recommendations in my subjective order of importance.

### Algorithmic tips

* **min. and max. signal rates**: I found the min. signal rate to be an important
hyperparameter. Setting it too low will make the generated images oversaturated, while
setting it too high will make them undersaturated. I recommend tuning it carefully. Also,
setting it to 0 will lead to a division by zero error. The max. signal rate can be set to
1, but I found that setting it lower slightly improves generation quality.
* **loss function**: While large models tend to use mean squared error (MSE) loss, I
recommend using mean absolute error (MAE) on this dataset. In my experience MSE loss
generates more diverse samples (it also seems to hallucinate more
[Section 3](https://arxiv.org/abs/2111.05826)), while MAE loss leads to smoother images.
I recommend trying both.
* **weight decay**: I did occasionally run into diverged trainings when scaling up the
model, and found that weight decay helps in avoiding instabilities at a low performance
cost. This is why I use
[AdamW](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/AdamW)
instead of [Adam](https://keras.io/api/optimizers/adam/) in this example.
* **exponential moving average of weights**: This helps to reduce the variance of the KID
metric, and helps in averaging out short-term changes during training.
* **image augmentations**: Though I did not use image augmentations in this example, in
my experience adding horizontal flips to the training increases generation performance,
while random crops do not. Since we use a supervised denoising loss, overfitting can be
an issue, so image augmentations might be important on small datasets. One should also be
careful not to use
[leaky augmentations](https://keras.io/examples/generative/gan_ada/#invertible-data-augmentation),
which can be done following
[this method (end of Section 5)](https://arxiv.org/abs/2206.00364) for instance.
* **data normalization**: In the literature the pixel values of images are usually
converted to the -1 to 1 range. For theoretical correctness, I normalize the images to
have zero mean and unit variance instead, exactly like the random noises.
* **noise level input**: I chose to input the noise variance to the network, as it is
symmetrical under our sampling schedule. One could also input the noise rate (similar
performance), the signal rate (lower performance), or even the
[log-signal-to-noise ratio (Appendix B.1)](https://arxiv.org/abs/2107.00630)
(did not try, as its range is highly
dependent on the min. and max. signal rates, and would require adjusting the min.
embedding frequency accordingly).
* **gradient clipping**: Using global gradient clipping with a value of 1 can help with
training stability for large models, but decreased performance significantly in my
experience.
* **residual connection downscaling**: For
[deeper models (Appendix B)](https://arxiv.org/abs/2205.11487), scaling the residual
connections with 1/sqrt(2) can be helpful, but did not help in my case.
* **learning rate**: For me, [Adam optimizer's](https://keras.io/api/optimizers/adam/)
default learning rate of 1e-3 worked very well, but lower learning rates are more common
in the [literature (Tables 11-13)](https://arxiv.org/abs/2105.05233).

### Architectural tips

* **sinusoidal embedding**: Using sinusoidal embeddings on the noise level input of the
network is crucial for good performance. I recommend setting the min. embedding frequency
to the reciprocal of the range of this input, and since we use the noise variance in this
example, it can be left always at 1. The max. embedding frequency controls the smallest
change in the noise variance that the network will be sensitive to, and the embedding
dimensions set the number of frequency components in the embedding. In my experience the
performance is not too sensitive to these values.
* **skip connections**: Using skip connections in the network architecture is absolutely
critical, without them the model will fail to learn to denoise at a good performance.
* **residual connections**: In my experience residual connections also significantly
improve performance, but this might be due to the fact that we only input the noise
level embeddings to the first layer of the network instead of to all of them.
* **normalization**: When scaling up the model, I did occasionally encounter diverged
trainings, using normalization layers helped to mitigate this issue. In the literature it
is common to use
[GroupNormalization](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization)
(with 8 groups for example) or
[LayerNormalization](https://keras.io/api/layers/normalization_layers/layer_normalization/)
in the network, I however chose to use
[BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/),
as it gave similar benefits in my experiments but was computationally lighter.
* **activations**: The choice of activation functions had a larger effect on generation
quality than I expected. In my experiments using non-monotonic activation functions
outperformed monotonic ones (such as
[ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu)), with
[Swish](https://www.tensorflow.org/api_docs/python/tf/keras/activations/swish) performing
the best (this is also what [Imagen uses, page 41](https://arxiv.org/abs/2205.11487)).
* **attention**: As mentioned earlier, it is common in the literature to use
[attention layers](https://keras.io/api/layers/attention_layers/multi_head_attention/) at low
resolutions for better global coherence. I ommitted them for simplicity.
* **upsampling**:
[Bilinear and nearest neighbour upsampling](https://keras.io/api/layers/reshaping_layers/up_sampling2d/)
in the network performed similarly, I did not try
[transposed convolutions](https://keras.io/api/layers/convolution_layers/convolution2d_transpose/)
however.

For a similar list about GANs check out
[this Keras tutorial](https://keras.io/examples/generative/gan_ada/#gan-tips-and-tricks).
"""

"""
## What to try next?

If you would like to dive in deeper to the topic, a recommend checking out
[this repository](https://github.com/beresandras/clear-diffusion-keras) that I created in
preparation for this code example, which implements a wider range of features in a
similar style, such as:

* stochastic sampling
* second-order sampling based on the
[differential equation view of DDIMs (Equation 13)](https://arxiv.org/abs/2010.02502)
* more diffusion schedules
* more network output types: predicting image or
[velocity (Appendix D)](https://arxiv.org/abs/2202.00512) instead of noise
* more datasets
"""

"""
## Related works

* [Score-based generative modeling](https://yang-song.github.io/blog/2021/score/)
(blogpost)
* [What are diffusion models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
(blogpost)
* [Annotated diffusion model](https://huggingface.co/blog/annotated-diffusion) (blogpost)
* [CVPR 2022 tutorial on diffusion models](https://cvpr2022-tutorial-diffusion-models.github.io/)
(slides avaliable)
* [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364):
attempts unifying diffusion methods under a common framework
* High-level video overviews: [1](https://www.youtube.com/watch?v=yTAMrHVG1ew),
[2](https://www.youtube.com/watch?v=344w5h24-h8)
* Detailed technical videos: [1](https://www.youtube.com/watch?v=fbLgFrlTnGU),
[2](https://www.youtube.com/watch?v=W-O7AZNzbzQ)
* Score-based generative models: [NCSN](https://arxiv.org/abs/1907.05600),
[NCSN+](https://arxiv.org/abs/2006.09011), [NCSN++](https://arxiv.org/abs/2011.13456)
* Denoising diffusion models: [DDPM](https://arxiv.org/abs/2006.11239),
[DDIM](https://arxiv.org/abs/2010.02502), [DDPM+](https://arxiv.org/abs/2102.09672),
[DDPM++](https://arxiv.org/abs/2105.05233)
* Large diffusion models: [GLIDE](https://arxiv.org/abs/2112.10741),
[DALL-E 2](https://arxiv.org/abs/2204.06125/), [Imagen](https://arxiv.org/abs/2205.11487)


"""