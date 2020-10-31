from neural_tangents import stax


def MLPNet():
    return stax.serial(stax.Flatten(), stax.Dense(512), stax.Relu(), stax.Dense(1))


def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
    Main = stax.serial(
       stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
       stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))
    Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
       channels, (3, 3), strides, padding='SAME')
    return stax.serial(stax.FanOut(2), stax.parallel(Main, Shortcut), stax.FanInSum())


def WideResnetGroup(n, channels, strides=(1, 1)):
    blocks = []
    blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks += [WideResnetBlock(channels, (1, 1))]
    return stax.serial(*blocks)


def WideResnet(block_size, k, num_classes):
    return stax.serial(stax.Conv(16, (3, 3), padding='SAME'),
                       WideResnetGroup(block_size, int(16 * k)),
                       WideResnetGroup(block_size, int(32 * k), (2, 2)),
                       WideResnetGroup(block_size, int(64 * k), (2, 2)),
                       stax.AvgPool((8, 8)),
                       stax.Flatten(),
                       stax.Dense(num_classes, 1., 0.))


def CNNBlock(channels, strides=(1, 1)):
    return stax.serial(
       stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
       stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))


def CNNGroup(n, channels, strides=(1, 1)):
    blocks = []
    blocks += [CNNBlock(channels, strides)]
    for _ in range(n - 1):
        blocks += [CNNBlock(channels, (1, 1))]
    return stax.serial(*blocks)


def CNN(block_size, k, num_classes):
    return stax.serial(stax.Conv(16, (3, 3), padding='SAME'),
                       CNNGroup(block_size, int(16 * k)),
                       CNNGroup(block_size, int(32 * k), (2, 2)),
                       CNNGroup(block_size, int(64 * k), (2, 2)),
                       # stax.AvgPool((8, 8)),
                       stax.Flatten(),
                       stax.Dense(num_classes, 1., 0.))


if __name__ == '__main__':
    init_fn, apply_fn, kernel_fn = WideResnet(block_size=4, k=1, num_classes=1)
    init_fn, apply_fn, kernel_fn = CNN(block_size=1, k=1, num_classes=1)
