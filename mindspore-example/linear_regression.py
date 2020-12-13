from mindspore import context
import numpy as np
from mindspore import dataset as ds 
from mindspore.common.initializer import Normal
from mindspore import nn, Model
from mindspore.train.callback import Callback

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


def get_data(num, w=2.0, b=3.0):
    """synthetic linear regression data

    Args:
        num (int): number of rows
        w (float, optional): [description]. Defaults to 2.0.
        b (float, optional): [description]. Defaults to 3.0.

    Yields:
        list: (x, y)
    """
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)


def create_dataset(num_data, batch_size=16, repeat_size=1):
    """[summary]

    Args:
        num_data ([type]): [description]
        batch_size (int, optional): [description]. Defaults to 16.
        repeat_size (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data


num_data = 1600
batch_size = 16
repeat_size = 1

ds_train = create_dataset(num_data, batch_size=batch_size, repeat_size=repeat_size)
print("The dataset size of ds_train:", ds_train.get_dataset_size())
dict_datasets = ds_train.create_dict_iterator().get_next()

print(dict_datasets.keys())
print("The x label value shape:", dict_datasets["data"].shape)
print("The y label value shape:", dict_datasets["label"].shape)


class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        x = self.fc(x)
        return x

net = LinearNet()
model_params = net.trainable_params()
net_loss = nn.loss.MSELoss()
opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
model = Model(net, net_loss, opt)
ckpt_cb = Callback()
model.train(1, ds_train, callbacks= [ckpt_cb], dataset_sink_mode=False)



# def train_net(model, batch_size, ds_train, ckpt_cb, sink_mode):
#     model.train(batch_size, ds_train, callbacks= [ckpt_cb], dataset_sink_mode=sink_mode)

print(net.trainable_params()[0], "\n%s" % net.trainable_params()[1])