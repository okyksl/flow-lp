import torch

class Classifier(torch.nn.Module):
    def __init__(self, num_classes, resolution=(1, 28, 28),  **kwargs):
        """
        Initialize classifier.

        :param num_classes: number of classes to classify
        :type num_classes: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param architecture: architecture builder to use
        :type architecture: str
        """

        super(Classifier, self).__init__()

        assert num_classes > 0, 'positive num_classes expected'
        assert len(resolution) <= 3
        resolution = list(resolution)

        self.num_classes = int(num_classes) # Having strange bug where torch complaints about numpy.in64 being passed to nn.Linear.
        """ (int) Number of classes. """

        self.layers = []
        """ ([str]) Will hold layer names. """

        assert len(resolution) == 3

        #
        # All layer names will be saved in self.layers.
        # This allows the forward method to run the network without
        # knowing its exact structure.
        # The representation layer is assumed to be labeled accordingly.
        #

        layer = 0
        channels = []
        resolutions = []

        # Determine parameters of network
        activation = None
        kwargs_activation = kwargs.get('activation', 'relu')
        kwargs_activation = kwargs_activation.lower()
        if kwargs_activation == 'relu':
            activation = torch.nn.ReLU
        elif kwargs_activation == 'sigmoid':
            activation = torch.nn.Sigmoid
        elif kwargs_activation == 'tanh':
            activation = torch.nn.Tanh
        elif kwargs_activation == 'leaky_relu':
            activation = torch.nn.LeakyReLU
        elif kwargs_activation == 'none':
            pass
        else:
            raise ValueError('Unsupported activation %s.' % kwargs_activation)

        gain = 1
        if activation:
            gain = torch.nn.init.calculate_gain(kwargs_activation)

        batch_normalization = kwargs.get('batch_normalization', True)
        dropout = kwargs.get('dropout', False)
        start_channels = kwargs.get('start_channels', 16)
        kernel_size = kwargs.get('kernel_size', 4)
        assert kernel_size%2 == 0

        while True:
            input_channels = resolution[0] if layer == 0 else channels[layer - 1]
            output_channels = start_channels if layer == 0 else channels[layer - 1]*2

            # Large kernel size was result of poor discriminator;
            # generator did only produce very "thin" EMNIST digits.
            conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2-1)
            torch.nn.init.kaiming_normal_(conv.weight, gain)
            #torch.nn.init.normal_(conv.weight, 0, 0.001)
            torch.nn.init.constant_(conv.bias, 0)
            self.add_layer('conv%d' % layer, conv)

            if batch_normalization:
                bn = torch.nn.BatchNorm2d(output_channels)
                torch.nn.init.constant_(bn.weight, 1)
                torch.nn.init.constant_(bn.bias, 0)
                self.add_layer('bn%d' % layer, bn)

            if activation:
                relu = activation(True)
                self.add_layer('act%d' % layer, relu)

            channels.append(output_channels)
            resolutions.append([
                resolution[1] // 2 if layer == 0 else resolutions[layer - 1][0] // 2,
                resolution[2] // 2 if layer == 0 else resolutions[layer - 1][1] // 2
            ])
            if resolutions[-1][0] // 2 < 2 or resolutions[-1][1] < 2:
                break;

            layer += 1

        #if dropout:
        #    drop = torch.nn.Dropout2d()
        #    self.add_layer('drop%d' % layer, drop)

        # Having strange bug where torch complaints about numpy.in64 being passed to nn.Linear.
        representation = int(resolutions[-1][0]*resolutions[-1][1]*channels[-1])
        view = torch.nn.Flatten()
        self.add_layer('view', view)

        linear1 = torch.nn.Linear(representation, 10*self.num_classes)
        linear2 = torch.nn.Linear(10*self.num_classes, self.num_classes)

        torch.nn.init.kaiming_normal_(linear1.weight, gain)
        #torch.nn.init.normal_(linear1.weight, 0, 0.001)
        torch.nn.init.constant_(linear1.bias, 0)
        torch.nn.init.kaiming_normal_(linear2.weight, gain)
        #torch.nn.init.normal_(linear2.weight, 0, 0.001)
        torch.nn.init.constant_(linear2.bias, 0)

        self.add_layer('representation', linear1)
        if dropout:
            drop = torch.nn.Dropout()
            self.add_layer('drop%d' % layer, drop)
        self.add_layer('logits', linear2)
    

    def add_layer(self, name, layer):
        """
        Add a layer.
        :param name:
        :param layer:
        :return:
        """

        setattr(self, name, layer)
        self.layers.append(name)

    def forward(self, image, return_features=False):
        """
        Forward pass, takes an image and outputs the predictions.

        :param image: input image
        :type image: torch.autograd.Variable
        :param return_representation: whether to also return representation layer
        :type return_representation: bool
        :return: logits
        :rtype: torch.autograd.Variable
        """

        features = []
        output = image

        for name in self.layers:
            output = getattr(self, name)(output)
            features.append(output)
        if return_features:
            return output, features
        else:
            return output
