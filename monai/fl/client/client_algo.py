# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc

from monai.fl.utils.exchange_object import ExchangeObject


class ClientAlgo(abc.ABC):
    """
    objective: provide an abstract base class for defining algo to run on any platform.
    To define a new algo script, subclass this class and implement the
    following abstract methods:

        - #Algo.train()
        - #Algo.get_weights()
        - #Algo.evaluate()

    initialize() and finalize() can be optionally be implemented to help with lifecycle management of the object.
    """

    def initialize(self, extra=None):
        """call to initialize the ClientAlgo class"""
        pass

    def finalize(self, extra=None):
        """call to finalize the ClientAlgo class"""
        pass

    def abort(self, extra=None):
        """call to abort the ClientAlgo training or evaluation"""
        pass

    @abc.abstractmethod
    def train(self, data: ExchangeObject, extra=None) -> None:
        """
        objective: train network and produce new network from train data.
        # Arguments
        data: ExchangeObject containing current network weights to base training on.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_weights(self, extra=None) -> ExchangeObject:
        """
        objective: get current local weights or weight differences

        # Returns
        ExchangeObject: current local weights or weight differences.

        # Example returns ExchangeObject, e.g.::

            ExchangeObject(
                weights = self.trainer.network.state_dict(),
                optim = None,  # could be self.optimizer.state_dict()
                weight_type = WeightType.WEIGHTS
            )
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, data: ExchangeObject, extra=None) -> ExchangeObject:
        """
        objective: get evaluation metrics on test data.
        # Arguments
        data: ExchangeObject with network weights to use for evaluation

        # Returns
        metrics: ExchangeObject with evaluation metrics.
        """
        raise NotImplementedError
