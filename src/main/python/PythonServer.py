# -*- coding: utf-8 -*-
"""

@author: roshan94
"""

#!/usr/bin/env python

#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
#

#import glob
import sys
sys.path.append('gen-py')
#sys.path.insert(0,glob.glob('./build/lib*')[0])

from thriftinterface import PythonNeuralNetInterface
from thriftinterface.ttypes import NeuralNetScores

from Vassar_Net_deploy import NeuralNetScienceAndCost

#from thrift-py import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

class NeuralNetHandler:
    def __init__(self):
        self.log = {}

    def ping(self):
        print('ping()')
    
    def neuralNetArchitectureEval(self, arch):
        array, science, cost  = NeuralNetScienceAndCost(arch)
        print('Test arch ' + str(array) + 'evaluated. Science: ' + str(science[0][0]) + ', Cost: ' + str(cost[0][0]))
        nn_scores = NeuralNetScores(arch = array, science=science, cost=cost)
        return nn_scores
    

if __name__ == '__main__':
    handler = NeuralNetHandler()
    processor = PythonNeuralNetInterface.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    # You could do one of these for a multithreaded server
    # server = TServer.TThreadedServer(
    #     processor, transport, tfactory, pfactory)
    # server = TServer.TThreadPoolServer(
    #     processor, transport, tfactory, pfactory)

    print('Starting the server...')
    server.serve()
    print('done.')
