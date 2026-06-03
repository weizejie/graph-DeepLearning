//
// Copyright (C) 2020 CBA research group, Technical University of Catalonia.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package netsimulator;

//
// TODO auto-generated module
//
simple NodeQueueQoS
{
    parameters:
        // Each QoS represents a node queue
        int levelsQoS  = default (1);
        // String mapping the tos to queue 
        // Queues separated by ';', tos separated by ',' 
        // ex: 0;1,2;3,4,5 => 3 queus and 5 tos
        // If not specified: One ToS to one queue
        string tosToQoSqueue = default("-");
        // String specifiing the size of the queue separated with ','. 
        // Should have the same number of fields as levels of QoS
        string queueSizes = default("-");
        string bufferSizes = default("-");
        string schedulingPolicy = default ("SP");
        // String specifiing the weight of each queue separated with ','. 
        // Used for WFQ and WRR
        // Should have the same number of fields as levels of QoS
        string schedulingWeights = default ("-");
        @display("i=block/queue");
    gates:
        input in;
        output out;
        inout line;
}
