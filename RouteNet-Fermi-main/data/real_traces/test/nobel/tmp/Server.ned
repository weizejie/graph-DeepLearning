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

module Server
{
    parameters:
        int numNodes;
        int id;
        
        @display("i=misc/node_vs,gold;bgb=435,425");

    gates:
        inout port[];
        input tControl;

    submodules:
        tcontroller: TrafficController {
            numNodes = numNodes;
        }
        app[numNodes]: Application {
            id = id;
            dest = index;
            numNodes = numNodes;
            @display("p=151,149");
        }
        routing: Routing {
            numNodes = numNodes;
            id = id;
            @display("p=151,238");
        }
        queue[sizeof(port)]: NodeQueueQoS {
            @display("p=151,340");
        }
    connections:
        for i=0..numNodes-1 {
            routing.localIn++ <-- app[i].out;
            tcontroller.out++ --> app[i].tcontrol;
        }
        for i=0..sizeof(port)-1 {
            routing.out++ --> queue[i].in;
            routing.in++ <-- queue[i].out;
            queue[i].line <--> port[i];
        }
        tControl --> tcontroller.in;
}
