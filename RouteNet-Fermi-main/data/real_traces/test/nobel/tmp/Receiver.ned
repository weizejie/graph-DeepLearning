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

module Receiver
{
    parameters:
        int id;
        int numNodes;
        @display("i=misc/node_vs,gold;bgb=224,246");
        

        
        
    gates:
        inout port[];
        
    submodules:
        routing: Routing {
            numNodes = numNodes;
            id = id;
        };
        queue[sizeof(port)]: NodeQueue;
        
    connections:

        for i=0..sizeof(port)-1 {
            routing.out++ --> queue[i].in;
            routing.in++ <-- queue[i].out;
            queue[i].line <--> port[i];
        }
}
