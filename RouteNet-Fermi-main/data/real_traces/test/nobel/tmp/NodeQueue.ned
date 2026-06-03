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
simple NodeQueue
{
    parameters:
        int capacity @unit(byte) = default (1024B);
        int queueSize = default(32);
        @display("i=block/queue");
    gates:
        input in;
        output out;
        inout line;
}
