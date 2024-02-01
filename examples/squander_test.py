from bqskit.ir import Circuit
circuit = Circuit.from_file('test_circuit/heisenberg-16-20.qasm')


from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner
from bqskit.passes import ForEachBlockPass
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.passes import SquanderSynthesisPass
from bqskit.qis.unitary import Unitary
import numpy as np
import time
import pickle 
# import

start_squander = time.time()


###########################################################################
# SQUANDER synthesis
 
 
config = { 'max_outer_iterations': 1, 
                'agent_lifetime':200,
                'max_inner_iterations_agent': 100000,
                'convergence_length': 10,
                'max_inner_iterations_compression': 10000,
                'max_inner_iterations' : 10000,
                'max_inner_iterations_final': 10000, 		
                'Randomized_Radius': 0.3, 
                'randomized_adaptive_layers': 1,
                'optimization_tolerance_agent': 1e-3} #1e-2
                

task = CompilationTask(circuit, [
    QuickPartitioner(6), # itt megpróbálni 4-re emelni
    ForEachBlockPass([SquanderSynthesisPass(squander_config=config, optimizer_engine="AGENTS" ), ScanningGateRemovalPass()]), 
    UnfoldPass(),
])
print("\n the original gates are: \n")

    
original_gates = []

for gate in circuit.gate_set:
    case_original = {f"{gate}count:": circuit.count(gate)}
    original_gates.append(case_original)
    
print(original_gates, "\n")

with open("original_gates.pickle", "wb") as file:
    pickle.dump(original_gates, file, pickle.HIGHEST_PROTOCOL)

# Finally, we construct a compiler and submit the task
with Compiler(num_workers=1) as compiler:
    synthesized_circuit_squander = compiler.compile(task)

#np.savetxt("squander unitary matrix",synthesized_circuit_squander.get_unitary())
Circuit.save(synthesized_circuit_squander,"squnder_circuit.qasm")

# itt is kiolvasni circuit és unitért és összehasonlítani az eredetivel


print("\n the gates with squander :")


squander_gates = []

for gate in synthesized_circuit_squander.gate_set:
    case_squander = {f"{gate}count:":  synthesized_circuit_squander.count(gate)}
    squander_gates.append(case_squander)
 
end_squander = time.time()
time_squander = {"the execution time with squander:": end_squander-start_squander}
squander_gates.append(time_squander)
print(squander_gates, "\n")

with open("squander_gates.pickle", "wb") as file:
    pickle.dump(squander_gates, file, pickle.HIGHEST_PROTOCOL)

exit()

###########################################################################
# QSearch synthesis

start_qsearch = time.time()


task = CompilationTask(circuit, [
    QuickPartitioner(4), 
    ForEachBlockPass([QSearchSynthesisPass(), ScanningGateRemovalPass()]), 
    UnfoldPass(),
])

print("\n the gates with qsearch :")
# Finally, we construct a compiler and submit the task
with Compiler() as compiler:
    synthesized_circuit_qsearch = compiler.compile(task)


#np.savetxt("qsearch unitary matrix",synthesized_circuit_qsearch.get_unitary())
Circuit.save(synthesized_circuit_qsearch,"qsearch_circuit.qasm")
# itt is kiolvasni circuit és unitért és összehasonlítani az eredetivel



qsearch_gates = []

for gate in synthesized_circuit_qsearch.gate_set:
    case_qsearch = {f"{gate}count:":  synthesized_circuit_qsearch.count(gate)}   
    qsearch_gates.append(case_qsearch)
 
end_qsearch = time.time()
time_qsearch = {"the execution time with qsearch:": end_qsearch-start_qsearch}
qsearch_gates.append(time_qsearch)
print(qsearch_gates, "\n")
# végső eredményt kiiratni fájlba quasm-ba és ellenőrizni .

with open("qsearch_gates.pickle", "wb") as file:
    pickle.dump(qsearch_gates, file, pickle.HIGHEST_PROTOCOL)


 

