"""This module implements the QSearchSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.passdata import PassData

import math 

from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.seed import SeedLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics import AStarHeuristic
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number


_logger = logging.getLogger(__name__)


class SquanderSynthesisPass(SynthesisPass):
    """
    A pass implementing the Squander synthesis algorithm.

    References:
       Dr. Peter Rakyta
    """

    def __init__(
        self,
        heuristic_function: HeuristicFunction = AStarHeuristic(),
        layer_generator: LayerGenerator | None = None,
        success_threshold: float = 1e-7,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_layer: int | None = None,
        store_partial_solutions: bool = False,
        partials_per_depth: int = 25,
        instantiate_options: dict[str, Any] = {},
        squander_config: dict[str, Any] = {} ,
        optimizer_engine: str = 'BFGS'
    ) -> None:
        """
        Construct a search-based synthesis pass.

        Args:
            heuristic_function (HeuristicFunction): The heuristic to guide
                search.

            layer_generator (LayerGenerator | None): The successor function
                to guide node expansion. If left as none, then a default
                will be selected before synthesis based on the target
                model's gate set. (Default: None)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                to unlimited. (Default: None)

            store_partial_solutions (bool): Whether to store partial solutions
                at different depths inside of the data dict. (Default: False)

            partials_per_depth (int): The maximum number of partials
                to store per search depth. No effect if
                `store_partial_solutions` is False. (Default: 25)

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})
                
            squander_config: 

        Raises:
            ValueError: If `max_depth` is nonpositive.
        """
        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                f'Expected HeursiticFunction, got {type(heuristic_function)}.',
            )

        if layer_generator is not None:
            if not isinstance(layer_generator, LayerGenerator):
                raise TypeError(
                    f'Expected LayerGenerator, got {type(layer_generator)}.',
                )

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                f', got {type(success_threshold)}',
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                'Expected cost to be a CostFunctionGenerator'
                f', got {type(cost)}',
            )

        if max_layer is not None and not is_integer(max_layer):
            raise TypeError(
                f'Expected max_layer to be an integer, got {type(max_layer)}.',
            )

        if max_layer is not None and max_layer <= 0:
            raise ValueError(
                f'Expected max_layer to be positive, got {int(max_layer)}.',
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options'
                f', got {type(instantiate_options)}',
            )

        self.heuristic_function = heuristic_function
        self.layer_gen = layer_generator
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.instantiate_options: dict[str, Any] = {
            'cost_fn_gen': self.cost,
        }
        self.instantiate_options.update(instantiate_options)
        self.store_partial_solutions = store_partial_solutions
        self.partials_per_depth = partials_per_depth
        self.squander_config = squander_config
        self.optimizer_engine = optimizer_engine
        
        if squander_config is None :
                #Python map containing hyper-parameters	
            squander_config = { 'max_outer_iterations': 1, 
                'agent_lifetime':200,
                'max_inner_iterations_agent': 10000,
                'convergence_length': 10,
                'max_inner_iterations_compression': 10000,
                'max_inner_iterations' : 500,
                'max_inner_iterations_final': 5000, 		
                'Randomized_Radius': 0.3, 
                'randomized_adaptive_layers': 1,
                'optimization_tolerance_agent': 1e-2} #1e-2
                
        squander_config['optimization_tolerance'] = self.success_threshold
        


     
    def transform_circuit_from_squander_to_qsearch(self,
        cDecompose,
        qubitnum)-> Circuit:
        "a function to change the circuit from bqskit to squander"
    #import all the gates
        from bqskit.ir.gates.constant.cx import CNOTGate
        from bqskit.ir.gates.parameterized.cry import CRYGate
        from bqskit.ir.gates.constant.cz import CZGate
        from bqskit.ir.gates.constant.ch import CHGate
        from bqskit.ir.gates.constant.sycamore import SycamoreGate as SYC
        from bqskit.ir.gates.parameterized.u3 import U3Gate 
        from bqskit.ir.gates.parameterized.rx import RXGate
        from bqskit.ir.gates.parameterized.ry import RYGate
        from bqskit.ir.gates.parameterized.rz import RZGate
        from bqskit.ir.gates.constant.x import XGate
        from bqskit.ir.gates.constant.y import YGate
        from bqskit.ir.gates.constant.z import ZGate
        from bqskit.ir.gates.constant.sx import SqrtXGate
        
        
        
        circuit=Circuit(qubitnum)
        gates=cDecompose.get_Gates()
        for idx in range(len(gates)-1, -1, -1):
            
            gate = gates[idx]

            if gate.get("type") == "CNOT":
                # adding CNOT gate to the quantum circuit
                control_qbit = qubitnum - gate.get("control_qbit")  - 1              
                target_qbit = qubitnum - gate.get("target_qbit") - 1               
                circuit.append_gate(CNOTGate(), (control_qbit, target_qbit))
                

            if gate.get("type") == "CRY":
                # adding CRY gate to the quantum circuit
                Theta=  gate.get("Theta")     
                control_qbit = qubitnum - gate.get("control_qbit") - 1                
                target_qbit = qubitnum - gate.get("target_qbit") - 1               
                circuit.append_gate(CRYGate() ,(control_qbit, target_qbit),[THeta])

            elif gate.get("type") == "CZ":
                # adding CZ gate to the quantum circuit
            
                control_qbit = qubitnum - gate.get("control_qbit") - 1               
                target_qbit = qubitnum - gate.get("target_qbit") - 1                  
                circuit.append_gate(CZGate(), (control_qbit, target_qbit))
               

            elif gate.get("type") == "CH":
                # adding CZ gate to the quantum circuit
                control_qbit = qubitnum - gate.get("control_qbit") -1                
                target_qbit = qubitnum - gate.get("target_qbit") - 1               
                circuit.append_gate(CZGate(), (control_qbit, target_qbit))
               

            elif gate.get("type") == "SYC":
                # Sycamore gate
                control_qbit = qubitnum - gate.get("control_qbit") -1                
                target_qbit = qubitnum - gate.get("target_qbit") - 1               
                circuit.append_gate(SycamoreGate(), (control_qbit, target_qbit))
               
                
            elif gate.get("type") == "U3":
                target_qbit = qubitnum - gate.get("target_qbit") - 1 
                Theta = gate.get("Theta")     
                Lambda =  gate.get("Lambda")  
                Phi = gate.get("Phi")  
                circuit.append_gate(U3Gate(),target_qbit,[Theta,Phi,Lambda])

            elif gate.get("type") == "RX":
                # RX gate               
                target_qbit = qubitnum - gate.get("target_qbit") - 1                
                Theta = gate.get("Theta")               
                circuit.append_gate(RXGate(),( target_qbit),[Theta]) 
              

            elif gate.get("type") == "RY":
                # RY gate
                target_qbit = qubitnum - gate.get("target_qbit") - 1                
                Theta = gate.get("Theta")               
                circuit.append_gate(RYGate(),(target_qbit),[Theta])
                

            elif gate.get("type") == "RZ":
                # RZ gate
                       
                target_qbit = qubitnum - gate.get("target_qbit") - 1               
                Phi =  gate.get("Phi")               
                circuit.append_gate(RZGate(), (target_qbit),[Phi] )
               

            elif gate.get("type") == "X":
                # X gate
                control_qbit = qubitnum - gate.get("control_qbit") - 1               
                target_qbit = qubitnum - gate.get("target_qbit") - 1                
                circuit.append_gate(XGate(), (target_qbit))
               

            elif gate.get("type") == "Y":
                # Y gate
                control_qbit = qubitnum - gate.get("control_qbit") - 1               
                target_qbit = qubitnum - gate.get("target_qbit") - 1               
                circuit.append_gate(YGate(), (target_qbit))
             

            elif gate.get("type") == "Z":
                # Z gate
                control_qbit = qubitnum - gate.get("control_qbit") - 1               
                target_qbit = qubitnum - gate.get("target_qbit") - 1               
                circuit.append_gate(ZGate(), (target_qbit))

            elif gate.get("type") == "SX":
                # SX gate
                control_qbit = qubitnum - gate.get("control_qbit") - 1               
                target_qbit = qubitnum - gate.get("target_qbit") - 1               
                circuit.append_gate(RZGate(), (target_qbit))
       
  
        return(circuit)


    async def synthesize(
        self,
        utry: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        # Initialize run-dependent options
        
        Umtx = utry.numpy
        from squander import N_Qubit_Decomposition_adaptive
        import numpy as np
        import numpy.linalg as LA
        from squander import utils
        import random
        import math 
        
        qubitnum = math.floor(math.log2(Umtx.shape[0]))


        # use SQAUNDER to decompose bigger than 2-qubit unitaries
        if qubitnum > 2 :

            
            cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, config=self.squander_config, accelerator_num=0 )
            #cDecompose.set_Project_Name( project_name )

            cDecompose.set_Verbose( -1 )

            cDecompose.set_Cost_Function_Variant( 0)
	
	
            # adding decomposing layers to the gate structure
            cDecompose.add_Finalyzing_Layer_To_Gate_Structure()	

	
            # setting intial parameter set
            parameter_num = cDecompose.get_Parameter_Num()
            parameters = np.zeros( (parameter_num,1), dtype=np.float64 )
            cDecompose.set_Optimized_Parameters( parameters )
	
	

            # adding new layer to the decomposition until threshold
            for new_layer in range(5):

		# setting optimizer
                cDecompose.set_Optimizer("AGENTS")

		# starting the decomposition
                cDecompose.get_Initial_Circuit()

		# store parameters after the evolutionary optimization 
		# (After BFGS converge to local minimum, evolutionary optimization wont make it better)
                params_AGENTS = cDecompose.get_Optimized_Parameters()

		# setting optimizer
                cDecompose.set_Optimizer("BFGS")

		# continue the decomposition with a second optimizer method
                cDecompose.get_Initial_Circuit()   

                params_BFGS  = cDecompose.get_Optimized_Parameters()
                decomp_error = cDecompose.Optimization_Problem( params_BFGS )

                if decomp_error <= self.squander_config['optimization_tolerance']:
                     break
                else:
                    # restore parameters to the evolutionary ones since BFGS iterates to local minima
                    cDecompose.set_Optimized_Parameters( params_AGENTS )

                    # add new layer to the decomposition if toleranace was not reached
                    cDecompose.add_Layer_To_Imported_Gate_Structure()



            cDecompose.set_Optimizer("BFGS") 

	    # starting compression iterations
            cDecompose.Compress_Circuit()	


	
	    # finalize the gate structure (replace CRY gates with CNOT gates)
            cDecompose.Finalize_Circuit()	
            
   
            Circuit_squander = self.transform_circuit_from_squander_to_qsearch(cDecompose,qubitnum)          
            dist             = self.cost.calc_cost(Circuit_squander, utry)  

            _logger.debug("The error of the decomposition with SQUANDER is "  + str(dist))            
           
            
            if dist >  self.success_threshold:
                _logger.debug('the squander decomposition error is bigger than the succes_treshold, with the value of:', dist)                
            else: 
                _logger.debug('Successful synthesis with squander.')


            squander_gates = []

            for gate in Circuit_squander.gate_set:
                case_squander = {f"{gate}count:":  Circuit_squander.count(gate)}
                squander_gates.append(case_squander)
 
            print('qbit_num = ', qubitnum, ' dist = ', dist, squander_gates, "\n")
                
            return(Circuit_squander) 

        
        instantiate_options = self.instantiate_options.copy()

        # Seed the PRNG
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        # Get layer generator for search
        layer_gen = self._get_layer_gen(data)

        # Begin the search with an initial layer
        frontier = Frontier(utry, self.heuristic_function)
        initial_layer = layer_gen.gen_initial_layer(utry, data)
        initial_layer.instantiate(utry, **instantiate_options)
        frontier.add(initial_layer, 0)

        # Track best circuit, initially the initial layer
        best_dist = self.cost.calc_cost(initial_layer, utry) 
        best_circ = initial_layer
        best_layer = 0

        # Track partial solutions
        psols: dict[int, list[tuple[Circuit, float]]] = {}

        _logger.debug(f'Search started, initial layer has cost: {best_dist}.')

        # Evalute initial layer
        if best_dist < self.success_threshold:
            _logger.debug('Successful synthesis.')
            return initial_layer

        # Main loop
        while not frontier.empty():
            top_circuit, layer = frontier.pop()

            # Generate successors
            successors = layer_gen.gen_successors(top_circuit, data)

            if len(successors) == 0:
                continue

            # Instantiate successors
            circuits = await get_runtime().map(
                Circuit.instantiate,
                successors,
                target=utry,
                **instantiate_options,
            )

            # Evaluate successors
            for circuit in circuits:
                dist = self.cost.calc_cost(circuit, utry)

                if dist < self.success_threshold: #ifben átírni succes tressholdra
                    _logger.debug('Successful synthesis.')
                    if self.store_partial_solutions:
                        data['psols'] = psols

                    print('qsearch, qbit_num = ', qubitnum, ' dist = ', dist, "\n")
                    return circuit

                if dist < best_dist:
                    plural = '' if layer == 0 else 's'
                    _logger.debug(
                        f'New best circuit found with {layer + 1} '
                        f'layer{plural} and cost: {dist:.12e}.',
                    )
                    best_dist = dist
                    best_circ = circuit
                    best_layer = layer

                if self.store_partial_solutions:
                    if layer not in psols:
                        psols[layer] = []

                    psols[layer].append((circuit.copy(), dist))

                    if len(psols[layer]) > self.partials_per_depth:
                        psols[layer].sort(key=lambda x: x[1])
                        del psols[layer][-1]

                if self.max_layer is None or layer + 1 < self.max_layer:
                    frontier.add(circuit, layer + 1)

        _logger.warning('Frontier emptied.')
        _logger.warning(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (best_layer, '' if best_layer == 1 else 's', best_dist),
        )
        if self.store_partial_solutions:
            data['psols'] = psols

        print('qsearch, qbit_num = ', qubitnum, ' dist = ', dist, "\n")
        return best_circ

    def _get_layer_gen(self, data: PassData) -> LayerGenerator:
        """
        Set the layer generator.

        If a layer generator has been passed into the constructor, then that
        layer generator will be used. Otherwise, a default layer generator will
        be selected by the gateset.

        If seeds are passed into the data dict, then a SeedLayerGenerator will
        wrap the previously selected layer generator.
        """
        # TODO: Deduplicate this code with leap synthesis
        layer_gen = self.layer_gen or data.gate_set.build_mq_layer_generator()

        # Priority given to seeded synthesis
        if 'seed_circuits' in data:
            return SeedLayerGenerator(data['seed_circuits'], layer_gen)

        return layer_gen
