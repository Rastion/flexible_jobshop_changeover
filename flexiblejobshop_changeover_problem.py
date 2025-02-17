from qubots.base_problem import BaseProblem
import random
import os

# Constant to denote an incompatible machine.
INFINITE = 1000000

class FlexibleJobShopChangeoverProblem(BaseProblem):
    """
    Flexible Job Shop Scheduling with Changeover Problem for Qubots.
    
    In this problem, each job consists of an ordered sequence of operations.
    Each operation can be processed on one of several compatible machines with a machine-dependent processing time.
    When consecutive operations in the same job are processed on different machines, a changeover time (which depends on the two machines)
    must be added between them. Additionally, each machine can process only one operation at a time.
    
    **Solution Representation:**
      A dictionary mapping each job (0-indexed) to a list of operations.
      Each operation is represented as a dictionary with keys:
        - "machine": the selected machine (0-indexed)
        - "start": the start time of the operation
        - "end": the finish time (which should equal start + processing time on the selected machine)
    """
    
    def __init__(self, instance_file: str):
        (self.nb_jobs,
         self.nb_machines,
         self.nb_tasks,
         self.task_processing_time_data,
         self.job_operation_task,
         self.nb_operations,
         self.max_start,
         self.machine_changeover_time) = self._read_instance(instance_file)
    
    def _read_instance(self, filename: str):
        """
        Reads an instance file with the following format:
        
        - First line: two integers representing the number of jobs and the number of machines (an extra number may be present but is ignored).
        - For each job (next nb_jobs lines):
            * The first integer is the number of operations in that job.
            * Then, for each operation:
                  - An integer indicating the number of compatible machines.
                  - For each compatible machine: a pair of integers (machine id and processing time).
                  (Machine ids are 1-indexed in the file.)
        - For each machine (next nb_machines lines):
            * A line containing nb_machines integers that represent the changeover times from that machine to all machines.
        - A trivial upper bound (max_start) is computed as the sum of the maximum processing times of all operations.
        """

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(filename, 'r') as f:
            lines = f.readlines()
        
        lines = [line.strip() for line in lines if line.strip()]
        
        # Read number of jobs and machines.
        first_line = lines[0].split()
        nb_jobs = int(first_line[0])
        nb_machines = int(first_line[1])
        
        # Read number of operations per job.
        nb_operations = [int(lines[j + 1].split()[0]) for j in range(nb_jobs)]
        
        # Total number of tasks (operations)
        nb_tasks = sum(nb_operations[j] for j in range(nb_jobs))
        
        # Initialize processing times for each task on each machine.
        task_processing_time = [[INFINITE for _ in range(nb_machines)] for _ in range(nb_tasks)]
        
        # Build a mapping: for each job, for each operation, store the corresponding task id.
        job_operation_task = [[0 for _ in range(nb_operations[j])] for j in range(nb_jobs)]
        
        task_id = 0
        for j in range(nb_jobs):
            line_parts = lines[j + 1].split()
            tmp = 1  # Start after the first number (which is nb_operations)
            for o in range(nb_operations[j]):
                nb_machines_op = int(line_parts[tmp])
                tmp += 1
                for i in range(nb_machines_op):
                    # Machine id is provided as 1-indexed in the file.
                    machine = int(line_parts[tmp + 2 * i]) - 1
                    time = int(line_parts[tmp + 2 * i + 1])
                    task_processing_time[task_id][machine] = time
                job_operation_task[j][o] = task_id
                task_id += 1
                tmp += 2 * nb_machines_op
        
        # Read machine changeover times.
        machine_changeover_time = [[0 for _ in range(nb_machines)] for _ in range(nb_machines)]
        for m1 in range(nb_machines):
            parts = lines[nb_jobs + 1 + m1].split()
            for m2 in range(nb_machines):
                machine_changeover_time[m1][m2] = int(parts[m2])
        
        # Compute a trivial upper bound for the start times.
        max_start = 0
        for i in range(nb_tasks):
            valid_times = [t for t in task_processing_time[i] if t != INFINITE]
            max_start += max(valid_times, default=0)
        
        return nb_jobs, nb_machines, nb_tasks, task_processing_time, job_operation_task, nb_operations, max_start, machine_changeover_time
    
    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.
        
        Expects:
          solution: a dictionary mapping each job (0-indexed) to a list of operations.
          Each operation is a dictionary with keys 'machine', 'start', and 'end'.
        
        Returns:
          - The makespan (the maximum completion time over all jobs) if the solution is feasible.
          - A penalty value (1e9) if any constraint is violated.
        """
        penalty = 1e9
        
        # Check that the solution is a dictionary.
        if not isinstance(solution, dict):
            return penalty
        
        # To collect intervals for each machine (for disjunctive constraints).
        machine_intervals = {m: [] for m in range(self.nb_machines)}
        
        overall_end = 0
        
        # Check each job.
        for j in range(self.nb_jobs):
            if j not in solution:
                return penalty
            ops = solution[j]
            if len(ops) != self.nb_operations[j]:
                return penalty
            
            prev_end = None
            prev_machine = None
            
            for o, op in enumerate(ops):
                # Validate operation structure.
                if not isinstance(op, dict):
                    return penalty
                for key in ['machine', 'start', 'end']:
                    if key not in op:
                        return penalty
                
                m = op['machine']
                start = op['start']
                end = op['end']
                
                # Validate machine index.
                if m < 0 or m >= self.nb_machines:
                    return penalty
                
                # Map operation to task id and retrieve processing time.
                task_id = self.job_operation_task[j][o]
                proc_time = self.task_processing_time_data[task_id][m]
                if proc_time == INFINITE:
                    return penalty
                if end != start + proc_time:
                    return penalty
                
                # Check precedence and changeover constraints within the job.
                if prev_end is not None:
                    changeover = self.machine_changeover_time[prev_machine][m]
                    if start < prev_end + changeover:
                        return penalty
                
                prev_end = end
                prev_machine = m
                
                # Record the interval for the machine.
                machine_intervals[m].append((start, end))
                
                # Update overall end.
                overall_end = max(overall_end, end)
        
        # Check disjunctive constraints: on each machine, intervals must not overlap.
        for m in range(self.nb_machines):
            intervals = sorted(machine_intervals[m], key=lambda x: x[0])
            for i in range(len(intervals) - 1):
                if intervals[i][1] > intervals[i + 1][0]:
                    return penalty
        
        return overall_end

    def random_solution(self):
        """
        Generates a random candidate solution.
        
        For each job, operations are scheduled sequentially.
        For each operation, a compatible machine is randomly chosen (processing time ≠ INFINITE),
        and the start time is set to be no earlier than the finish time of the previous operation
        plus the required changeover time. A random slack is added to introduce variability.
        
        Note: This random solution does not resolve conflicts across jobs on the same machine.
        """
        solution = {}
        for j in range(self.nb_jobs):
            ops = []
            current_time = random.randint(0, self.max_start // 4)
            prev_machine = None
            for o in range(self.nb_operations[j]):
                task_id = self.job_operation_task[j][o]
                # Choose among compatible machines.
                compatible = [m for m in range(self.nb_machines) 
                              if self.task_processing_time_data[task_id][m] != INFINITE]
                if not compatible:
                    m = 0
                else:
                    m = random.choice(compatible)
                proc_time = self.task_processing_time_data[task_id][m]
                # Determine changeover if this is not the first operation.
                if prev_machine is not None:
                    changeover = self.machine_changeover_time[prev_machine][m]
                else:
                    changeover = 0
                # Set start time ensuring the changeover requirement is met.
                current_time += changeover + random.randint(0, self.max_start // 10)
                start = current_time
                end = start + proc_time
                ops.append({
                    "machine": m,
                    "start": start,
                    "end": end
                })
                current_time = end
                prev_machine = m
            solution[j] = ops
        return solution
