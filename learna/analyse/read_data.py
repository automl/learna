class _SequenceResult(object):
    def __init__(self, sequence_path, run_id):
        self.run = run_id
        self.id = int(sequence_path.name[:-5])
        self.time = float(sequence_path.read_text().rstrip())

    def __iter__(self):
        return iter((self.run, self.id, self.time))

    def __repr__(self):
        return f"(run={self.run}, id={self.id}, time={self.time})"


def _read_data_from_run_path(run_path, timeout):
    run_id = int(run_path.name[4:])
    # Lexicographical sorting has the problem: "20.time" < "3.time"
    sequence_paths = sorted(run_path.glob("*.time"), key=lambda s: int(s.name[:-5]))
    sequence_results = (
        _SequenceResult(sequence_path, run_id) for sequence_path in sequence_paths
    )
    sequence_results = [
        sequence_result
        for sequence_result in sequence_results
        if sequence_result.time <= timeout
    ]
    return sequence_results


def read_data_from_method_path(data_path, timeout):
    def flatten_list(list_):
        return [item for sublist in list_ for item in sublist]

    sequence_results_per_run = [
        _read_data_from_run_path(run_path, timeout)
        for run_path in data_path.glob("run-*")
    ]
    sequence_results = flatten_list(sequence_results_per_run)
    runs, ids, times = zip(*sequence_results)
    return runs, ids, times


def read_sequence_lengths(sequences_dir):
    def get_id(sequence_path):
        return int(sequence_path.name[:-4])

    def get_size(sequence_path):
        return sequence_path.stat().st_size - 1  # \n

    return {
        get_id(sequence): get_size(sequence) for sequence in sequences_dir.glob("*.rna")
    }
