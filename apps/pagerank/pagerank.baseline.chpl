use Time;
use Random;

////////////////////////////////////////////////////////////////////////////////
/// Runtime Options
////////////////////////////////////////////////////////////////////////////////
config const nDocs = 32: int;
config const nIters = 1: int;
config const numTrials = 1: int;
config const output = 0: int;
config param verbose = false;

////////////////////////////////////////////////////////////////////////////////
/// Global Arrays
////////////////////////////////////////////////////////////////////////////////
// For now, these arrays are global so the arrays can be seen from CUDAWrapper
// TODO: Explore the possiblity of declaring the arrays and CUDAWrapper
//       in the main proc (e.g., by using lambdas)
var link_counts: [1..nDocs] int(32);
var links: [1..nDocs, 1..2] int(32);
var ranks: [1..nDocs] real(32);
var nLinks = 0: int(32);

////////////////////////////////////////////////////////////////////////////////
/// Utility Functions
////////////////////////////////////////////////////////////////////////////////
proc printResults(execTimes) {
    const totalTime = + reduce execTimes,
	avgTime = totalTime / numTrials,
	minTime = min reduce execTimes;
    writeln("Execution time:");
    writeln("  tot = ", totalTime);
    writeln("  avg = ", avgTime);
    writeln("  min = ", minTime);
}

////////////////////////////////////////////////////////////////////////////////
/// Chapel main
////////////////////////////////////////////////////////////////////////////////
proc main() {
    writeln("PageRank Baseline");
    writeln("nDocs: ", nDocs);
    writeln("nIters: ", nIters);
    writeln("nTrials: ", numTrials);
    writeln("output: ", output);

    var execTimes: [1..numTrials] real;
    for trial in 1..numTrials {
	var random = new NPBRandomStream(eltType=real, seed=23);
	
	for d in 1..nDocs {
	    ranks(d) = 100 * random.getNext(): real(32);
	    link_counts(d) = (10 * random.getNext() + 5): int(32);
	    nLinks += link_counts(d);
	}
	var current_src_doc = 1: int(32);
	var used_links = 0: int(32);
	for l in 1..nLinks {
	    if (used_links == link_counts(current_src_doc)) {
		current_src_doc += 1;
		used_links = 0;
	    }	    
	    links(l, 1) = current_src_doc;
	    links(l, 2) = (random.getNext() * nDocs + 1): int(32);

	    used_links += 1;
	}
	
	const startTime = timeSinceEpoch().totalSeconds();
	var link_weights: [1..nLinks] real(32);	
	for ite in 1..nIters {
	    forall i in 1..nLinks {
		link_weights(i) = (ranks(links(i, 1)) / link_counts(links(i, 1))): real(32);
	    }
	    forall i in 1..nDocs {
		var new_rank = 0: real(32);
		for l in 1..nLinks {
		    if (links(l, 2) == i) {
			new_rank += link_weights(l);
		    }
		}
		ranks(i) = new_rank;
	    }
	}	
	execTimes(trial) = timeSinceEpoch().totalSeconds() - startTime;
	if (output) {
	    writeln(ranks);
	}
    }
    printResults(execTimes);
}
