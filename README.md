# Improving Packed-Compressed-Sparse-Row for Efficient Dynamic Graph Analytics (EECS 573)

Dynamic data structure for sparse graphs.

Commands:

g++ -std=c++11 {filename}.cpp -O3 -o a

sudo perf stat -e cache-references,cache-misses ./a

Three implementations:

(1) PCSR_baseline.cpp: Basic version of PCSR without any modifications.

(2) PCSR_cache.cpp: PCSR with cached densities.

(3) PCSR.cpp: PCSR with cached densities and uneven rebalance



```
  // initialize the structure with how many nodes you want it to start with
  PCSR pcsr = PCSR(10);

  // add some edges
  for (int i = 0; i < 5; i++) {
    pcsr.add_edge(i, i, 1);
  }
  // update the values of some edges

  for (int i = 0; i < 5; i++) {
    pcsr.add_edge_update(i, i, 2);
  }

  // print out the graph
  pcsr.print_graph();
```

For more information see https://ieeexplore.ieee.org/abstract/document/8547566

Please cite as:
```
@inproceedings{wheatman2018packed,
  title={Packed Compressed Sparse Row: A Dynamic Graph Representation},
  author={Wheatman, Brian and Xu, Helen},
  booktitle={2018 IEEE High Performance extreme Computing Conference (HPEC)},
  pages={1--7},
  year={2018},
  organization={IEEE}
}
```
