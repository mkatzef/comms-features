# Experimental Dataset

The contained configs and samples have been used in experiments on distributed anomaly detection for a paper (under review)

There are four simulation and sampling config files, which each describe a different network layout and a receiver at position `(0, 0, 0)m` in the local network; acting as the origin.

These four networks each have different device configurations as displayed below (in regions 1 through 4).
Anomaly detection experiments can be performed by treating a subset of these regions as "typical" and the remainder as "anomalous". For submitted papers, the divide has been: regions 1-3 as typical, and region 4 as anomalous.

## Region 1
Router MAC: 2e:a6:0c:f2:2c:40
N devices: 10
Comm types: wifi, UDP
Comm patterns: bursty, low

## Region 2
Router MAC: 6b:f7:ed:31:0a:e9
N devices: 20
Comm types: wifi, UDP
Comm patterns: bursty, low, and hawkes low

## Region 3
Router MAC: 31:a5:26:5e:a7:aa
N devices: 15
Comm types: wifi, UDP
Comm patterns: bursty, low, and hawkes high

## Region 4
Router MAC: 50:75:b7:f8:f3:7a
N devices: 5
Comm types: wifi, UDP
Comm patterns: bursty, high, and hawkes high

# PCAP Files
After running the simulator using a pair of (`sim_cfgX.json`, `sampling_cfgX.json`), a new directory `outX/` is created containing a directory `outX/pcaps/` and file `outX/samples.npy`. To reduce the size of this repository, the PCAP data has been omitted.

# Sample Structure
The samples that were collected for each of the above regions consist of the following properties:
0. number of packets  
1. smallest iat  
2. largest iat  
3. mean iat  
4. number of TCP  
5. number of UDP  
6. min packet len  
7. max packet len  
8. mean packet len  
9. number of unique packet lengths  
10. number of unique sources  
11. number of unique destinations  
12. (index 12 to 212) the first 200 readings from the first transmission from that region's router  

The only difference between `sampling_cfg` files is the output file location.

# Sample Usage
After collecting samples from multiple regions, to load these during runtime, each sampling file could be read individually using `numpy.load`, or some preprocessing can occur using `dataset_loader.py`.

The output of `dataset_loader.py` is a standalone file `wireless_ds.npy` that contains all of the source samples, ready to be imported and used in machine learning applications.

# Utils
```
def get_mac():
    return ':'.join('%02x' % random.randint(0, 255) for _ in range(6))
```
