### Notes to self

- experiment with reward shaping (easier task) first
- see effect of standardization of observations, rewards (only scaling), prediction targets
- how does parameter sharing affect learning


### Diagnostics

- look at min/max/mean/stdev of episode returns (distribution of returns)
- look at episode lenghts 
- measure parameter updates in the policy (should be stable) [update size in output space and parameter space]
- measure norms of gradients
- policy entropy