# Cooperative-Search
Multi-agent RL algorithms are applied to large-scale cooperative target searching mission

##### Env:
1. flight_easy: Continuous search environment

![image](https://user-images.githubusercontent.com/55376167/120631146-1d44fc00-c49a-11eb-9272-ee0505c351a3.png)
![image](https://user-images.githubusercontent.com/55376167/120631159-21711980-c49a-11eb-862b-04fa23a5cee8.png)

3. flight: Add probability graph to flight_easy
4. search_env: Discrete search environment
5. simple_spread: A test env

##### Algorithm:
1. QMIX
2. DOP
3. VDN
4. Reinforce

##### Result:
env = flight_easy
target_num = 15
1. agent_num = 1, agent_mode = 0:

![image](https://user-images.githubusercontent.com/55376167/120631264-44033280-c49a-11eb-8947-5ee6d02636be.png)
![image](https://user-images.githubusercontent.com/55376167/120631273-46fe2300-c49a-11eb-90cb-8c11250b27f7.png)

2. agent_num = 3, agent_mode = 0:

![image](https://user-images.githubusercontent.com/55376167/120631354-5da47a00-c49a-11eb-8f4d-33632cf01915.png)
![image](https://user-images.githubusercontent.com/55376167/120631365-6006d400-c49a-11eb-8152-3e1836714e5d.png)

3. agent_num = 5, agent_mode = 0:

![image](https://user-images.githubusercontent.com/55376167/120631407-6d23c300-c49a-11eb-8d79-37db95df8baf.png)
![image](https://user-images.githubusercontent.com/55376167/120631419-70b74a00-c49a-11eb-9710-3e9584446a5f.png)

...

##### Partial code Reference:
1. https://github.com/starry-sky6688/StarCraft
2. https://github.com/TonghanWang/DOP
3. https://github.com/openai/multiagent-particle-envs
