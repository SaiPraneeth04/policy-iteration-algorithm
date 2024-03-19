# POLICY ITERATION ALGORITHM

## AIM
To develop a Python porgram to find the optiml policy for the given MDP using the policy iteration algorithm

## PROBLEM STATEMENT
The situation is to create a python program where a policy is generated with random actions and for that policy we have to perform policy evaluation and policy improvement to find the optimal new policy and state value function and check for the difference between the new policy and the old policy. The combine operation of the policy evaluation and policy improvement is reffered to as policy iteration.

## POLICY ITERATION ALGORITHM
tep 1: Initialize :
Start with an arbitary policy pi(s) for each state s in the environment.

Step 2: Policy Evaluation:
Evaluate the current policy pi(s) by estimating the state-value function vpi(s) or action value Qpi(s,a) which can be achieved by using iterative policy evaluation method. Stop the iteration when the difference is less than the threshold value.

Step 3: Policy Improvement:
Update the policy pi by acting greedily with respect to the value function.Stop the iteration when the previous policy is equal to the current policy

Step 4: Policy Iteration:
Here the algorithm alternates between policy evaluation and policy improvement steps until convergence to an optimal policy is achieved.

Step 5: Display the results:
Final print the new optimal policy and state value function.

## POLICY IMPROVEMENT FUNCTION
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a] += prob*(reward + gamma * V[next_state] * (not done))
          new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return new_pi



pi_2 = policy_improvement(V1, P)
print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)
```

## POLICY ITERATION FUNCTION
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s:{s:a for s, a in enumerate(random_actions)}[s]
    while True:
      old_pi = {s:pi(s) for s in range(len(P))}
      V = policy_evaluation(pi, P, gamma, theta)
      pi= policy_improvement(V, P, gamma)
      if old_pi == {s:pi(s) for s in range(len(P))}:
        break
    return V, pi

optimal_V, optimal_pi = policy_iteration(P)


print('Optimal policy and state-value function (PI):')
print_policy(optimal_pi, P, action_symbols=('<', '>'), n_cols=7)

```
## OUTPUT:
Policy 1:

![output](/2.1.png)

Policy Evaluation:

![output](/2.2.png)

Improved Policy:

![output](/2.3.png)

Value function for improved policy:

![output](/2.4.png)

Optimal Policy and State value function:

![output](/2.5.png)

State Value Function:

![output](/2.6.png)

## RESULT:
Therefore a Python program is successfully developed to find the optimal policy for the given MDP using the policy iteration algorithm.
