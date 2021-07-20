# Demonstrations-Efficient Inverse Reinforcement Learning in Procedural Environments
This repository is intended as **Supplementary Materials** to the paper Demonstrations-Efficient Inverse Reinforcement Learning in procedural environments.

<p align="center">
    <img src="https://i.imgur.com/1VuJPrd.gif" width="800">
</p>

You can access the **Appendix** section of the paper [here](Supplementary_Material.pdf).

### Prerequsites

* The code was tested with **Python v3.6**.

* To install all required packages:
    ```
   cd Demonstration-Efficient-AIRL
   pip install -r requirements.txt
    ```  
* To download and install DeepCrawl **Potions** task:
    ```
    python download_envs.py
    sudo chmod +x envs/DeepCrawl-Potions-SeedEnv.x86_64
    sudo chmod +x envs/DeepCrawl-Potions-ProcEnv.x86_64
    ```

## Examples

Running DE-AIRL in PCG on **Potions** task:

1) Train a reward function and a policy in *SeedEnv* with 20 *Seed Levels* 
(expert demonstrations are provided in project files):
    ```
    python demonstrations_script.py -tn=potions -se=20
    ```

2) Train a policy with the learnt reward function in the *ProcEnv*. Tipically
 the best model to use is around ~5000 episodes (later ones tend to overfit to final 
 behavior):
    ```
    python demonstrations_script.py -tn=potions -fr -rm=potions_20_5000
    ```  
------------------------------------------------------------------------

Running DE-AIRL in PCG on **Multiroom** task:

1) Train a reward function and a policy in *SeedEnv* with 40 *Seed Levels* 
(expert demonstrations are provided in project files):
    ``` 
    python demonstrations_script.py -tn=minigrid -se=40
    ```  
2) Train a policy with the learnt reward function in the *ProcEnv*. Tipically the best model 
to use is around ~15000 episodes (later ones tend to overfit to final behavior):
    ```
    python demonstrations_script.py -tn=minigrid -fr -rm=minigrid_40_15000
    ```  










