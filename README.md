# Portfolio Optimization Using Reinforcement Learning
![image]()

## Table of Contents
- <a href="#business-case">Business Case</a>
- <a href="#requirements">Requirements</a>
- <a href="#installation">Installation</a>
- <a href="#demo-and-usage">Demo and Usage</a>
- <a href="#reinforcement-learning">Reinforcement Learning</a>
- <a href="#dataset">Dataset</a>
- <a href="#feature-engineering">Feature Engineering</a>
- <a href="#modelling">Modelling</a>
- <a href="#model-performance">Model Performance</a>
- <a href="#conclusion">Conclusion</a>

## Business Case
This project aims to solve the problem of **‘given what happened in the market yesterday, what is the best way to allocate my portfolio of assets?’** by leveraging Reinforcement learning. Specifically, the algorithm will learn how best to **buy**, **hold** and **sell** stocks within a portfolio of assets in such a way as to maximise return on investment (ROI).

This approach is a shift from traditional stock market decision-making techniques which focus on predictions. Rather, this approach places emphasis on reacting to prior states of the market.

## Prior Work
This project aims to build upon the work done by [finrl](https://github.com/AI4Finance-Foundation/FinRL).  

## Requirements
This project assumes you already have these system dependencies set up:
- **Python 3.8.5**
- **pip**
- **Conda environment set-up**
- **swig**

Note that swig is a **system dependency** (cannot be installed via pip) that is needed to install and run reinforcement learning algorithms. An installation guide can be found [here](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/).

## Installation
**Swig** must be installed first.

After, all the dependencies are stored in the **requirements.txt** file, running this command in your bash/terminal will install them all. <br>
```bash
pip: -r requirements.txt
```

## Demo and Usage
The project has been summarized in the form of an **investment tool** where users can: <br>
- gather historical pricing data of stocks
- backtest the model over a specified period and compare returns against market return
- predict how best to allocate the portfolio for the next day
![image]()

## Reinforcement Learning
Reinforcement Learning (RL) is a type of machine learning technique that uses a reward/punishment system in order to train a machine. Specifically, the machine acts within a premade environment and learns to take the best actions by maximizing its rewards. The possible actions that are available are determined by the action space whilst the information given to the machine is determined by the observation state (a sample of the environment).
![image]()

<br><br>
The machine is given an observation, takes an action in response to this and creates a new state. That new state is analysed and if it is desirable, then a reward is given. If not, a punishment is given. This sequence repeats until all observations have been met or a reward threshold is reached.
![image]()

## Dataset
- **Portfolio**: DOW Jones Industrial Index of 30 constituents
- **Period**: 2013-04-26 - 2022-04-22
