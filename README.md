# Portfolio Optimization Using Reinforcement Learning
![image](https://github.com/AbsIbs/portfolio_optimization/raw/main/images/stocks.png)

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
- <a href="#limitations">Limitations</a>
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

The webapp must be run locally from your computer. To do this, navigate to the project folder using terminal/bash. You can check if you are in the correct folder using,
```bash
pwd
```
The result should display a path that ends in "portfolio_optimization".
<br>
Once there, navigate into the webapp folder.
```bash
cd webapp
```
Once inside, run the following command:
```bash
flask run
```
Note that the command can only be run after flask has been installed from the requirements.txt file. After the code has been run, a message will display in your terminal/bash window 'Running on http://...' e.g.
```bash
* Running on http://123.4.5.6:7890
```
You need to copy everything from http:// to the end and enter that into your browser as a url. In this case, you'd copy **http://123.4.5.6:7890** and place that into your browser.

## Reinforcement Learning
Reinforcement Learning (RL) is a type of machine learning technique that uses a reward/punishment system in order to train a machine. Specifically, the machine acts within a premade environment and learns to take the best actions by maximizing its rewards. The possible actions that are available are determined by the action space whilst the information given to the machine is determined by the observation state (a sample of the environment).

![image](https://github.com/AbsIbs/portfolio_optimization/raw/main/images/rl_1.png)

<br><br>
The machine is given an observation, takes an action in response to this and creates a new state. That new state is analysed and if it is desirable, then a reward is given. If not, a punishment is given. This sequence repeats until all observations have been met or a reward threshold is reached.

![image](https://github.com/AbsIbs/portfolio_optimization/raw/main/images/rl_2_old.png)

## Dataset
- **Portfolio**: DOW Jones Industrial Index of 30 constituents
- **Period**: 2013-04-26 - 2022-04-22

## Feature Engineering
The dataset comprised of the 30 constituents from the DOW Jones Industrial Index. Historical stock price information was gathered through the yahoo finance api. The data was structured as a panel dataset.

The initial features consisted of High, Low, Close, Volume. Technical indicators focusing on trends, momentum and volume were also included e.g., MACD, RSI and MFI.

![image](https://github.com/AbsIbs/portfolio_optimization/raw/main/images/rl_Dataset.png)

## Modelling
### Methodology
![image](https://github.com/AbsIbs/portfolio_optimization/raw/main/images/rl_methodology.png)

Observation space: 252 trading-days covariance matrix based on ROI
- Action space: buy/sell/hold
- Training period: 2013-04-26 to 2019-01-28
- Validation period: 2019-01-29 to 2020-07-07
- Testing period: 2020-07-08 to 2022-04-22
- The machine would trade with an initial capital of $1,000,000.

### Algorithms
>A2C<br>
>PPO<br>
>DDPG<br>
>SAC

## Model Performance
The model results were certainly interesting. For the validation set, the market return was 12.2% whilst our best model achieved 14.1% ROI.

Whilst this may look impressive at first, we must also investigate another key metric, sharpe ratio. This refers to how much ROI is gained for every unit of risk taken.

This is a metric that should be maximized. The market achieved an ROI of 12.2% with a sharpe ratio of around 2.2. In comparison, our best model achieved a 14.1% ROI with a sharpe ratio of around 0.46.

This means that our model took on a significantly greater amount of risk for a 2% ROI difference.

![image](https://github.com/AbsIbs/portfolio_optimization/raw/main/images/model_results.png)

## Limitations
The greatest limitation to this project is the fact that a next day prediction assumed that portfolio is equally distributed i.e. one has an equal amount of holdings for each stock on day 0. This means that, if an investor has a portfolio which does not has equal holdings, they cannot use this feature. A great opportunity would be to allow investors to specify their current holdings.

## Conclusion
To conclude, our RL models do seem useful as they were able to beat the market. However, they did so at a far greater cost.
To increase the sharpe, different neural network architectures for the agent should be experimented with.
Additional technical indicators may also be included along with a longer training period.
At present, this model is only useful for those willing to take on a disproporational amount of risk to beat the market
