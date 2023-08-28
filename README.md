# Mathematical Finance, Open Markets, and the Rank Jacobi Model
## Introduction
This repository houses the work completed for my Bachelor's thesis in Mathematics, with a strong focus on Mathematical Finance. The primary subject of this research revolves around open market investment frameworks, especially strategies restricted to the top 'N' stocks based on market capitalization.

The research venture has two primary facets: a comprehensive theoretical exploration, followed by empirical testing of derived strategies on historical data. The former delves into the realm of Stochastic Portfolio Theory, investigating the rank Jacobi Model's application in a pure open market environment. The latter facet of the project employs the CRSP data on the WRDS for backtesting the growth optimal strategies obtained through the theoretical exploration, as well as estimating necessary parameters for strategy implementation.

## Thesis Overview
The thesis begins by laying a strong foundation for understanding the domain of Stochastic Portfolio Theory, shedding light on the motivations behind the study and the objectives we set out to achieve. It then progresses towards a detailed literature review and the mathematical framework necessary for the study.

The core of the thesis, however, lies in the 'Theoretical Development' and 'Backtesting' chapters. The former expounds on the complex mathematical models and techniques used throughout the research, focusing on two open market settings and the rank Jacobi model. The 'Backtesting' chapter brings the theoretical exploration to a practical plane, elaborating on the backtesting methodology used, detailing the data cleaning process, parameter estimation, and the computation of optimal strategies.

The concluding part of the thesis summarizes the research findings, reflecting on the research process and potential avenues for future work.

## Repository Structure
This repository is structured into several directories, each serving a specific purpose.

### Thesis
Can be found as `thesis.pdf`.

### Step-by-step Development
`1-step_by_step_development `- Contains Jupyter notebooks (`1-open_market_fixed_universe.ipynb` and `2-open_market_moving_universe.ipynb`) detailing the development of strategies for both fixed and moving universe scenarios.

### Notebooks
`2-notebooks` - Houses Jupyter notebooks for backtesting (`1-backtest.ipynb`), hyperparameter testing (`2-hyperparameter_testing.ipynb`), testing convergence strategies (`3-testing_convergence_strategies.ipynb`), estimating robust growth (`4-estimating_robust_growth.ipynb`), and plotting results for the thesis (`5-plots_report.ipynb`).

### Results
`3-results` - Stores the results of the backtests and hyperparameter testing, as well as images for reporting.

### Source Code
`src` - Contains the Python scripts implementing the core of the investment strategies and backtesting procedures. These scripts include `a_estimation.py`, `backtest.py`, `dataframe_construction.py`, and `optimal_strategies.py`.

## Dependencies
This project makes use of various Python libraries for data handling, mathematical computation, and visualization. The primary dependencies include:

- `numpy` for numerical computation
- `pandas` for data handling and manipulation
- `scipy` for scientific and technical computing
- `matplotlib` and seaborn for data visualization
- `jupyter` for interactive computing and presenting the work in a user-friendly manner.

## Getting Started
To get started with using this code, clone this repository into your local environment. You would need Python installed on your system, preferably via an Anaconda distribution to access all necessary libraries. Navigate to the `src` directory to access the source Python scripts. You will also need to get access to the CRSP database and change the data path according to the location of your data.

If you want to replicate the backtesting, navigate to the `2-notebooks` directory and run the desired Jupyter notebook. Backtesting can be easily carried out with the notebook `1-backtest.ipynb`, adjusting the parameters as desired. Make sure to update the path to the data files to your CRSP monthly or daily data directory.

Should you encounter any issues or need further clarification, please refer to the extensive documentation present in my thesis, included in this repository as `thesis.pdf`. It provides a detailed explanation of the theoretical aspects, the rationale behind the implemented strategies, and the results obtained from empirical testing.

This repository stands as a testament to the significant role that Stochastic Portfolio Theory and advanced mathematical models like the rank Jacobi model can play in modern investment strategy development. It's a step forward in bridging the gap between theoretical finance and practical implementation, offering invaluable insights for both academic research and investment practitioners.

## Contributing
While this project forms the core of my bachelor's thesis, I firmly believe in the value of collaborative development. If you are interested in contributing to this project, please feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements
I would like to express my deep gratitude to my thesis advisor, Dr. David Itkin, for his continuous support, patience, and invaluable guidance throughout the course of this research. I also extend my thanks to the Department of Mathematics at Imperial College London for making possible the development of this project.

Special thanks to the Wharton Research Data Services (WRDS) for granting access to the CRSP dataset, which significantly contributed to the empirical facet of this research. Lastly, my gratitude extends to all those who have indirectly contributed to this work by developing the open-source tools and libraries utilized throughout this project.

