# Brain-machine interfaces: Neural Decoding of Arm Trajectories using LDA-based Classification and Regression
Authors: Nicolas Dehandschoewercker, Sonya Kalsi, Matthieu Pallud, Pranathi Poojary
MATLAB 2024b
Some models may require the Statistics and Machine Learning MATLAB toolbox.

# Best model
Our best model was an LDA classifier with linear regression (RMSE 9.85cm). Parameters can be found in our report and the best_model folder.

# Figures
Figures can be found under figure_plotting. Use the custom test_func script in figure_plotting for thorough analysis.

# How to test your code
1. Run the test script in the simulations folder by entering the name of the folder in the teamName variable
2. The test script will call your training function and then your position estimator function
3. The test script will then calculate the RMSE of your position estimator
4. The test script will then print the RMSE of your position estimator
