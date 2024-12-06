Please follow the below instructions to run the file -

1) Create a python virtual environment -

		conda create -n py_env   

2) Activate the virtual environment -

		conda activate py_env

3) Install torch library -
	
		pip3 install torch     ---- on mac
		pip install torch      ---- on windows

4) Install scikit-learn and matplotlib - 

		pip3 install scikit-learn matplotlib transformers plotly optuna

5) Run the BOW model - 

		python3 main.py --model BOW   


6) Run the RoBERTa model - 

		python3 main.py --model RoBERTa 