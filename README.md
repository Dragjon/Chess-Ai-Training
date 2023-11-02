# Chess-Ai-Training
 Chess ai training using tensorflow
# Install Jupyter notebook
1. You must have python downloaded, if not, go to the official python downloads page https://www.python.org/downloads/ and select the suitable one for your operating system. You can follow the guide on https://www.geeksforgeeks.org/how-to-install-python-on-windows/ or https://www.geeksforgeeks.org/how-to-download-and-install-python-latest-version-on-linux/
2. Then run `pip install notebook` in your terminal, for windows, it is the Command Prompt.
3. To open the notebook, run `python -m notebook`
# Download
1.  Download all the files in this github repository by clicking *Code* and the *Download zip*
2.  Then unzip the all the files
# Running the completed model
1. I have already trained the ai model on games played by Paul Morphy
2. To run the ai, navigate into \latest_model\morphy and copy the file path
3. Then edit chess_ai.py using notepad replacing `path_to_model = 'C:/Users/dragon/Documents/Chess-Ai-Training/latest_model/morphy'` with the actual file path you copied
4. Remember to replace the backward slash (\\) with the forward slash (/)
5. Go into the terminal
6.  Run `cd [path of the unzipped files]`
7.  And run `python chess_ai.py`
