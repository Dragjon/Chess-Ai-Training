# Chess-Ai-Training
 Chess ai training using tensorflow
# Install Jupyter notebook
1. You must have python downloaded, if not, go to the official python downloads page https://www.python.org/downloads/ and select the suitable one for your operating system. You can follow the guide on https://www.geeksforgeeks.org/how-to-install-python-on-windows/ or https://www.geeksforgeeks.org/how-to-download-and-install-python-latest-version-on-linux/
2. Then run `pip install notebook` in your terminal, for windows, it is the Command Prompt.
3. To open the notebook, run `python -m notebook`
# Download
1.  Download all the files in this github repository by clicking *Code* and the *Download zip*
2.  Then unzip the all the files
3.  After that, open terminal and run `pip install -r requirements.txt` to install all required modules and dependencies.
# Playing the default model (morphy)
1. I have already trained the ai model on games played by Paul Morphy
2. To run the ai, navigate into \latest_model\morphy and copy the file path, or you can do it for any other pretrained models there too
3. Then edit *chess_ai.py* using notepad replacing `path_to_model = 'C:/Users/dragon/Documents/Chess-Ai-Training/latest_model/morphy'` with the actual file path you copied
4. Remember to replace the backward slash (\\) with the forward slash (/)
5. Go into the terminal
6. Run `cd [path of the unzipped files]`
7. And run `python chess_ai.py`
8. To play against a lower depth version, which I recommend, run `python chess_ai_lowerdepth.py`
# Playing against other models
1. Navigate into /scripts for normal depth or /scripts-for-lowerdepth for lowerdepth version
2. Then edit *[NAME OF PLAYER].py* using notepad replacing `path_to_model = 'C:/Users/dragon/Documents/Chess-Ai-Training/latest_model/morphy'` with the actual file path you copied
3. Remember to replace the backward slash (\\) with the forward slash (/)
4. Go into the terminal
5. Run `cd [path of the unzipped files]/scripts` or `cd [path of the unzipped files]/scripts-for lowerdepth` 
6. And run `python [NAME OF PLAYER].py`
# Playing against the completed model using google colaboratory
1. Sign in to google colaboratory with your google account in https://colab.research.google.com/
2. Then access this colaboratory link https://colab.research.google.com/drive/1nxjoaTGyaluC8SDsb9VZEeLL5U3hXGP0?usp=sharing
3. And just run all the code cells and you are good!
# Creating a custom ai (Splitting pgn data)
1. Navigate into raw_pgn_data and paste the pgn containing all the games played by a player
2. You can find many games played by Grandmasters here https://www.pgnmentor.com/files.html
3. Then in the terminal, run `python -m notebook` to start jupyter notebook
4. Navigate to the directory of this repository
5. Open *split_pgn_data.ipynb*
6. Replace `input_pgn_file = 'C:/Users/dragon/Documents/Chess-Ai-Training/raw_pgn_data/Morphy.pgn'` with the file path for the pgn file containing your new pgn file.
7. Create the dir *split_pgn_data/[NAME OF NEW PLAYER]* 
8. Replace `output_directory = 'split_pgn_data/morphy'` with `output_directory = 'split_pgn_data/[NAME OF NEW PLAYER]'` for the output after splitting the pgn files. 
9. Then hit shift + enter to run the cell
10. Check in your local file system that there is a new directory called `split_pgn_data/[NAME OF NEW PLAYER]` and it consist of the splitted pgn data and wait for the execution to complete.
# Creating a custom ai (Processing data into csvs)
1. After splitting the pgn data, you will now need to open the *process_data.ipynb* file
2. Similarly, replace this line of code `for dirname, _, filenames in os.walk('C:/Users/dragon/Documents/Chess-Ai-Training/split_pgn_data/morphy'):` with `for dirname, _, filenames in os.walk('[PATH OF THE REPOSITORY]/split_pgn_data/[NEW PLAYER NAME]'):`
3. Create the dir *processed_pgn_data/[NAME OF NEW PLAYER]* 
4. Also replace `new_dirname = 'C:/Users/dragon/Documents/Chess-Ai-Training/processed_pgn_data/morphy'`, which is where the csvs will be located with `new_dirname = '[PATH OF THE REPOSITORY]/processed_pgn_data/[NEW PLAYER NAME]'`
5. Now, you can just hit shift + enter to run the cell
6. Check in your local file system there is a new directory named `[PATH OF THE REPOSITORY]/processed_pgn_data/[new player name]`, note that this execution may take some time, depending on the number of games in your pgn.
# Creating a custom ai (Training time)
1. After finish proccessing the games you can open *engine_train.ipynb* for the REAL training.
2. First, you must run the first code block containing all the import's necessary with shift + enter
3. Then, replace this line `path_csv = 'C:/Users/dragon/Documents/Chess-Ai-Training/processed_pgn_data/morphy'` with `path_csv = '[PATH OF THE REPOSITORY]/processed_pgn_data/[NEW PLAYER NAME]'`
4. Create the dir *estimator/[NAME OF NEW PLAYER]* 
5. Then find this line of code below `linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns, model_dir='C:\\Users\\dragon\\Documents\\Chess-Ai-Training\\estimator\\morphy')` and replace it with `linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns, model_dir='[PATH OF THE REPOSITORY]\\estimator\\[NEW PLAYER NAME]')` !THE DOUBLE BACKWARD SLASH IS IMPORTANT HERE!
6. Then replace the estimator base path `estimator_base_path = 'C:\\Users\\dragon\\Documents\\Chess-Ai-Training\\estimator\\morphy'` with `[PATH OF THE REPOSITORY]\\estimator\\[NEW PLAYER NAME]'`
7. In your file system, you may need to navigate into *estimator* and create a new folder called: [NEW PLAYER NAME]
8. You are all set, run this notebook from start to finish! This is going to take a few hours so why not take a rest. All your progress will be saved in the *estimator/[NEW PLAYER NAME]* folder.
9. After the code runs, you can find your *saved_model.pb* file in one of the folders, eg. 1698915549 in your estimator folder.
10. Now you can move the folder into the *latest_model* folder and change the model folder name to *[NEW PLAYER NAME]*
# Creating a custom ai (Running the custom model)
1. Open the *chess_ai.py* file with notepad
2. Change `path_to_model = 'C:/Users/dragon/Documents/Chess-Ai-Training/latest_model/morphy'` to `path_to_model = '[PATH OF THE REPOSITORY]/latest_model/[NEW PLAYER NAME]'`
3. Save the updated python file
4. Run it with `python chess_ai.py` and play against your newly created ai!
# Games it had played
1. White: Chess Ai trained with Marshall's games at lowerdepth, Black: Me, rated 2000 on chess.com (https://lichess.org/8uxS4c1G/)
2. White: Chess Ai trained with Morphy's games at lowerdepth, Black: Me, rated 2000 on chess.com (https://lichess.org/o3g4hr4T)
3. White: Chess Ai trained with Morphy's games, Black: Me, rated 2000 on chess.com (https://lichess.org/NFxZZP5g)
