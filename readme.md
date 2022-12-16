# Seq2Seq ChatBot
 A Sequence-to-Sequence chatbot with Attention mechanism
## HOW TO RUN

1. Clone Repository:
```
$git clone https://github.com/RyanHSL/Seq2SeqChatbot.git
```
2. Set up the local Python 3.x environment in your local machine.
```
https://www.python.org/downloads/
```
3. Install TensorFlow 2.x with pip
```
pip install --upgrade pip

pip list  # show packages installed within the virtual environment

pip install --upgrade tensorflow

python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
4. Preprocessing Data
```
Make sure the training data is in the correct directory then run data_util.py
```
5. Changing the mode and the parameters
```
Change the 'mode' parameter in the parameter.ini to 'train' to trigger the training mode
The model parameters can be modified before the training for tuning.
Change the 'mode' parameter in the parameter.ini to 'serve' to trigger the prediction function.
```