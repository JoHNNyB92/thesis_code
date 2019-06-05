# General/Useful Information
Implementation for thesis

Currently, the program does the following (with the respective limitations):

1)CSV file(github.csv) contains github url.Right now it has the Aymeric Damian repository that contains multiple tensorflow programs and clones the repository.

2)It parses each file at a time and placed prints and some tensorflow command to extract the pbtxt file and batch/epoch.If a batch_size variable is not identified,it can continue execution without it.
 <i>  
  Limitation:  
  1)Right now it can successfully parse only python tensorflow examples that contain whole code in one file. **Future work  to be adapted for multiple(eg repository contains one neural network not multiple**  
  2)Due to the fact that it is written and tested on windows it handles paths with "\".**Future work  to be adapted  for UNIX path separator**</i>  
 
3)It parses the produced pbtxt file for each neural network and tries to identify the structure.It was tested on some kind of neural networks such as:

 - AutoEncoder
 - Rnn/Dynamic RNN
 - Convolution network
 - Multilayer Perceptron
 - Deep Convolution Generative Adversarial network(2 Networks identified)
 
  *Limitations:  
  1)Unable to parse anything with custom name(parameter name in creation).It creates important structural problems while finding out layers.  
  2)Due to the fact that the only way to identify the networks is to tie the output of each one to a loss function or a metric function it cannot capture abstract networks eg a generative discriminator where only one loss function is provided and no metric function is provided to the generator network.*  
 
3)Whe the parsing of a pbtxt file is done ,a check is performed on whether a loss function was found or at least one layer is present.If everything was successful, the program connects to a virtuoso db and starts the insertion of multiple objects of it. A small log file is produced with the name of the file into log directory along with the multiple calls to rdf wrapper.Currently the insertion to rdf is disabled for debugging purposes,the log file though is created and it contains the calls to rdf.


## Next steps
Now that the program has identified a series of networks , it is wise to proceed with some more github repositories to prepare the program for every probable tensorflow code structure.
