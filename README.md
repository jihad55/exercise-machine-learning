# exercise-machine-learning
Exercise 2
Your boss was extremely happy with your work on the housing price prediction model and decided to entrust you with a more challenging task. They've seen a lot of people leave the company recently and they would like to understand why that's happening. They have collected historical data on employees and they would like you to build a model that is able to predict which employee will leave next. The would like a model that is better than random guessing. They also prefer false negatives than false positives, in this first phase. Fields in the dataset include:

Employee satisfaction level
Last evaluation
Number of projects
Average monthly hours
Time spent at the company
Whether they have had a work accident
Whether they have had a promotion in the last 5 years
Department
Salary
Whether the employee has left
Your goal is to predict the binary outcome variable left using the rest of the data. Since the outcome is binary, this is a classification problem. Here are some things you may want to try out:

load the dataset at ../data/HR_comma_sep.csv, inspect it with .head(), .info() and .describe().
Establish a benchmark: what would be your accuracy score if you predicted everyone stay?
Check if any feature needs rescaling. You may plot a histogram of the feature to decide which rescaling method is more appropriate.
convert the categorical features into binary dummy columns. You will then have to combine them with the numerical features using pd.concat.
do the usual train/test split with a 20% test size
play around with learning rate and optimizer
check the confusion matrix, precision and recall
check if you still get the same results if you use a 5-Fold cross validation on all the data
Is the model good enough for your boss?

#code
df = pd.read_csv('../data/HR_comma_sep.csv')
df.head()
df.info()
df.describe()
.....



model.fit(X_train, y_train)
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-213-d768f88d541e> in <module>()
----> 1 model.fit(X_train, y_train)

~\Miniconda2\envs\ztdl\lib\site-packages\keras\models.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, **kwargs)
    854                               class_weight=class_weight,
    855                               sample_weight=sample_weight,
--> 856                               initial_epoch=initial_epoch)
    857 
    858     def evaluate(self, x, y, batch_size=32, verbose=1,

~\Miniconda2\envs\ztdl\lib\site-packages\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, **kwargs)
   1427             class_weight=class_weight,
   1428             check_batch_axis=False,
-> 1429             batch_size=batch_size)
   1430         # Prepare validation data.
   1431         if validation_data:

~\Miniconda2\envs\ztdl\lib\site-packages\keras\engine\training.py in _standardize_user_data(self, x, y, sample_weight, class_weight, check_batch_axis, batch_size)
   1303                                     self._feed_input_shapes,
   1304                                     check_batch_axis=False,
-> 1305                                     exception_prefix='input')
   1306         y = _standardize_input_data(y, self._feed_output_names,
   1307                                     output_shapes,

~\Miniconda2\envs\ztdl\lib\site-packages\keras\engine\training.py in _standardize_input_data(data, names, shapes, check_batch_axis, exception_prefix)
    137                             ' to have shape ' + str(shapes[i]) +
    138                             ' but got array with shape ' +
--> 139                             str(array.shape))
    140     return arrays
    141 

ValueError: Error when checking input: expected dense_17_input to have shape (None, 20) but got array with shape (11999, 24)
